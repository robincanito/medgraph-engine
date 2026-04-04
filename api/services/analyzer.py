"""Analyzer inteligente — Gemini 3.1 Flash Lite como orquestador de queries.
Clasifica la pregunta, detecta entidades, y decide qué capas activar.
No busca en Neo4j — solo analiza la pregunta y devuelve un plan de ejecución.
"""

import json
import re
import os
from google import genai

# Reuse existing preprocessing
try:
    from services.query import preprocess
except ImportError:
    preprocess = None

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GCP_API_KEY", ""))
    return _client


ANALYZER_PROMPT = """Sos un analizador de queries medicas. Dada una pregunta, devolvé SOLO un JSON válido (sin markdown, sin texto extra).

Formato EXACTO (respetá los valores de tipo y buscar_en tal cual):
{{
  "entidades": [
    {{"texto": "...", "tipo": "VALOR_TIPO", "buscar_en": "VALOR_LABEL"}}
  ],
  "intencion": "VALOR_INTENCION",
  "capas": ["CAPAS"],
  "sub_queries": ["reformulacion 1", "reformulacion 2", "reformulacion 3"],
  "ambigua": false,
  "clarificacion": null
}}

Si la pregunta es AMBIGUA o DEMASIADO AMPLIA (ej: "resumen de todo el curso", "todo sobre farmacología", "preparame para el examen"), setear:
  "ambigua": true,
  "clarificacion": {{"pregunta": "pregunta para el usuario", "opciones": ["opcion 1", "opcion 2", "opcion 3"]}}
Igualmente, SIEMPRE generar entidades, sub_queries y capas con la mejor interpretación posible. La clarificación es un COMPLEMENTO, no un reemplazo de la búsqueda.

Valores EXACTOS permitidos para tipo y buscar_en:
- tipo=patologia, buscar_en=Patologia (enfermedades: otitis, neumonía, diabetes, HTA, psoriasis)
- tipo=farmaco, buscar_en=Farmaco (fármacos específicos: furosemida, amoxicilina, ibuprofeno, metformina)
- tipo=clase_farmacologica, buscar_en=CategoriaATC (clases de fármacos: diuréticos, betalactámicos, AINEs, corticoides, opioides)
- tipo=sistema_corporal, buscar_en=CategoriaSNOMED (sistemas: cardiovascular, respiratorio, nervioso, digestivo)
- tipo=anatomia, buscar_en=EstructuraAnatomica (estructuras: oído medio, retina, riñón, hígado, membrana timpánica)
- tipo=procedimiento, buscar_en=Procedimiento (técnicas: otoscopía, ECG, Rx tórax, toma de TA)
- tipo=signo, buscar_en=Signo (signos clínicos: fiebre, edema, soplo, cianosis)
- tipo=sintoma, buscar_en=Sintoma (síntomas: otalgia, cefalea, disnea, dolor torácico)
- tipo=metodo_dx, buscar_en=MetodoDx (métodos diagnósticos: hemograma, ecografía, audiometría)
- tipo=actividad_academica, buscar_en=Actividad (TPs, seminarios, talleres, guías, contenidos UP)

Valores para intencion: tratamiento|diagnostico|anatomia|fisiologia|clinica|etiologia|clasificacion|procedimiento|actividad|bibliografia|general

Valores para capas: ONTOLOGY, GRAPH, BIBLIOGRAPHY, ACTIVITIES, DAGS

Reglas para decidir capas:
- BIBLIOGRAPHY y ACTIVITIES siempre incluir
- Si hay patologia, farmaco, clase_farmacologica o sistema_corporal → agregar ONTOLOGY y GRAPH
- Si hay signo, sintoma, procedimiento, metodo_dx o anatomia → agregar ONTOLOGY y GRAPH (procedimientos, signos y anatomía también están clasificados en SNOMED)
- Si pregunta "qué hago ante", "cómo manejar", "paciente con" → agregar DAGS
- Si pregunta "fisiopatología", "mecanismo", "por qué se produce" → agregar DAGS
- Si pregunta "clasificación", "qué tipo", "a qué clase pertenece" → agregar ONTOLOGY

sub_queries: genera 3 reformulaciones especializadas para búsqueda en textos médicos, cubriendo diferentes aspectos del tema (definición, clínica, tratamiento, etc.)

EJEMPLOS:

Q: "qué diuréticos se usan en insuficiencia cardíaca"
{{"entidades":[{{"texto":"diuréticos","tipo":"clase_farmacologica","buscar_en":"CategoriaATC"}},{{"texto":"insuficiencia cardíaca","tipo":"patologia","buscar_en":"Patologia"}}],"intencion":"tratamiento","capas":["ONTOLOGY","GRAPH","BIBLIOGRAPHY","ACTIVITIES"],"sub_queries":["diuréticos tratamiento insuficiencia cardíaca","furosemida hidroclorotiazida espironolactona IC dosis","insuficiencia cardíaca manejo farmacológico guías"]}}

Q: "preparame para el TP de otoscopía"
{{"entidades":[{{"texto":"TP otoscopía","tipo":"actividad_academica","buscar_en":"Actividad"}},{{"texto":"otoscopía","tipo":"procedimiento","buscar_en":"Procedimiento"}}],"intencion":"procedimiento","capas":["ACTIVITIES","GRAPH","ONTOLOGY","BIBLIOGRAPHY"],"sub_queries":["otoscopía técnica pasos procedimiento","guía TP otoscopía lista cotejo","membrana timpánica hallazgos normales patológicos otoscopía"]}}

Q: "paciente con otalgia y fiebre qué hago"
{{"entidades":[{{"texto":"otalgia","tipo":"sintoma","buscar_en":"Sintoma"}},{{"texto":"fiebre","tipo":"signo","buscar_en":"Signo"}}],"intencion":"clinica","capas":["DAGS","GRAPH","ONTOLOGY","BIBLIOGRAPHY","ACTIVITIES"],"sub_queries":["otalgia fiebre diagnóstico diferencial","otitis media aguda diagnóstico tratamiento","manejo clínico otalgia aguda evaluación otoscópica"]}}

Q: "contenidos unidad 1"
{{"entidades":[{{"texto":"unidad 1","tipo":"actividad_academica","buscar_en":"Actividad"}},{{"texto":"contenidos","tipo":"actividad_academica","buscar_en":"Actividad"}}],"intencion":"actividad","capas":["ACTIVITIES","BIBLIOGRAPHY"],"sub_queries":["contenidos unidad 1 temas unidad","guía contenidos first unit","unit 1 core topics fundamentals introduction"],"ambigua":false,"clarificacion":null}}

Q: "resumen completo del curso"
{{"entidades":[{{"texto":"course","tipo":"actividad_academica","buscar_en":"Actividad"}}],"intencion":"actividad","capas":["ACTIVITIES","BIBLIOGRAPHY"],"sub_queries":["contenidos curso temas unidades","guía contenidos del curso"],"ambigua":true,"clarificacion":{{"pregunta":"The course has multiple units. ¿Sobre cuál querés el resumen?","opciones":["Unit 1: Introduction and fundamentals","Unit 2: Core concepts","Unit 3: Advanced topics","Unit 4: Applied knowledge","All units"]}}}}

Pregunta: {pregunta}"""


# Normalización post-Gemini: mapea tipos genéricos a los del sistema
_TYPE_NORMALIZE = {
    # Fármacos
    "medicamento": ("farmaco", "Farmaco"),
    "medicina": ("farmaco", "Farmaco"),
    "droga": ("farmaco", "Farmaco"),
    "drug": ("farmaco", "Farmaco"),
    "farmaco": ("farmaco", "Farmaco"),
    "fármaco": ("farmaco", "Farmaco"),
    # Clases farmacológicas
    "clase de farmaco": ("clase_farmacologica", "CategoriaATC"),
    "clase farmacologica": ("clase_farmacologica", "CategoriaATC"),
    "grupo farmacologico": ("clase_farmacologica", "CategoriaATC"),
    "clase_farmacologica": ("clase_farmacologica", "CategoriaATC"),
    # Patologías
    "enfermedad": ("patologia", "Patologia"),
    "patologia": ("patologia", "Patologia"),
    "patología": ("patologia", "Patologia"),
    "condicion": ("patologia", "Patologia"),
    "trastorno": ("patologia", "Patologia"),
    "sindrome": ("patologia", "Patologia"),
    "disease": ("patologia", "Patologia"),
    # Anatomía
    "organo": ("anatomia", "EstructuraAnatomica"),
    "órgano": ("anatomia", "EstructuraAnatomica"),
    "estructura": ("anatomia", "EstructuraAnatomica"),
    "anatomia": ("anatomia", "EstructuraAnatomica"),
    "anatomía": ("anatomia", "EstructuraAnatomica"),
    "estructura_anatomica": ("anatomia", "EstructuraAnatomica"),
    # Sistemas
    "sistema": ("sistema_corporal", "CategoriaSNOMED"),
    "sistema_corporal": ("sistema_corporal", "CategoriaSNOMED"),
    "aparato": ("sistema_corporal", "CategoriaSNOMED"),
    # Procedimientos
    "procedimiento": ("procedimiento", "Procedimiento"),
    "tecnica": ("procedimiento", "Procedimiento"),
    "estudio": ("procedimiento", "Procedimiento"),
    # Signos y síntomas
    "signo": ("signo", "Signo"),
    "sintoma": ("sintoma", "Sintoma"),
    "síntoma": ("sintoma", "Sintoma"),
    "hallazgo": ("signo", "Signo"),
    # Diagnóstico
    "metodo_dx": ("metodo_dx", "MetodoDx"),
    "metodo diagnostico": ("metodo_dx", "MetodoDx"),
    "prueba": ("metodo_dx", "MetodoDx"),
    "test": ("metodo_dx", "MetodoDx"),
    # Actividades
    "actividad_academica": ("actividad_academica", "Actividad"),
    "actividad": ("actividad_academica", "Actividad"),
    "tp": ("actividad_academica", "Actividad"),
    "documento": ("actividad_academica", "Actividad"),
    # Tipos genéricos que Gemini inventa
    "agente": ("patologia", "Agente"),
    "agente_infeccioso": ("patologia", "Agente"),
    "microorganismo": ("patologia", "Agente"),
    "virus": ("patologia", "Agente"),
    "bacteria": ("patologia", "Agente"),
    "parasito": ("patologia", "Agente"),
    "parametro": ("metodo_dx", "Parametro"),
    "valor": ("metodo_dx", "Parametro"),
    "laboratorio": ("metodo_dx", "MetodoDx"),
    "hallazgo": ("signo", "Hallazgo"),
    "grupo_farmacologico": ("clase_farmacologica", "GrupoFarmacologico"),
    "clase": ("clase_farmacologica", "CategoriaATC"),
    "categoria": ("clase_farmacologica", "CategoriaATC"),
    "tratamiento": ("farmaco", "Farmaco"),
    "cirugia": ("procedimiento", "Procedimiento"),
    "cirugía": ("procedimiento", "Procedimiento"),
    "imagen": ("metodo_dx", "MetodoDx"),
    "radiografia": ("metodo_dx", "MetodoDx"),
    "ecografia": ("metodo_dx", "MetodoDx"),
    "analisis": ("metodo_dx", "MetodoDx"),
}

_BUSCAR_EN_NORMALIZE = {
    "medicamentos": "Farmaco",
    "enfermedades": "Patologia",
    "farmacos": "Farmaco",
    "organos": "EstructuraAnatomica",
    "estructuras": "EstructuraAnatomica",
    "procedimientos": "Procedimiento",
    "signos": "Signo",
    "sintomas": "Sintoma",
    "sistemas": "CategoriaSNOMED",
    "actividades": "Actividad",
    "patologias": "Patologia",
}


def _normalize_entities(entities: list) -> list:
    """Normaliza tipos genéricos de Gemini a los labels exactos del sistema."""
    for ent in entities:
        tipo = ent.get("tipo", "").lower().strip()
        buscar = ent.get("buscar_en", "").strip()

        # Normalizar tipo
        if tipo in _TYPE_NORMALIZE:
            ent["tipo"], ent["buscar_en"] = _TYPE_NORMALIZE[tipo]

        # Normalizar buscar_en si Gemini devolvió algo genérico
        buscar_lower = buscar.lower()
        if buscar_lower in _BUSCAR_EN_NORMALIZE:
            ent["buscar_en"] = _BUSCAR_EN_NORMALIZE[buscar_lower]

    return entities


def analyze_query(pregunta: str) -> dict:
    """Analiza una pregunta médica y devuelve plan de ejecución."""

    # 1. Preprocessing existente (sinónimos, intención básica)
    pre = None
    if preprocess:
        try:
            pre = preprocess(pregunta)
        except Exception:
            pass

    expandida = pre.get("expandida", pregunta) if pre else pregunta

    # 2. Gemini clasifica via Google GenAI
    client = _get_client()
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=ANALYZER_PROMPT.format(pregunta=pregunta),
            config={
                "temperature": 0.1,
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
        )
        raw_text = resp.text

        # Log raw for debugging
        import logging
        logging.info(f"Gemini raw ({len(raw_text)} chars): {raw_text[:200]}")

        # Limpiar markdown
        text = raw_text.strip()
        if "```" in text:
            text = re.sub(r'```\w*\n?', '', text)
            text = text.strip()

        # Extraer JSON: buscar primer { y último }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            text = text[first_brace:last_brace+1]

        result = json.loads(text)

        # Normalizar keys (Gemini puede usar nombres diferentes)
        if "entidades" not in result:
            # Buscar keys alternativas
            for alt_key in ["entities", "entidad", "detected_entities"]:
                if alt_key in result:
                    result["entidades"] = result.pop(alt_key)
                    break
            else:
                result["entidades"] = []

        if "capas" not in result:
            for alt_key in ["layers", "capa", "activated_layers"]:
                if alt_key in result:
                    result["capas"] = result.pop(alt_key)
                    break
            else:
                result["capas"] = ["BIBLIOGRAPHY"]

        if "intencion" not in result:
            for alt_key in ["intent", "intention"]:
                if alt_key in result:
                    result["intencion"] = result.pop(alt_key)
                    break
            else:
                result["intencion"] = "general"

        if "sub_queries" not in result:
            result["sub_queries"] = [pregunta]

        # Normalizar tipos genéricos de Gemini a labels del sistema
        if result.get("entidades"):
            result["entidades"] = _normalize_entities(result["entidades"])

    except Exception as e:
        # Log the error AND the raw response for debugging
        import logging
        try:
            raw = response.text[:300] if 'response' in dir() and hasattr(response, 'text') else 'no response'
        except:
            raw = 'no response'
        logging.error(f"Gemini analyzer failed: {type(e).__name__}: {str(e)[:200]} | Raw: {raw}")
        # Fallback determinista
        result = _fallback_analysis(pregunta, pre)

    # 3. Combinar con preprocessing existente
    analysis = {
        "original": pregunta,
        "intencion": result.get("intencion", "general"),
        "expandida": expandida,
        "sub_queries": result.get("sub_queries", [pregunta]),
        "entidades_detectadas": result.get("entidades", []),
        "capas": result.get("capas", ["BIBLIOGRAPHY"]),
    }

    # Propagar clarificación si Gemini detectó ambigüedad
    if result.get("ambigua"):
        analysis["ambigua"] = True
        analysis["clarificacion"] = result.get("clarificacion")

    # BIBLIOGRAPHY y ACTIVITIES siempre activas (somos un sistema para estudiantes)
    for capa_base in ["BIBLIOGRAPHY", "ACTIVITIES"]:
        if capa_base not in analysis["capas"]:
            analysis["capas"].append(capa_base)

    return analysis


def _fallback_analysis(pregunta: str, pre: dict = None) -> dict:
    """Análisis determinista como fallback si Gemini falla."""
    pregunta_lower = pregunta.lower()

    capas = ["BIBLIOGRAPHY", "ACTIVITIES"]
    entidades = []
    intencion = pre.get("intencion", "general") if pre else "general"

    # Detectar actividades
    activity_keywords = ["tp", "seminario", "taller", "acreditación", "acreditacion", "guía", "guia", "lista de cotejo"]
    if any(kw in pregunta_lower for kw in activity_keywords):
        capas.append("ACTIVITIES")
        entidades.append({"texto": pregunta, "tipo": "actividad_academica", "buscar_en": "Actividad"})

    # Detectar intención clínica
    clinical_keywords = ["qué hago", "que hago", "cómo manejar", "como manejar", "ante un paciente", "manejo de"]
    if any(kw in pregunta_lower for kw in clinical_keywords):
        capas.append("DAGS")

    # Detectar intención ontológica
    onto_keywords = ["clasificación", "clasificacion", "qué tipo", "que tipo", "a qué clase", "pertenece"]
    if any(kw in pregunta_lower for kw in onto_keywords):
        capas.append("ONTOLOGY")

    # Default: siempre grafo
    if "GRAPH" not in capas:
        capas.append("GRAPH")

    return {
        "entidades": entidades,
        "intencion": intencion,
        "capas": list(set(capas)),
        "sub_queries": [pregunta],
    }
