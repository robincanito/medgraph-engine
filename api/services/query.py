"""Query preprocessing: clasificación de intención, expansión de sinónimos y decomposición."""

import re

# Sinónimos médicos comunes (abreviatura → expansiones)
SINONIMOS = {
    # Cardiología
    "hta": ["hipertension arterial"],
    "iam": ["infarto agudo de miocardio", "infarto"],
    "icc": ["insuficiencia cardiaca congestiva", "insuficiencia cardiaca"],
    "ecg": ["electrocardiograma"],
    "fa": ["fibrilacion auricular"],
    "tev": ["tromboembolismo venoso"],
    "tep": ["tromboembolismo pulmonar", "embolia pulmonar"],
    "tvp": ["trombosis venosa profunda"],

    # Neumología
    "epoc": ["enfermedad pulmonar obstructiva cronica"],
    "sdra": ["sindrome de dificultad respiratoria aguda"],
    "nac": ["neumonia adquirida en la comunidad"],
    "rx": ["radiografia"],

    # Endocrinología
    "dbt": ["diabetes", "diabetes mellitus"],
    "dm": ["diabetes mellitus"],
    "dm2": ["diabetes mellitus tipo 2"],
    "dm1": ["diabetes mellitus tipo 1"],
    "tsh": ["tirotropina", "hormona estimulante de tiroides"],

    # Infectología
    "hiv": ["virus de inmunodeficiencia humana", "vih", "sida"],
    "hbv": ["hepatitis b", "virus hepatitis b"],
    "hcv": ["hepatitis c", "virus hepatitis c"],
    "tbc": ["tuberculosis"],
    "egb": ["estreptococo grupo b"],
    "its": ["infeccion de transmision sexual"],
    "itu": ["infeccion del tracto urinario", "infeccion urinaria"],

    # Nefrología
    "irc": ["insuficiencia renal cronica"],
    "ira": ["insuficiencia renal aguda"],
    "tfg": ["tasa de filtracion glomerular"],

    # Gastroenterología
    "eii": ["enfermedad inflamatoria intestinal"],
    "rge": ["reflujo gastroesofagico"],

    # Neurología
    "acv": ["accidente cerebrovascular", "stroke"],
    "ait": ["accidente isquemico transitorio"],
    "lcr": ["liquido cefalorraquideo"],

    # Pediatría / Neonatología
    "rn": ["recien nacido"],
    "rnpt": ["recien nacido pretermino"],
    "rnt": ["recien nacido de termino"],
    "bpn": ["bajo peso al nacer"],
    "apgar": ["apgar"],
    "ehrn": ["enfermedad hemorragica del recien nacido"],

    # Oftalmología
    "av": ["agudeza visual"],
    "cv": ["campo visual"],
    "pio": ["presion intraocular"],
    "dpar": ["defecto pupilar aferente relativo"],

    # ORL
    "oma": ["otitis media aguda"],
    "ome": ["otitis media con efusion", "otitis media secretora"],
    "cae": ["conducto auditivo externo"],

    # Dermatología
    "da": ["dermatitis atopica", "eccema atopico"],

    # Farmacología
    "aine": ["antiinflamatorio no esteroideo", "antiinflamatorios no esteroideos"],
    "atb": ["antibiotico", "antibioticos"],
    "vo": ["via oral"],
    "im": ["intramuscular"],
    "iv": ["intravenoso", "endovenoso"],
    "sc": ["subcutaneo"],
    "ev": ["endovenoso", "intravenoso"],

    # Sinónimos anatómicos
    "papila optica": ["disco optico", "cabeza del nervio optico"],
    "disco optico": ["papila optica", "cabeza del nervio optico"],
    "mt": ["membrana timpanica", "timpano"],
    "membrana timpanica": ["timpano"],
}

# Patrones para clasificar intención
INTENCION_PATTERNS = {
    "tratamiento": [
        r"tratamiento\b", r"terapia\b", r"como se trata",
        r"farmaco", r"medicament", r"dosis", r"posologia",
        r"primera linea", r"segunda linea", r"se trata con",
    ],
    "diagnostico": [
        r"diagnostico\b", r"como se diagnostica", r"metodo dx",
        r"criterios", r"laboratorio", r"imagen", r"ecografia",
        r"estudios complementarios", r"diferencial",
    ],
    "anatomia": [
        r"anatomia\b", r"histologia\b", r"estructura",
        r"ubicacion", r"donde se encuentra", r"partes de",
        r"capas", r"tunica", r"musculo", r"nervio",
    ],
    "fisiologia": [
        r"fisiologia\b", r"mecanismo", r"funcion de",
        r"como funciona", r"transduccion", r"via",
    ],
    "clinica": [
        r"clinica\b", r"sintomas", r"signos", r"manifestacion",
        r"cuadro clinico", r"presenta con", r"cursa con",
    ],
    "etiologia": [
        r"etiologia\b", r"causa\b", r"causas\b", r"agente",
        r"patogenia", r"fisiopatologia", r"por que se produce",
    ],
    "epidemiologia": [
        r"epidemiologia\b", r"prevalencia", r"incidencia",
        r"frecuencia", r"factores de riesgo",
    ],
    "procedimiento": [
        r"procedimiento\b", r"tecnica\b", r"como se hace",
        r"como se realiza", r"pasos", r"lista de cotejo",
    ],
}


def expandir_sinonimos(query_text: str) -> str:
    """Expande abreviaturas y sinónimos médicos en la query."""
    words = query_text.lower().split()
    expanded = list(words)

    for i, word in enumerate(words):
        clean = word.strip(".,;:?!()")
        if clean in SINONIMOS:
            # Agregar sinónimos al final
            for sin in SINONIMOS[clean]:
                expanded.append(sin)

    # También buscar frases de 2 palabras
    text_lower = query_text.lower()
    for phrase, sins in SINONIMOS.items():
        if " " in phrase and phrase in text_lower:
            for sin in sins:
                expanded.append(sin)

    return " ".join(expanded)


def clasificar_intencion(query_text: str) -> str:
    """Clasifica la intención de la query médica."""
    text_lower = query_text.lower()

    scores = {}
    for intencion, patterns in INTENCION_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += 1
        if score > 0:
            scores[intencion] = score

    if not scores:
        return "general"

    return max(scores, key=scores.get)


def descomponer_query(query_text: str, intencion: str) -> list[str]:
    """Descompone una query compleja en sub-queries más específicas."""

    # Para queries simples (1-3 palabras), no descomponer
    words = query_text.strip().split()
    if len(words) <= 3:
        return [query_text]

    # Para queries de diagnóstico diferencial
    if "diferencial" in query_text.lower() or " vs " in query_text.lower():
        # Extraer los dos términos
        parts = re.split(r"\bvs\b|\bdiferencial\b|\bentre\b", query_text.lower())
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return [f"{parts[0]} definicion clinica", f"{parts[1]} definicion clinica",
                    f"{parts[0]} {parts[1]} diferencias"]
        return [query_text]

    # Para queries largas (>6 palabras), mantener original + versión condensada
    if len(words) > 6:
        # Mantener query original + versión con solo sustantivos médicos (quitar preposiciones)
        stop_words = {"de", "del", "la", "el", "los", "las", "un", "una", "y", "o", "en", "con", "por", "para", "que", "como", "se"}
        filtered = [w for w in words if w.lower() not in stop_words]
        if len(filtered) >= 2:
            return [query_text, " ".join(filtered)]

    return [query_text]


def preprocess(query_text: str) -> dict:
    """Pipeline completo de preprocesamiento de query.

    Returns:
        {
            "original": "tratamiento de HTA",
            "intencion": "tratamiento",
            "expandida": "tratamiento de HTA hipertension arterial",
            "sub_queries": ["tratamiento de HTA hipertension arterial"],
        }
    """
    intencion = clasificar_intencion(query_text)
    expandida = expandir_sinonimos(query_text)
    sub_queries = descomponer_query(expandida, intencion)

    return {
        "original": query_text,
        "intencion": intencion,
        "expandida": expandida,
        "sub_queries": sub_queries,
    }
