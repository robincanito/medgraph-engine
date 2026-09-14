"""LA TAXONOMIA DE EXTRACCION DE UN DOMINIO: el prompt, el validador, la canonicalizacion
y la carga. Una copia, como `parseo`/`carga`/`embeddings`, replicada byte a byte al engine.

POR QUE EXISTE (14-sep-2026, decision E). Hasta hoy `profiles/<dominio>.yaml` declaraba once
tipos de entidad, trece de relacion con sus `from`/`to`, un `prompt_template`, unas `rules` y
una seccion `canonicalization`, y NINGUNA de esas cinco cosas la leia una linea de codigo. La
taxonomia que de verdad regia estaba escrita CINCO VECES a mano en el repo vivo:

  1. `extract_entities.ENTITY_TYPES`      — el set del validador
  2. `extract_entities.RELATION_TYPES`    — otro set, con CUATRO relaciones que el prompt no
                                            pedia y el perfil no publicaba (las huerfanas de
                                            la etapa de anatomia)
  3. `extract_entities.TYPE_TO_LABEL`     — tipo -> label de Neo4j, con fallback "Entidad",
                                            un label que no esta en ninguna taxonomia
  4. `extract_entities.EXTRACTION_PROMPT` — la taxonomia con descripciones, en prosa
  5. `extract_entities_fast.BATCH_PROMPT` — la taxonomia plana, y es la que se usa

Coincidian por casualidad en los tipos y diferian en las relaciones. El sintoma aguas abajo
esta escrito en `api/services/analyzer.py`: un mapa de "tipos genericos que Gemini inventa"
(`agente_infeccioso`, `virus`, `bacteria`…) en la capa de CONSULTA, para traducir lo que el
extractor no supo restringir.

LO QUE ESTE MODULO GARANTIZA, Y ES LO UNICO QUE IMPORTA DE LA PRIMERA TANDA:

    armar_prompt(taxonomia(perfiles.cargar("medicina")), chunks, titulo)
        ==  el prompt que extrajo las 159K entidades del grafo          # BYTE A BYTE

Lo fija `tests/test_pipeline_extraccion.py` contra un golden congelado ANTES de tocar el
codigo (`tests/golden/extraccion/prompt_medicina_v1.txt`). Si el render difiere en una coma,
el proximo libro se extrae con otro criterio y el grafo mezcla dos cosechas sin que nada lo
diga.

FALLA CERRADO. `taxonomia()` valida al construirse, con la misma doctrina que
`estrategia._validar`: todo `from`/`to` referencia ids declarados, los ids matchean su forma,
los labels son unicos, `min_nombre < max_nombre`, la plantilla existe y sus marcadores son
conocidos. Un `from: [patologa]` con un typo hoy no falla NUNCA —porque nadie lee el campo—;
desde hoy falla ANTES de pagar una llamada.

SIN `print`: este paquete se replica al engine y lo vigila `test_bitacora.PAQUETE_SIN_PRINT`.
Lo que hay que contar viaja por `pipeline.eventos`.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from pipeline import eventos

log = logging.getLogger(__name__)

#: Marca de linaje, hermana de `embeddings.EMBEDDING_FORMA`: de donde salio la taxonomia con
#: que se extrajo. Lo que se escriba en el artefacto con este valor se puede comparar; lo que
#: quedo del codigo viejo no lleva marca y por eso se sabe que es de antes.
TAXONOMIA_FORMA = "perfil"

#: Las etiquetas del CONTEXTO de cada fragmento. SIN TILDES, y es a proposito: asi viajaron en
#: el prompt que produjo el corpus. El prefijo de EMBEDDING usa las mismas tres palabras CON
#: tilde ("Capítulo: … Sección: …", `pipeline/embeddings.py`). Son dos convenciones distintas
#: para la misma metadata, las dos en produccion, ninguna equivocada: el que venga que no
#: "arregle" una de las dos sin leer esto.
ETIQUETA_LIBRO, ETIQUETA_CAPITULO, ETIQUETA_SECCION = "Libro", "Capitulo", "Seccion"

#: Valores que rigen cuando el perfil no los declara. Son los literales que estaban en el
#: codigo, asi que un perfil sin estas claves se comporta como antes de que existieran.
MAX_CHARS_FRAGMENTO = 2500
MIN_NOMBRE, MAX_NOMBRE = 2, 100
MAX_MENCIONES = 10
BATCH_SIZE, MIN_WORDS = 3, 100

#: EL FLAG DE POLITICA (decision 3 de §8 del diseño). Con `False` una relacion que viola los
#: `from`/`to` del perfil se CUENTA y se emite, pero ENTRA. Medicina ya pago una vez el precio
#: de reglas mal escritas: la primera version de los `from`/`to` marcaba 71.569 relaciones como
#: invalidas y la mayoria eran reglas malas, no datos malos (`medicina.yaml`). Darlo vuelta
#: cambia lo que entra al grafo, asi que espera la evidencia del juez mas una pasada real.
RECHAZO_FROM_TO = False

#: Los motivos de descarte que viajan en el evento `extraccion_descarte`.
MOTIVOS = ("tipo_entidad", "tipo_relacion", "nombre_corto", "nombre_largo", "from_to",
           "extremo_ausente")

_ID_ENTIDAD = re.compile(r"^[a-z][a-z0-9_]*$")
_ID_RELACION = re.compile(r"^[A-Z][A-Z0-9_]*$")
#: Un marcador es `{` + minusculas/guion bajo + `}`. La llave de un ejemplo de JSON
#: (`{"resultados"…`) NO matchea, que es lo que permite escribir la plantilla con llaves
#: simples y legibles en vez de duplicadas como exigia el `str.format` del codigo viejo.
_MARCADOR = re.compile(r"\{([a-z][a-z0-9_]*)\}")
MARCADORES = ("n", "fragments", "entities", "entities_desc", "relations", "relations_desc",
              "rules", "libro")


# ══════════════════════════════════════════════════════════════════════════════════════
# LA TAXONOMIA
# ══════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Relacion:
    """Un tipo de relacion del perfil. `extraer=False` = valida para el grafo y publicada en
    el descriptor, pero FUERA del prompt (las cuatro huerfanas de anatomia de medicina)."""

    id: str
    desde: frozenset
    hasta: frozenset
    extraer: bool = True


@dataclass(frozen=True)
class Taxonomia:
    """Lo que un dominio le pide al modelo y lo que acepta de vuelta. Inmutable: se construye
    una vez por corrida, como `Estrategia`."""

    dominio: str
    version: int
    #: id -> {"label": str, "desc": str}, EN EL ORDEN DEL YAML (el prompt lo respeta).
    tipos: dict = field(default_factory=dict)
    #: id -> Relacion, en el orden del YAML.
    relaciones: dict = field(default_factory=dict)
    name_property: str = "nombre"
    modelo: str = ""
    batch: int = BATCH_SIZE
    min_words: int = MIN_WORDS
    max_chars_fragmento: int = MAX_CHARS_FRAGMENTO
    reglas: str = ""
    plantilla: str = ""
    min_nombre: int = MIN_NOMBRE
    max_nombre: int = MAX_NOMBRE
    fundir_sinonimos: bool = True
    plegar_acentos: bool = False
    max_menciones: int = MAX_MENCIONES

    # ── lecturas ──────────────────────────────────────────────────────────────────────
    @property
    def perfil(self) -> str:
        """`medicina@2`, tal como viaja en el linaje y en el descriptor de admin/v1."""
        return f"{self.dominio}@{self.version}"

    @property
    def ids_entidad(self) -> frozenset:
        return frozenset(self.tipos)

    @property
    def ids_relacion(self) -> frozenset:
        """TODAS, huerfanas incluidas: el validador acepta lo que el grafo puede tener."""
        return frozenset(self.relaciones)

    @property
    def extraibles(self) -> tuple:
        """Solo las que el prompt pide, en el orden del YAML."""
        return tuple(r.id for r in self.relaciones.values() if r.extraer)

    def label_de(self, tipo: str) -> str | None:
        """El label de Neo4j de un tipo, o None. **None y no "Entidad"**: el fallback del
        codigo viejo escribia un label que no esta en ninguna taxonomia ni en ningun
        constraint, o sea nodos que ninguna consulta encuentra."""
        entrada = self.tipos.get(tipo)
        return entrada["label"] if entrada else None

    @property
    def labels(self) -> tuple:
        return tuple(e["label"] for e in self.tipos.values())


def taxonomia(perfil: dict, *, ruta=None, plantilla: str | None = None) -> Taxonomia:
    """La taxonomia que declara un perfil de dominio ya cargado. Falla cerrado.

    `ruta` es la del YAML, y sirve para UNA sola cosa: resolver `extraction.prompt_template`
    relativo al directorio del perfil. `plantilla` la pisa con un texto (los tests arman
    perfiles sinteticos sin archivo). El perfil se recibe como dict y NO se toca: un cargador
    que le metiera claves privadas se las estaria pasando a `admin_v1.descriptor()`.
    """
    if not perfil:
        raise ValueError("no hay perfil: sin perfil no se sabe que se le esta pidiendo al modelo")

    tipos = {}
    for entrada in perfil.get("entities") or []:
        tipos[entrada["id"]] = {"label": entrada.get("label") or entrada["id"],
                                "desc": entrada.get("desc") or ""}

    relaciones = {}
    for entrada in perfil.get("relations") or []:
        relaciones[entrada["id"]] = Relacion(
            id=entrada["id"],
            desde=frozenset(entrada.get("from") or []),
            hasta=frozenset(entrada.get("to") or []),
            extraer=entrada.get("extraer", True) is not False,
        )

    extraccion = perfil.get("extraction") or {}
    canon = perfil.get("canonicalization") or {}
    grafo = perfil.get("graph") or {}

    tx = Taxonomia(
        dominio=perfil.get("profile") or "?",
        version=int(perfil.get("version") or 0),
        tipos=tipos,
        relaciones=relaciones,
        name_property=grafo.get("name_property") or "nombre",
        modelo=extraccion.get("model") or "",
        batch=int(extraccion.get("batch_size") or BATCH_SIZE),
        min_words=int(extraccion.get("min_words") or MIN_WORDS),
        max_chars_fragmento=int(extraccion.get("max_chars_fragmento") or MAX_CHARS_FRAGMENTO),
        reglas=extraccion.get("rules") or "",
        plantilla=_plantilla(perfil, ruta, plantilla),
        min_nombre=int(canon.get("min_name_length") or MIN_NOMBRE),
        max_nombre=int(canon.get("max_name_length") or MAX_NOMBRE),
        fundir_sinonimos=canon.get("merge_synonyms", True) is not False,
        plegar_acentos=canon.get("fold_accents", False) is True,
        max_menciones=int(canon.get("max_menciones") or MAX_MENCIONES),
    )
    _validar(tx)
    return tx


def _plantilla(perfil: dict, ruta, plantilla: str | None) -> str:
    """El texto de `extraction.prompt_template`, resuelto RELATIVO AL DIRECTORIO DEL PERFIL.

    Hasta el 14-sep-2026 ese campo era un path relativo a nada: el directorio `prompts/` no
    existia ni en el contrato, ni en `medgraph/`, ni en `medgraph-engine/`, y los TRES perfiles
    apuntaban a un archivo inexistente desde el 9-sep.

    LOS SALTOS DE LINEA FINALES SE DESCARTAN: un archivo de texto termina en newline y un
    prompt no. Sin esto la plantilla de medicina no podria ser byte a byte la del codigo.
    """
    if plantilla is not None:
        return plantilla.rstrip("\n")
    referencia = (perfil.get("extraction") or {}).get("prompt_template")
    if not referencia:
        return ""
    base = (Path(ruta).parent if ruta
            else Path(__file__).resolve().parents[1] / "profiles")
    destino = base / referencia
    if not destino.exists():
        raise FileNotFoundError(
            f"el perfil '{perfil.get('profile')}' declara prompt_template '{referencia}' y el "
            f"archivo no esta en {destino}; copialo de nomos-contracts/profiles/prompts/")
    return destino.read_text(encoding="utf-8").rstrip("\n")


def _validar(tx: Taxonomia) -> None:
    """Un perfil incoherente produce extraccion basura y se descubre con el grafo cargado."""
    if not tx.tipos:
        raise ValueError(f"perfil '{tx.dominio}': no declara ni un tipo de entidad")
    for tipo in tx.tipos:
        if not _ID_ENTIDAD.match(tipo):
            raise ValueError(
                f"perfil '{tx.dominio}': el id de entidad {tipo!r} no matchea ^[a-z][a-z0-9_]*$ "
                "(los ids viajan en el prompt y vuelven en el JSON del modelo)")
    labels = [e["label"] for e in tx.tipos.values()]
    if len(set(labels)) != len(labels):
        repetidos = sorted({x for x in labels if labels.count(x) > 1})
        raise ValueError(
            f"perfil '{tx.dominio}': labels repetidos {repetidos}: dos tipos distintos "
            "escribirian en los mismos nodos del grafo")
    for rid, rel in tx.relaciones.items():
        if not _ID_RELACION.match(rid):
            raise ValueError(
                f"perfil '{tx.dominio}': el id de relacion {rid!r} no matchea ^[A-Z][A-Z0-9_]*$ "
                "(es el tipo del arco en Cypher: `(a)-[:ID]->(b)`)")
        for extremo, ids in (("from", rel.desde), ("to", rel.hasta)):
            desconocidos = sorted(ids - tx.ids_entidad)
            if desconocidos:
                raise ValueError(
                    f"perfil '{tx.dominio}': {rid}.{extremo} referencia tipos que no existen "
                    f"{desconocidos}; los declarados son {sorted(tx.ids_entidad)}")
    if not 0 < tx.min_nombre < tx.max_nombre:
        raise ValueError(
            f"perfil '{tx.dominio}': canonicalization tiene que cumplir 0 < min_name_length "
            f"({tx.min_nombre}) < max_name_length ({tx.max_nombre})")
    if tx.max_menciones < 1:
        raise ValueError(f"perfil '{tx.dominio}': max_menciones ({tx.max_menciones}) < 1: "
                         "ninguna entidad quedaria ligada a su chunk")
    if tx.batch < 1:
        raise ValueError(f"perfil '{tx.dominio}': batch_size ({tx.batch}) < 1")
    if tx.plantilla:
        desconocidos = sorted({m for m in _MARCADOR.findall(tx.plantilla)} - set(MARCADORES))
        if desconocidos:
            raise ValueError(
                f"perfil '{tx.dominio}': la plantilla usa marcadores que nadie renderiza "
                f"{desconocidos}; los que hay son {list(MARCADORES)}")
        if "{fragments}" not in tx.plantilla:
            raise ValueError(
                f"perfil '{tx.dominio}': la plantilla no tiene {{fragments}}: seria un prompt "
                "sin el texto a extraer, y se pagaria igual")


# ══════════════════════════════════════════════════════════════════════════════════════
# EL PROMPT — el UNICO armador
# ══════════════════════════════════════════════════════════════════════════════════════

def _render(plantilla: str, valores: dict) -> str:
    """Sustituye SOLO los marcadores conocidos; cualquier otra llave queda literal."""
    return _MARCADOR.sub(lambda m: str(valores.get(m.group(1), m.group(0))), plantilla)


def armar_fragmento(tx: Taxonomia, indice: int, chunk: dict, libro_titulo: str) -> str:
    """Un fragmento del prompt: su numero, su contexto y su texto recortado."""
    contexto = f"{ETIQUETA_LIBRO}: {libro_titulo}"
    if chunk.get("titulo_capitulo"):
        contexto += f", {ETIQUETA_CAPITULO}: {chunk['titulo_capitulo']}"
    if chunk.get("titulo_seccion"):
        contexto += f", {ETIQUETA_SECCION}: {chunk['titulo_seccion']}"
    texto = (chunk.get("text") or "")[:tx.max_chars_fragmento]
    return (f"--- FRAGMENTO {indice} ---\n"
            f"Contexto: {contexto}\n"
            f"Texto:\n\"\"\"\n{texto}\n\"\"\"")


def armar_prompt(tx: Taxonomia, chunks: list, libro_titulo: str = "") -> str:
    """EL armador. Todo lo que se le pide al modelo sale de `tx`, y `tx` sale del perfil."""
    if not tx.plantilla:
        raise ValueError(
            f"perfil '{tx.dominio}': no hay plantilla de prompt (extraction.prompt_template); "
            "sin plantilla no hay que pedirle al modelo")
    fragmentos = "\n\n".join(armar_fragmento(tx, i, c, libro_titulo)
                             for i, c in enumerate(chunks))
    return _render(tx.plantilla, {
        "n": len(chunks),
        "fragments": fragmentos,
        "libro": libro_titulo,
        "entities": ", ".join(tx.tipos),
        "entities_desc": "\n".join(f"- {tid}: {e['desc']}" for tid, e in tx.tipos.items()),
        "relations": ", ".join(tx.extraibles),
        "relations_desc": "\n".join(
            f"- {r.id}: {' | '.join(sorted(r.desde)) or 'cualquiera'} -> "
            f"{' | '.join(sorted(r.hasta)) or 'cualquiera'}"
            for r in tx.relaciones.values() if r.extraer),
        "rules": tx.reglas,
    })


# ══════════════════════════════════════════════════════════════════════════════════════
# EL VALIDADOR
# ══════════════════════════════════════════════════════════════════════════════════════

def normalizar_nombre(nombre: str, tx: Taxonomia | None = None) -> str:
    """minuscula, sin puntuacion final, espacios colapsados. Y acentos SOLO si el perfil lo
    pide (`canonicalization.fold_accents`, hoy `false` en los tres).

    El `normalize_name` del codigo viejo decia "sin acentos extra" en su docstring y no sacaba
    ninguno, en tres copias. Encender el plegado cambia nombres canonicos del grafo: es una
    migracion (`dedup_entities.py`), no un cambio de YAML.
    """
    if not nombre:
        return ""
    nombre = nombre.strip().lower()
    nombre = re.sub(r"[\.;,]+$", "", nombre).strip()
    nombre = re.sub(r"\s+", " ", nombre)
    if tx is not None and tx.plegar_acentos:
        nombre = "".join(c for c in unicodedata.normalize("NFD", nombre)
                         if unicodedata.category(c) != "Mn")
    return nombre


def _descartar(descartes, libro_id, chunk_id, motivo: str, **campos) -> None:
    fila = {"motivo": motivo, "chunk_id": chunk_id, **campos}
    if descartes is not None:
        descartes.append(fila)
    eventos.emitir(log, "extraccion_descarte", libro_id=libro_id, **fila)


def validar(tx: Taxonomia, crudo: dict, chunk: dict, *,
            rechazo_from_to: bool = RECHAZO_FROM_TO, descartes: list | None = None) -> dict:
    """Lo que devolvio el modelo, filtrado contra `tx`. Los descartes se CUENTAN.

    Cuatro diferencias con el `_validate_extraction` que reemplaza, y solo la ultima cambia lo
    que sale:

      · los tipos y las relaciones se chequean contra el perfil, no contra dos sets de modulo;
      · `min_nombre`/`max_nombre` salen de `canonicalization` (eran 2 y 100 literales);
      · los `from`/`to` se CHEQUEAN y se emiten, pero no rechazan (ver `RECHAZO_FROM_TO`);
      · todo descarte deja rastro: el evento `extraccion_descarte` y la lista `descartes` del
        resultado. Antes se perdian en silencio y nadie podia medir la diferencia entre lo que
        el modelo devolvio y lo que quedo.

    LO QUE NO CAMBIA, y es una decision documentada: una relacion con UN SOLO extremo extraido
    en ESTE fragmento SOBREVIVE, como hasta hoy. El diseño proponia descartarla (§3.3, T3)
    dando por hecho que muere despues en la carga; no es cierto: `canonicalizar` funde las
    entidades de TODOS los chunks del libro, asi que el otro extremo puede ser una entidad de
    otro fragmento y la relacion entra al grafo. Descartarla aca perderia aristas reales. Se
    cuenta (`motivo="extremo_ausente"`) y se deja pasar.

    LOS DESCARTES NO VIAJAN EN EL RESULTADO, y tambien es a proposito: este dict se serializa
    tal cual en `extracted/{libro}_entities.json`, y `extraction/v1` declara sus items con
    `additionalProperties: false`. Quien los quiera pasa una lista en `descartes=`; el rastro
    permanente es el evento.
    """
    libro_id = chunk.get("libro_id", "")
    chunk_id = chunk.get("id", "")

    entidades, nombres = [], set()
    for cruda in crudo.get("entidades") or []:
        nombre = normalizar_nombre(cruda.get("nombre", ""), tx)
        tipo = (cruda.get("tipo") or "").lower().strip()
        if not nombre or len(nombre) < tx.min_nombre:
            _descartar(descartes, libro_id, chunk_id, "nombre_corto", tipo=tipo)
            continue
        if len(nombre) > tx.max_nombre:
            _descartar(descartes, libro_id, chunk_id, "nombre_largo", tipo=tipo)
            continue
        if tipo not in tx.tipos:
            _descartar(descartes, libro_id, chunk_id, "tipo_entidad", tipo=tipo)
            continue
        sinonimos = []
        for s in cruda.get("sinonimos") or []:
            sn = normalizar_nombre(s, tx)
            if sn and sn != nombre and len(sn) >= tx.min_nombre:
                sinonimos.append(sn)
        entidades.append({"nombre": nombre, "tipo": tipo, "sinonimos": sinonimos})
        nombres.add(nombre)

    tipo_de = {e["nombre"]: e["tipo"] for e in entidades}
    relaciones = []
    for cruda in crudo.get("relaciones") or []:
        desde = normalizar_nombre(cruda.get("desde", ""), tx)
        hasta = normalizar_nombre(cruda.get("hasta", ""), tx)
        rid = (cruda.get("relacion") or "").upper().strip()
        if not desde or not hasta or not rid:
            continue
        if rid not in tx.relaciones:
            _descartar(descartes, libro_id, chunk_id, "tipo_relacion", relacion=rid)
            continue
        if desde not in nombres and hasta not in nombres:
            _descartar(descartes, libro_id, chunk_id, "extremo_ausente", relacion=rid,
                       desde_tipo=None, hasta_tipo=None)
            continue
        regla = tx.relaciones[rid]
        desde_tipo, hasta_tipo = tipo_de.get(desde), tipo_de.get(hasta)
        viola = ((desde_tipo is not None and regla.desde and desde_tipo not in regla.desde) or
                 (hasta_tipo is not None and regla.hasta and hasta_tipo not in regla.hasta))
        if viola:
            _descartar(descartes, libro_id, chunk_id, "from_to", relacion=rid,
                       desde_tipo=desde_tipo, hasta_tipo=hasta_tipo)
            if rechazo_from_to:
                continue
        relaciones.append({"desde": desde, "relacion": rid, "hasta": hasta})

    return {"entidades": entidades, "relaciones": relaciones,
            "chunk_id": chunk_id, "libro_id": libro_id}


# ══════════════════════════════════════════════════════════════════════════════════════
# LA CANONICALIZACION
# ══════════════════════════════════════════════════════════════════════════════════════

def canonicalizar(tx: Taxonomia, extracciones: list) -> tuple:
    """Las menciones de todos los chunks, consolidadas en un set canonico de entidades.

    Dos cosas salen del perfil y antes eran literales:

      · `merge_synonyms`: con `false`, los sinonimos que devuelve el modelo se GUARDAN en el
        nodo pero no funden menciones (hoy es `true` en los tres perfiles, o sea que esto no
        cambia nada y queda declarado);
      · `max_menciones`: cuantos chunks se ligan a cada entidad.

    Y una tercera cosa cambia de verdad, aunque el numero no se mueva: `chunk_ids` era un
    `set`, asi que CUALES diez chunks quedaban ligados dependia del hash del proceso y dos
    corridas identicas producian grafos distintos. Ahora es una lista en orden de aparicion
    —o sea en orden del documento, porque las extracciones llegan en orden de chunk— y el
    corte toma los PRIMEROS. Sin esto ningun golden sobre entidades es posible.
    """
    entidades: dict = {}
    sinonimo_de: dict = {}

    for ext in extracciones:
        chunk_id = ext.get("chunk_id", "")
        for ent in ext.get("entidades") or []:
            nombre = ent["nombre"]
            canonico = sinonimo_de.get(nombre, nombre) if tx.fundir_sinonimos else nombre
            existente = entidades.get(canonico)
            if existente is not None:
                existente["freq"] += 1
                if chunk_id not in existente["_vistos"]:
                    existente["_vistos"].add(chunk_id)
                    existente["chunk_ids"].append(chunk_id)
                for s in ent.get("sinonimos") or []:
                    if s not in existente["sinonimos"] and s != canonico:
                        existente["sinonimos"].append(s)
                        if tx.fundir_sinonimos:
                            sinonimo_de[s] = canonico
            else:
                entidades[nombre] = {
                    "nombre": nombre,
                    "tipo": ent["tipo"],
                    "sinonimos": list(ent.get("sinonimos") or []),
                    "freq": 1,
                    "chunk_ids": [chunk_id],
                    "_vistos": {chunk_id},
                }
                if tx.fundir_sinonimos:
                    for s in ent.get("sinonimos") or []:
                        sinonimo_de[s] = nombre

    vistas, relaciones = set(), []
    for ext in extracciones:
        for rel in ext.get("relaciones") or []:
            desde = sinonimo_de.get(rel["desde"], rel["desde"])
            hasta = sinonimo_de.get(rel["hasta"], rel["hasta"])
            clave = (desde, rel["relacion"], hasta)
            if clave not in vistas:
                vistas.add(clave)
                relaciones.append({"desde": desde, "relacion": rel["relacion"], "hasta": hasta})

    salida = []
    for ent in entidades.values():
        ent.pop("_vistos", None)
        salida.append(ent)
    salida.sort(key=lambda e: e["freq"], reverse=True)
    return salida, relaciones


# ══════════════════════════════════════════════════════════════════════════════════════
# LA CARGA
# ══════════════════════════════════════════════════════════════════════════════════════

LOTE = 200


def cypher_entidades(tx: Taxonomia, label: str) -> str:
    """El MERGE de un lote de entidades de un label. La propiedad del nombre sale del perfil
    (`graph.name_property`): medicina escribe `nombre` (legado) y todo dominio nuevo `name`.
    Un rename silencioso partiria las 159K entidades en dos poblaciones que ninguna consulta
    cruza, asi que hay un test que exige el literal `{nombre:` para medicina."""
    return (f"\n                UNWIND $batch AS ent\n"
            f"                MERGE (e:{label} {{{tx.name_property}: ent.nombre}})\n"
            f"                SET e.tipo = ent.tipo,\n"
            f"                    e.sinonimos = ent.sinonimos,\n"
            f"                    e.freq = ent.freq,\n"
            f"                    e.fuente_libro = ent.libro_id\n                ")


def cargar(write, tx: Taxonomia, entidades: list, relaciones: list, libro_id: str,
           *, dev_mode: bool = False) -> dict:
    """Sube entidades, `MENCIONA` y relaciones al grafo. DEVUELVE LAS VIOLACIONES.

    El `except Exception: print(...)` por lote del codigo viejo se tragaba los errores de
    escritura y el paso reportaba lo CANONICALIZADO: con el grafo caido a mitad, `extract`
    informaba "N entidades" y en el grafo no habia ninguna. Ahora cada lote que no entra deja
    una fila en `violaciones` y el llamador decide (hoy: metrica de degradado, sin hacer
    fallar la ingesta, que es opcional por contrato).

    `dev_mode` por defecto **False** (era True): el sufijo `Dev` escribe un grafo paralelo que
    ninguna consulta mira. Se conserva porque `promote_dev_entities` lo usa.
    """
    sufijo = "Dev" if dev_mode else ""
    violaciones: list = []
    creadas = mencionadas = relacionadas = 0

    def _escribir(cypher, parametros, contexto):
        try:
            write(cypher, parametros)
            return True
        except Exception as e:
            violaciones.append({"contexto": contexto, "n": len(parametros.get("batch") or []),
                                "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return False

    por_tipo: dict = {}
    for ent in entidades:
        por_tipo.setdefault(ent["tipo"], []).append(ent)

    for tipo, ents in por_tipo.items():
        label = tx.label_de(tipo)
        if label is None:
            violaciones.append({"contexto": f"tipo_sin_label:{tipo}", "n": len(ents),
                                "error": f"el tipo {tipo!r} no esta en el perfil {tx.perfil}"})
            continue
        label += sufijo
        for i in range(0, len(ents), LOTE):
            lote = ents[i:i + LOTE]
            datos = [{"nombre": e["nombre"], "tipo": e["tipo"],
                      "sinonimos": e.get("sinonimos", []), "freq": e.get("freq", 1),
                      "libro_id": libro_id} for e in lote]
            if _escribir(cypher_entidades(tx, label), {"batch": datos}, f"entidades:{label}"):
                creadas += len(lote)

    menciones: dict = {}
    for ent in entidades:
        label = tx.label_de(ent["tipo"])
        if label is None:
            continue
        for chunk_id in (ent.get("chunk_ids") or [])[:tx.max_menciones]:
            if chunk_id:
                menciones.setdefault(label + sufijo, []).append(
                    {"nombre": ent["nombre"], "chunk_id": chunk_id})

    for label, items in menciones.items():
        cypher = (f"\n                UNWIND $batch AS m\n"
                  f"                MATCH (e:{label} {{{tx.name_property}: m.nombre}})\n"
                  f"                MATCH (c:Chunk {{id: m.chunk_id}})\n"
                  f"                MERGE (c)-[:MENCIONA]->(e)\n                ")
        for i in range(0, len(items), LOTE):
            lote = items[i:i + LOTE]
            if _escribir(cypher, {"batch": lote}, f"menciona:{label}"):
                mencionadas += len(lote)

    label_de_entidad = {}
    for ent in entidades:
        label = tx.label_de(ent["tipo"])
        if label is not None:
            label_de_entidad[ent["nombre"]] = label + sufijo

    grupos: dict = {}
    for rel in relaciones:
        desde_l = label_de_entidad.get(rel["desde"])
        hasta_l = label_de_entidad.get(rel["hasta"])
        if desde_l and hasta_l:
            grupos.setdefault((desde_l, rel["relacion"], hasta_l), []).append(
                {"desde": rel["desde"], "hasta": rel["hasta"]})

    for (desde_l, tipo_rel, hasta_l), items in grupos.items():
        cypher = (f"\n                UNWIND $batch AS r\n"
                  f"                MATCH (a:{desde_l} {{{tx.name_property}: r.desde}})\n"
                  f"                MATCH (b:{hasta_l} {{{tx.name_property}: r.hasta}})\n"
                  f"                MERGE (a)-[:{tipo_rel}]->(b)\n                ")
        for i in range(0, len(items), LOTE):
            lote = items[i:i + LOTE]
            if _escribir(cypher, {"batch": lote},
                         f"relaciones:{desde_l}-[{tipo_rel}]->{hasta_l}"):
                relacionadas += len(lote)

    return {"entities": creadas, "menciona": mencionadas, "relations": relacionadas,
            "violaciones": violaciones}
