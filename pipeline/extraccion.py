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

LAS CUATRO REGLAS DE v3 (18-sep-2026, `docs/DISENO-curar-extractor-18sep.md`). El juez-LLM midio
la extraccion el 14-sep contra seis controles positivos firmados: reales 3,63/5 y **`j3_relacion`
2,08**. El eje debil no era la entidad: era la RELACION. Los cuatro defectos que el juez nombro se
curan mitad en el perfil (`medicina@3`: tipo `molecula_biologica`, `sinonimos` de clase, `ejemplo`
por relacion, tres reglas nuevas) y mitad aca:

  1. ORIENTACION IMPOSIBLE (`RECHAZO_ORIENTACION`, ENCENDIDO). Una relacion cuyo par de tipos
     viola los `from`/`to` EN ESTE SENTIDO y los cumpliria DADA VUELTA esta invertida, y eso no
     es una regla dudosa: es un error. Se rechaza. Medido sobre el corpus ya extraido (227.388
     relaciones de los tres tratados largos): 8,27%, con `farmaco SE_TRATA_CON patologia` (9.081),
     `grupo_farmacologico SE_TRATA_CON patologia` (2.585) y `agente CAUSADA_POR patologia` (2.178)
     a la cabeza — los tres estan en la cola de peores del juez.
     El par ILEGAL EN LOS DOS SENTIDOS sigue contandose sin rechazar (`RECHAZO_FROM_TO`, apagado):
     17,10%, dominado por pares que `ASOCIADA_A` —"laxa a proposito"— no incluye. Ahi la regla es
     sospechosa antes que el dato, y es la leccion de los 71.569 de julio.
  2. UN NOMBRE QUE ES UNA CLASE NO ES UNA ENTIDAD. `'tionamidas PERTENECE_A grupo_farmacologico'`:
     el destino tiene que ser el grupo CON NOMBRE. El perfil declara por tipo las superficies con
     que la prosa nombra a la clase (`entities[].sinonimos`), y se descarta la entidad —o el
     extremo de relacion— cuyo nombre sea exactamente una de ellas, el id o el label. Medido:
     1.697 destinos (0,75%), 1.542 en `PERTENECE_A` y 1.452 literalmente 'grupo farmacologico';
     mas 137 entidades (0,03%), 111 de ellas llamadas 'farmaco'.
  3. EVIDENCIA TEXTUAL. La entidad tiene que estar nombrada en el fragmento —los mismos
     `max_chars_fragmento` que vio el modelo—, por su nombre, por un sinonimo que el modelo
     devolvio, o por una variante de sufijo. Lo que no aparece se descarta y se CUENTA
     (`inferidas_descartadas`). Medido sobre 425.597 entidades: exigir la superficie literal
     descartaria 19,2%; con los sinonimos del modelo, 9,5%; con la variante de sufijo, 7,6%. La
     diferencia la explica la regla de traducir al español, que hace que el nombre canonico no
     aparezca literal en un fragmento en ingles ("encainide" -> "encainida").
  4. El tipo que faltaba es una decision de PERFIL y no de codigo: entra solo, porque
     `ENTITY_TYPES` y el prompt se derivan de `tx`.

LO QUE SE AGREGO EL 19-sep-2026 (`docs/DISENO-labels-del-perfil-19sep.md`): cuatro lecturas y
ni un estado nuevo. `tipo_de_label`, `desc_de`, `sinonimos_de` y `tipos_con_ontologia` no las
usa la EXTRACCION: las usa el RETRIEVAL, que hasta ese dia nombraba los labels a mano en
dieciseis lugares y por eso iba atrasado —diez de doce tipos en un lado, nueve en otro, ocho en
otro—. La taxonomia ya era la unica fuente para pedirle al modelo; desde hoy tambien lo es para
buscar lo que el modelo dejo.

SIN `print`: este paquete se replica al engine y lo vigila `test_bitacora.PAQUETE_SIN_PRINT`.
Lo que hay que contar viaja por `pipeline.eventos`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from pipeline import eventos, normalizacion

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

#: EL FLAG DE POLITICA (decision 3 de §8 del diseño de la extraccion por perfil). Con `False` una
#: relacion cuyo par de tipos es ILEGAL EN LOS DOS SENTIDOS se CUENTA y se emite, pero ENTRA.
#: Medicina ya pago una vez el precio de reglas mal escritas: la primera version de los
#: `from`/`to` marcaba 71.569 relaciones como invalidas y la mayoria eran reglas malas, no datos
#: malos (`medicina.yaml`). El 18-sep-2026 se volvio a medir sobre el corpus ya extraido y el
#: diagnostico se confirmo: el 17,10% de las 227.388 relaciones cae en este balde, y arriba de la
#: lista estan pares que `ASOCIADA_A` —declarada "laxa a proposito"— simplemente no incluye
#: (`farmaco ASOCIADA_A estructura_anatomica`, 3.125). Sigue apagado, y lo que se prendio es la
#: mitad que NO es ambigua: la de abajo.
RECHAZO_FROM_TO = False

#: LA MITAD QUE SI SE RECHAZA (18-sep-2026, §2 del diseño de curar el extractor): la orientacion
#: imposible. Si el par viola los `from`/`to` en este sentido y los CUMPLE dado vuelta, la
#: relacion esta invertida y no hay regla dudosa que discutir. Es el defecto mas frecuente que
#: nombro el juez, y son 18.798 relaciones (8,27%) del corpus ya extraido.
RECHAZO_ORIENTACION = True

#: Los motivos de descarte que viajan en el evento `extraccion_descarte`. Los tres ultimos son
#: de v3; el vocabulario tambien esta en `pipeline/eventos.py`, que es el contrato con el harness.
MOTIVOS = ("tipo_entidad", "tipo_relacion", "nombre_corto", "nombre_largo", "from_to",
           "extremo_ausente", "orientacion", "nombre_de_tipo", "sin_evidencia")

#: LA EVIDENCIA TEXTUAL, en dos numeros. Dos palabras son "la misma con otro sufijo" si comparten
#: todo menos el ultimo caracter de la mas corta y sus largos difieren en <= 3: eso hace
#: "enfermedades"/"enfermedad", "cronica"/"cronicas" y "encainide"/"encainida", y NO hace
#: "cardiopatia"/"cardiologia" (difieren en el caracter 7) ni "gastrico"/"gastrointestinal"
#: (8 caracteres de diferencia). Las palabras de menos de 4 caracteres exigen igualdad: con tres
#: letras, "salvo el sufijo" no dice nada.
MIN_TOKEN_SUFIJO, MAX_DIF_SUFIJO = 4, 3
#: LA EVIDENCIA DE LA RELACION (20-sep-2026, `docs/DISENO-relaciones-20sep.md`). El prompt pide
#: un tramo de hasta 160 caracteres; plegado y con margen, mas de 240 ya no es "el tramo que la
#: afirma" sino el fragmento entero, que nombra a los dos extremos trivialmente. Y la evidencia
#: que no esta literal en el fragmento (el modelo la copio con un tropiezo) se acepta si tiene al
#: menos MIN_PALABRAS_EVIDENCIA palabras y UMBRAL_EVIDENCIA de ellas estan en el fragmento.
MAX_EVIDENCIA, MIN_PALABRAS_EVIDENCIA, UMBRAL_EVIDENCIA = 240, 3, 0.85
#: Fin de oracion DENTRO de la evidencia cruda (v6, 20-sep): punto/cierre, espacio y mayuscula. Un
#: tramo que termina una oracion y empieza otra no afirma la relacion de una sola vez: es la
#: "inferencia por proximidad" que el juez siguio objetando sobre `medicina@5` ("AINE PUEDE_PRODUCIR
#: trastorno hemorragico... solo los menciona en proximidad"). "S. aureus" no matchea (minuscula).
_RX_FIN_DE_ORACION = re.compile(r"[.!?]\s+[A-ZÁÉÍÓÚÑ¿¡]")

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
    #: La relacion USADA, en una linea, con su direccion (`relations[].ejemplo` del perfil,
    #: 18-sep-2026). Viaja al prompt dentro de `{relations_desc}` y es la mitad-prompt del
    #: rechazo por orientacion: el validador tira lo invertido y el ejemplo es lo que evita que
    #: el modelo lo produzca. Vacio = la relacion no lo declara (las cuatro huerfanas no lo
    #: declaran: lo que no se pide no se ejemplifica).
    ejemplo: str = ""
    #: POLARIDAD (20-sep-2026, `docs/DISENO-relaciones-20sep.md` §1.3): raices de los verbos que
    #: AFIRMAN esta relacion y de los que la NIEGAN ("previene" niega PUEDE_PRODUCIR), tal como el
    #: perfil las escribe (`relations[].afirma` / `.niega`); viajan al prompt y las lee
    #: `negada_en`. Vacias = la relacion no declara polaridad y la regla no rige (derecho,
    #: generico, y las relaciones no causales de medicina).
    afirma: tuple = ()
    niega: tuple = ()

    def negada_en(self, evidencia: str) -> bool:
        """True si la evidencia (ya plegada) trae un verbo que NIEGA la relacion y, tachado ese
        tramo, ninguno que la afirme. "no causa X" niega aunque contenga "causa": lo negado se
        tacha ANTES de buscar la afirmacion. Con las dos cosas ("previene X pero causa Y") no se
        decide y pasa: la direccion y el sentido los juzga el juez, no una heuristica. Sin
        lexicon devuelve False: no se adivina.
        """
        if not self.niega or not evidencia:
            return False
        tachada, hubo = evidencia, False
        for raiz in self.niega:
            nueva = re.sub(r"\b" + re.escape(_plegar(raiz)), " ", tachada)
            hubo, tachada = hubo or nueva != tachada, nueva
        if not hubo:
            return False
        return not any(re.search(r"\b" + re.escape(_plegar(raiz)), tachada) for raiz in self.afirma)

    def acepta(self, desde_tipo: str, hasta_tipo: str) -> bool:
        """Si este par de tipos cumple los `from`/`to`. Un extremo sin declarar acepta todo."""
        return ((not self.desde or desde_tipo in self.desde)
                and (not self.hasta or hasta_tipo in self.hasta))

    def orientacion_imposible(self, desde_tipo: str | None, hasta_tipo: str | None) -> bool:
        """LA REGLA DE LA DIRECCION (18-sep-2026). True si el par viola los `from`/`to` en ESTE
        sentido y los CUMPLE dado vuelta: ahi la relacion esta invertida y no hay nada que
        discutir. `farmaco SE_TRATA_CON patologia` es True (`patologia -> farmaco` es legal);
        `grupo_farmacologico SE_TRATA_CON molecula_biologica` es False —ilegal en los dos
        sentidos— y cae en el balde de `from_to`, que se cuenta y no rechaza.

        Con un extremo sin tipo (la entidad vino de otro fragmento) devuelve False: no se puede
        decidir, y rechazar por lo que no se sabe perderia aristas reales.
        """
        if desde_tipo is None or hasta_tipo is None:
            return False
        return not self.acepta(desde_tipo, hasta_tipo) and self.acepta(hasta_tipo, desde_tipo)


@dataclass(frozen=True)
class Taxonomia:
    """Lo que un dominio le pide al modelo y lo que acepta de vuelta. Inmutable: se construye
    una vez por corrida, como `Estrategia`."""

    dominio: str
    version: int
    #: id -> {"label": str, "desc": str, "sinonimos": tuple, "ontologia": str}, EN EL ORDEN DEL
    #: YAML (el prompt lo respeta). `sinonimos` son las superficies con que la prosa nombra a LA
    #: CLASE, no a una entidad (ver `nombres_de_tipo`); `ontologia` es el nombre de la taxonomia
    #: externa a la que ESE tipo se mapea, o "" (ver `tipos_con_ontologia`).
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
    #: `extraction.evidencia_relaciones` (20-sep-2026): el validador EXIGE y LEE la `evidencia` de
    #: cada relacion (`_motivo_evidencia_relacion`). False = la clave, si viene, se ignora y se
    #: quita (derecho y generico no la declaran; sus goldens no se mueven).
    evidencia_relaciones: bool = False

    # ── lecturas ──────────────────────────────────────────────────────────────────────
    @property
    def perfil(self) -> str:
        """`medicina@3`, tal como viaja en el linaje y en el descriptor de admin/v1."""
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

    def tipo_de_label(self, label: str) -> str | None:
        """El inverso de `label_de`: el id del tipo que escribe ese label, o None.

        POR QUE EXISTE (19-sep-2026, `docs/DISENO-labels-del-perfil-19sep.md`). El retrieval
        razona en LABELS —`buscar_en: "Patologia"` es lo que el analizador emite y lo que viaja
        en el `WHERE` del Cypher— y el perfil razona en TIPOS. Sin este inverso, todo el que
        recibe un label tiene que llevar su propio mapa al lado, que es exactamente las
        dieciseis copias que esta tanda vino a borrar.

        **None y no una excepcion**, igual que `label_de`: un label que el perfil no declara
        existe de verdad en el grafo (`CategoriaATC`, `Chunk`, `Tema`) y preguntar por el es
        legitimo. Quien necesite que falle, que compare contra None.
        """
        for tid, entrada in self.tipos.items():
            if entrada["label"] == label:
                return tid
        return None

    def desc_de(self, tipo: str) -> str:
        """La descripcion de un tipo, o "" si no lo declara. Es la tercera pata de la tripleta
        `(id, label, desc)` con que el prompt del analizador publica su vocabulario: sin ella,
        una linea derivada del perfil diria menos que la que estaba escrita a mano."""
        entrada = self.tipos.get(tipo)
        return entrada["desc"] if entrada else ""

    def sinonimos_de(self, tipo: str) -> tuple:
        """Las superficies con que la prosa nombra a ESTA clase, TAL COMO LAS DECLARA EL PERFIL.

        SIN PLEGAR, y esa es toda la diferencia con `nombres_de_tipo`: aquella pliega acentos y
        puntuacion porque COMPARA, y estas son claves de un diccionario que se consulta con lo
        que el modelo devolvio (`api/services/analyzer._TYPE_NORMALIZE`). Plegarlas ahi juntaria
        "grupo farmacologico" y "grupo farmacológico" en una sola clave, que hoy son dos con
        destinos distintos.
        """
        entrada = self.tipos.get(tipo)
        return tuple(entrada["sinonimos"]) if entrada else ()

    def tipos_con_ontologia(self, cual: str | None = None) -> tuple:
        """Los tipos que declaran `ontologia`, en el orden del YAML. Con `cual`, solo los de esa.

        POR QUE ESTA EN EL PERFIL Y NO EN EL CODIGO (19-sep-2026). "Farmaco sube a ATC;
        Patologia, EstructuraAnatomica y Procedimiento suben a SNOMED" estaba escrito a mano en
        CUATRO lugares del retrieval (`api/routers/ontology_router.py`, `api/services/layers.py`
        y dos veces en `ontology.py`). No es una decision del codigo: es del dominio, igual que
        los `from`/`to`. Un perfil que no la declare en ningun tipo devuelve vacio, que es
        exactamente lo que le pasa hoy a `derecho` y a `generico`.
        """
        return tuple(tid for tid, e in self.tipos.items()
                     if e["ontologia"] and (cual is None or e["ontologia"] == cual))

    @cached_property
    def nombres_de_tipo(self) -> frozenset:
        """TODAS las formas de nombrar una CLASE del vocabulario, plegadas para comparar: el id
        (con y sin guiones bajos), el label y los `sinonimos` que el perfil declara.

        Para que sirve, y es el defecto 2 del juez (18-sep-2026): `'tionamidas PERTENECE_A
        grupo_farmacologico'`. El destino de una relacion tiene que ser una entidad con nombre;
        si el nombre ES la clase, no hay entidad. Se compara por IGUALDAD exacta sobre el nombre
        plegado, nunca por substring: "enfermedad de crohn" contiene "enfermedad" y es una
        patologia perfectamente valida.
        """
        salida = set()
        for tid, entrada in self.tipos.items():
            salida.add(_plegar(tid))
            salida.add(_plegar(tid.replace("_", " ")))
            salida.add(_plegar(entrada["label"]))
            salida.update(_plegar(s) for s in entrada.get("sinonimos") or ())
        return frozenset(x for x in salida if x)


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
                                "desc": entrada.get("desc") or "",
                                "sinonimos": tuple(entrada.get("sinonimos") or ()),
                                "ontologia": (entrada.get("ontologia") or "").strip()}

    relaciones = {}
    for entrada in perfil.get("relations") or []:
        relaciones[entrada["id"]] = Relacion(
            id=entrada["id"],
            desde=frozenset(entrada.get("from") or []),
            hasta=frozenset(entrada.get("to") or []),
            extraer=entrada.get("extraer", True) is not False,
            ejemplo=(entrada.get("ejemplo") or "").strip(),
            afirma=tuple(str(x).strip().lower() for x in (entrada.get("afirma") or ())
                         if str(x).strip()),
            niega=tuple(str(x).strip().lower() for x in (entrada.get("niega") or ())
                        if str(x).strip()),
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
        evidencia_relaciones=extraccion.get("evidencia_relaciones", False) is True,
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
    # `ontologia` (19-sep-2026) nombra una taxonomia externa y el codigo la compara por igualdad
    # (`tipos_con_ontologia("atc")`), asi que un "ATC " con mayusculas o un espacio de mas serian
    # un subconjunto vacio y un retrieval mudo. Misma forma que un id de entidad.
    for tid, entrada in tx.tipos.items():
        ontologia = entrada["ontologia"]
        if ontologia and not _ID_ENTIDAD.match(ontologia):
            raise ValueError(
                f"perfil '{tx.dominio}': {tid}.ontologia {ontologia!r} no matchea "
                "^[a-z][a-z0-9_]*$ (se compara por igualdad, no por parecido)")
    labels = [e["label"] for e in tx.tipos.values()]
    if len(set(labels)) != len(labels):
        repetidos = sorted({x for x in labels if labels.count(x) > 1})
        raise ValueError(
            f"perfil '{tx.dominio}': labels repetidos {repetidos}: dos tipos distintos "
            "escribirian en los mismos nodos del grafo")
    # Los `sinonimos` de clase deciden que nombre SE DESCARTA, asi que uno que pertenezca a dos
    # tipos hace ambiguo el descarte y uno que sea el id de otro tipo es el vocabulario
    # pisandose. Falla al construir, como todo lo demas de este validador.
    dueno: dict = {}
    for tid, entrada in tx.tipos.items():
        for sinonimo in entrada.get("sinonimos") or ():
            clave = _plegar(sinonimo)
            if not clave:
                raise ValueError(
                    f"perfil '{tx.dominio}': {tid}.sinonimos trae un texto vacio")
            if dueno.setdefault(clave, tid) != tid:
                raise ValueError(
                    f"perfil '{tx.dominio}': el sinonimo de clase {sinonimo!r} esta en "
                    f"'{dueno[clave]}' y en '{tid}': el descarte por 'el nombre es una clase' "
                    "seria ambiguo")
            ajenos = [otro for otro in tx.tipos if otro != tid
                      and clave in (_plegar(otro), _plegar(otro.replace("_", " ")))]
            if ajenos:
                raise ValueError(
                    f"perfil '{tx.dominio}': {tid}.sinonimos trae {sinonimo!r}, que es el id "
                    f"del tipo '{ajenos[0]}'")
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
        "relations_desc": "\n".join(_linea_relacion(r) for r in tx.relaciones.values()
                                    if r.extraer),
        "rules": tx.reglas,
    })


def _linea_relacion(r: Relacion) -> str:
    """`- ID: a | b -> c | d`, mas una segunda linea `    ej: …` si la relacion declara `ejemplo`.

    Sin `ejemplo` rinde exactamente lo de antes (derecho y generico no lo declaran, y sus goldens
    no se mueven). El ejemplo es lo que le dice al modelo PARA QUE LADO va la relacion: hasta
    `medicina@2` el prompt de medicina publicaba trece ids separados por coma, sin una palabra de
    semantica, y el juez midio ese prompt con `j3_relacion` 2,08 sobre 5.
    """
    linea = (f"- {r.id}: {' | '.join(sorted(r.desde)) or 'cualquiera'} -> "
             f"{' | '.join(sorted(r.hasta)) or 'cualquiera'}")
    if r.ejemplo:
        linea += f"\n    ej: {r.ejemplo}"
    # La polaridad (20-sep-2026), en las palabras del perfil: es la mitad-prompt de `polaridad`.
    if r.afirma:
        linea += f"\n    afirma: {', '.join(r.afirma)}"
    if r.niega:
        linea += f"\n    niega: {', '.join(r.niega)}"
    return linea


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
        # `normalizacion.sin_acentos` (R3, 22-sep-2026) y no las dos lineas de NFD que habia aca.
        # RAMA MUERTA HOY: `fold_accents` es `false` en los tres perfiles, asi que esto no corre y
        # el cambio (NFD -> NFKD) no puede mover nada del corpus actual. Lo que gana: el dia que se
        # encienda, la ligadura del PDF ("ﬁbrosis") se escribe como "fibrosis" en el nombre canonico
        # en vez de quedarse pegada.
        nombre = normalizacion.sin_acentos(nombre)
    return nombre


def _plegar(texto: str) -> str:
    """Minuscula, SIN ACENTOS, sin puntuacion, espacios colapsados. ES PARA COMPARAR.

    No confundir con `normalizar_nombre`, que ESCRIBE el nombre canonico del grafo y por eso
    respeta `canonicalization.fold_accents` (false en los tres perfiles: encenderlo es una
    migracion). Acá plegar acentos es gratis y necesario: "insuficiencia cardíaca" tiene que
    encontrarse en un pasaje que escriba "insuficiencia cardiaca", y ninguno de los dos nombres
    se guarda en ninguna parte.

    ES `normalizacion.plegar` DESDE EL 22-sep-2026 (R3), y el cambio NO fue cosmetico: la version
    anterior descomponia con **NFD** despues de pasar `_NO_PALABRA = [^0-9a-zA-ZÀ-ſ]+`, y la
    ligadura tipografica que todo PDF escupe (`ﬁ`, U+FB01) cae FUERA de ese rango. Resultado
    medido: `_plegar("ﬁbrosis")` daba **`"brosis"`** —la palabra perdia sus dos primeras letras— y
    `_aparece` no encontraba "fibrosis quistica" en un fragmento que la nombra, asi que el
    validador **descartaba la relacion**. Con NFKD la ligadura se descompone antes y da
    `"fibrosis"`. El cambio hace que el validador acepte MAS, y solo donde antes se equivocaba.
    """
    return normalizacion.plegar(texto)


def _mismo_token(a: str, b: str) -> bool:
    """Dos palabras que son LA MISMA CON OTRO SUFIJO (ver `MIN_TOKEN_SUFIJO`/`MAX_DIF_SUFIJO`)."""
    if a == b:
        return True
    corto, largo = (a, b) if len(a) <= len(b) else (b, a)
    if len(corto) < MIN_TOKEN_SUFIJO or len(largo) - len(corto) > MAX_DIF_SUFIJO:
        return False
    return corto[:-1] == largo[:len(corto) - 1]


def _aparece(nombre: str, fragmento: str, tokens: list) -> bool:
    """Si `nombre` (ya plegado) esta nombrado en el fragmento: literal, o como secuencia
    CONTIGUA de palabras que son las mismas con otro sufijo.

    La contiguidad es lo que hace que la regla no se afloje: sin ella, "insuficiencia renal"
    matchearia un pasaje que dice "insuficiencia cardiaca" y "funcion renal" en renglones
    distintos, que es justo la entidad inferida que se quiere descartar.
    """
    if not nombre:
        return False
    if nombre in fragmento:
        return True
    partes = nombre.split()
    if not partes or len(partes) > len(tokens):
        return False
    for i in range(len(tokens) - len(partes) + 1):
        if all(_mismo_token(a, b)
               for a, b in zip(partes, tokens[i:i + len(partes)], strict=True)):
            return True
    return False


def con_evidencia(nombre: str, sinonimos, fragmento: str, tokens: list) -> bool:
    """La entidad esta nombrada en el fragmento por su nombre o por uno de sus sinonimos.

    LOS SINONIMOS SON LA MITAD DE LA REGLA, y no un adorno: el perfil ordena traducir al español
    ("heart failure" -> "insuficiencia cardíaca") y pedir los sinonimos del texto, asi que en un
    fragmento en ingles la superficie que aparece es el SINONIMO y no el nombre canonico. Sin
    esta mitad la regla descartaria 19,2% de las entidades del corpus en vez de 7,6%.
    """
    if _aparece(nombre, fragmento, tokens):
        return True
    return any(_aparece(_plegar(s), fragmento, tokens) for s in sinonimos or ())


def _motivo_evidencia_relacion(evidencia, desde: str, hasta: str, sinonimos_de: dict,
                               fragmento: str, tokens: list, regla: Relacion) -> str | None:
    """LA REGLA DE EVIDENCIA DE LA RELACION (20-sep-2026, `docs/DISENO-relaciones-20sep.md` §1).
    El motivo del descarte, o None si la relacion pasa. Cuatro chequeos, en este orden:

      · `sin_evidencia_relacion`: no vino, o vino vacia. El modelo la sabia pero el texto no la
        dice: es lo que el juez objeto 28 veces sobre `medicina@4`.
      · `evidencia_larga`: mas de MAX_EVIDENCIA caracteres plegados. Ya no es "el tramo que la
        afirma" sino el fragmento, que nombra a los dos extremos trivialmente.
      · `evidencia_dos_oraciones` (v6): el tramo cruza un fin de oracion. Dos entidades en dos
        oraciones vecinas son proximidad, no afirmacion.
      · `evidencia_ajena`: no esta en el fragmento (`_tramo_del_fragmento`).
      · `evidencia_incompleta`: no nombra a los dos extremos (con sus sinonimos, como la regla de
        evidencia de las entidades). Un extremo de otro fragmento no tiene sinonimos aca y se
        busca por su nombre.
      · `polaridad`: el verbo la niega (`Relacion.negada_en`). Solo con lexicon.

    Lo que NO hace: adivinar la direccion entre tipos iguales. Si la evidencia nombra a los dos
    extremos, pasa; la direccion la juzga el juez.
    """
    ev = _plegar(evidencia or "")
    if not ev:
        return "sin_evidencia_relacion"
    if len(ev) > MAX_EVIDENCIA:
        return "evidencia_larga"
    if _RX_FIN_DE_ORACION.search(str(evidencia).strip()):
        return "evidencia_dos_oraciones"
    if not _tramo_del_fragmento(ev, fragmento, tokens):
        return "evidencia_ajena"
    ev_tokens = ev.split()
    for extremo in (desde, hasta):
        if not con_evidencia(_plegar(extremo), sinonimos_de.get(extremo, ()), ev, ev_tokens):
            return "evidencia_incompleta"
    if regla.negada_en(ev):
        return "polaridad"
    return None


def _tramo_del_fragmento(ev: str, fragmento: str, tokens: list) -> bool:
    """La evidencia (plegada) esta en el fragmento (plegado): literal, o —si el modelo la copio
    con un tropiezo de puntuacion o una palabra— con al menos UMBRAL_EVIDENCIA de sus palabras
    presentes en el fragmento y no menos de MIN_PALABRAS_EVIDENCIA palabras."""
    if ev in fragmento:
        return True
    partes = ev.split()
    if len(partes) < MIN_PALABRAS_EVIDENCIA:
        return False
    presentes = set(tokens)
    return sum(1 for p in partes if p in presentes) / len(partes) >= UMBRAL_EVIDENCIA


def _descartar(descartes, libro_id, chunk_id, motivo: str, **campos) -> None:
    fila = {"motivo": motivo, "chunk_id": chunk_id, **campos}
    if descartes is not None:
        descartes.append(fila)
    eventos.emitir(log, "extraccion_descarte", libro_id=libro_id, **fila)


def validar(tx: Taxonomia, crudo: dict, chunk: dict, *,
            rechazo_from_to: bool = RECHAZO_FROM_TO,
            rechazo_orientacion: bool = RECHAZO_ORIENTACION,
            descartes: list | None = None) -> dict:
    """Lo que devolvio el modelo, filtrado contra `tx`. Los descartes se CUENTAN.

    Cuatro diferencias con el `_validate_extraction` que reemplazo el 14-sep-2026:

      · los tipos y las relaciones se chequean contra el perfil, no contra dos sets de modulo;
      · `min_nombre`/`max_nombre` salen de `canonicalization` (eran 2 y 100 literales);
      · los `from`/`to` se CHEQUEAN y se emiten, pero no rechazan (ver `RECHAZO_FROM_TO`);
      · todo descarte deja rastro: el evento `extraccion_descarte` y la lista `descartes` del
        resultado. Antes se perdian en silencio y nadie podia medir la diferencia entre lo que
        el modelo devolvio y lo que quedo.

    Y TRES REGLAS NUEVAS DE v3 (18-sep-2026), las tres nacidas de la cola de peores del juez:

      · `orientacion` — el par viola los `from`/`to` en este sentido y los cumple dado vuelta:
        la relacion esta invertida y se RECHAZA (`RECHAZO_ORIENTACION`, encendido);
      · `nombre_de_tipo` — el nombre de una entidad, o un extremo de relacion, es exactamente el
        nombre de una CLASE del vocabulario ('tionamidas PERTENECE_A grupo_farmacologico');
      · `sin_evidencia` — la entidad no esta nombrada en el fragmento. Se descarta y el conteo
        viaja en `inferidas_descartadas`.

    LA EVIDENCIA SOLO SE EXIGE SI HAY FRAGMENTO, y hay que decirlo porque es un agujero con
    forma: `validar(tx, crudo, {"id": …, "libro_id": …})` sin `text` NO chequea evidencia,
    porque sin el texto no se puede decidir y descartar todo seria peor. El unico llamador de
    produccion (`extract_entities_fast.extract_batch`) pasa el chunk entero —el mismo dict que
    fue al prompt—, y un test lo fija. Un llamador nuevo que pase un chunk pelado se saltea la
    regla en silencio.

    LO QUE NO CAMBIA, y es una decision documentada: una relacion con UN SOLO extremo extraido
    en ESTE fragmento SOBREVIVE, como hasta hoy. El diseño proponia descartarla (§3.3, T3)
    dando por hecho que muere despues en la carga; no es cierto: `canonicalizar` funde las
    entidades de TODOS los chunks del libro, asi que el otro extremo puede ser una entidad de
    otro fragmento y la relacion entra al grafo. Descartarla aca perderia aristas reales. Se
    cuenta (`motivo="extremo_ausente"`) y se deja pasar.

    LA EVIDENCIA DE LA RELACION (20-sep-2026, `docs/DISENO-relaciones-20sep.md`): si el perfil
    declara `extraction.evidencia_relaciones`, cada relacion tiene que traer `evidencia` —el tramo
    del fragmento que la afirma— y se verifica en cuatro pasos (`_motivo_evidencia_relacion`):
    que exista, que este en el fragmento, que nombre a los dos extremos, que el verbo no la
    niegue. La evidencia VERIFICADA viaja en el artefacto (`extraction/v1` la declara opcional
    desde el 20-sep): es el linaje de la arista y lo que permite pasar lo guardado por un
    validador mejor sin pagar una llamada; al grafo no llega, porque `canonicalizar`
    reconstruye cada relacion con sus tres claves. Sin el flag la clave se ignora y se quita;
    sin fragmento (un chunk sin texto) no se exige, igual que la evidencia de las entidades.

    LOS DESCARTES NO VIAJAN EN EL RESULTADO —solo su CONTEO de inferidas—, y tambien es a
    proposito: este dict se serializa tal cual en `extracted/{libro}_entities.json`, y
    `extraction/v1` declara sus items con `additionalProperties: false`. `inferidas_descartadas`
    es una propiedad declarada del contrato desde el 18-sep-2026 y aparece SOLO cuando hay algo
    que contar, para que el artefacto de un chunk limpio no cambie de forma. El detalle de cada
    descarte se pide con `descartes=`; el rastro permanente es el evento.
    """
    libro_id = chunk.get("libro_id", "")
    chunk_id = chunk.get("id", "")
    #: El fragmento COMO LO VIO EL MODELO: el mismo recorte que hace `armar_fragmento`. Buscar la
    #: entidad en el chunk entero castigaria al modelo por lo que no se le mando.
    fragmento = _plegar((chunk.get("text") or "")[:tx.max_chars_fragmento])
    tokens = fragmento.split()

    entidades, nombres, inferidas = [], set(), 0
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
        if _plegar(nombre) in tx.nombres_de_tipo:
            _descartar(descartes, libro_id, chunk_id, "nombre_de_tipo", tipo=tipo)
            continue
        sinonimos = []
        for s in cruda.get("sinonimos") or []:
            sn = normalizar_nombre(s, tx)
            if sn and sn != nombre and len(sn) >= tx.min_nombre:
                sinonimos.append(sn)
        if fragmento and not con_evidencia(_plegar(nombre), sinonimos, fragmento, tokens):
            inferidas += 1
            _descartar(descartes, libro_id, chunk_id, "sin_evidencia", tipo=tipo)
            continue
        entidades.append({"nombre": nombre, "tipo": tipo, "sinonimos": sinonimos})
        nombres.add(nombre)

    tipo_de = {e["nombre"]: e["tipo"] for e in entidades}
    sinonimos_de = {e["nombre"]: e["sinonimos"] for e in entidades}
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
        if desde == hasta:
            # "tp SE_DIAGNOSTICA_CON tp" (v6, 20-sep): una arista de un nodo a si mismo no dice nada.
            _descartar(descartes, libro_id, chunk_id, "extremos_iguales", relacion=rid)
            continue
        de_tipo = [x for x in (desde, hasta) if _plegar(x) in tx.nombres_de_tipo]
        if de_tipo:
            _descartar(descartes, libro_id, chunk_id, "nombre_de_tipo", relacion=rid,
                       desde_tipo=tipo_de.get(desde), hasta_tipo=tipo_de.get(hasta))
            continue
        if desde not in nombres and hasta not in nombres:
            _descartar(descartes, libro_id, chunk_id, "extremo_ausente", relacion=rid,
                       desde_tipo=None, hasta_tipo=None)
            continue
        regla = tx.relaciones[rid]
        desde_tipo, hasta_tipo = tipo_de.get(desde), tipo_de.get(hasta)
        if regla.orientacion_imposible(desde_tipo, hasta_tipo):
            _descartar(descartes, libro_id, chunk_id, "orientacion", relacion=rid,
                       desde_tipo=desde_tipo, hasta_tipo=hasta_tipo)
            if rechazo_orientacion:
                continue
        elif ((desde_tipo is not None and regla.desde and desde_tipo not in regla.desde) or
              (hasta_tipo is not None and regla.hasta and hasta_tipo not in regla.hasta)):
            # Ilegal en los DOS sentidos: puede ser dato malo o regla mala, y por eso no rechaza.
            _descartar(descartes, libro_id, chunk_id, "from_to", relacion=rid,
                       desde_tipo=desde_tipo, hasta_tipo=hasta_tipo)
            if rechazo_from_to:
                continue
        # Sin fragmento no hay contra que verificar, y la regla de las entidades ya hace lo mismo:
        # no se castiga al modelo por lo que no se le mando.
        if tx.evidencia_relaciones and fragmento:
            motivo = _motivo_evidencia_relacion(cruda.get("evidencia"), desde, hasta, sinonimos_de,
                                                fragmento, tokens, regla)
            if motivo:
                _descartar(descartes, libro_id, chunk_id, motivo, relacion=rid,
                           desde_tipo=desde_tipo, hasta_tipo=hasta_tipo)
                continue
        fila = {"desde": desde, "relacion": rid, "hasta": hasta}
        if tx.evidencia_relaciones and cruda.get("evidencia"):
            # La evidencia VERIFICADA viaja en el artefacto: es el linaje de la arista y lo que
            # permite revalidar lo guardado con un lexicon mejor sin pagar una llamada (la
            # leccion del 18-sep). Al grafo no llega: `canonicalizar` reconstruye la relacion
            # con sus tres claves. Sin el flag, la clave se queda aca.
            fila["evidencia"] = str(cruda["evidencia"]).strip()
        relaciones.append(fila)

    salida = {"entidades": entidades, "relaciones": relaciones,
              "chunk_id": chunk_id, "libro_id": libro_id}
    if inferidas:
        salida["inferidas_descartadas"] = inferidas
    return salida


# ══════════════════════════════════════════════════════════════════════════════════════
# EL CHECKPOINT DE LA EXTRACCION (18-sep-2026, hallazgo E5 del banco;
# `docs/DISENO-deudas-del-banco-18sep.md` §2)
# ══════════════════════════════════════════════════════════════════════════════════════

#: Sufijo del archivo de checkpoint, al lado de `extracted/<libro>_entities.json`.
SUFIJO_CHECKPOINT = "_entities.checkpoint.jsonl"


class Checkpoint:
    """Lo ya extraido de un libro, en disco, para no volver a pagarlo.

    EL HALLAZGO QUE CIERRA. `extract_libro_fast` guardaba `extracted/<libro>_entities.json`
    recien al terminar TODOS los lotes y no lo leia nunca al arrancar: una extraccion cortada a
    mitad --la VM que se apaga, un 429 que agota los reintentos, un Ctrl-C-- se volvia a pagar
    entera, y para un tratado son cientos de llamadas a Gemini.

    LA FORMA, que el `xfail` dejaba abierta ("por lote en disco, o el estado en el job/v1"): POR
    CHUNK EN DISCO, en JSONL al lado de la salida. El job de `job/v1` es de la API y la
    extraccion corre por CLI en la notebook; el estado en el grafo obligaria a una escritura por
    lote contra una VM que vive apagada. El archivo esta en un directorio que ya existe y ya esta
    gitignoreado (`extracted/`), asi que no agrega ningun camino nuevo.

    QUE TIENE. Primera linea, la CABECERA: `libro_id`, `profile`, `model_id` y
    `taxonomia_forma`. Despues una linea por chunk extraido, escrita y `flush`eada apenas vuelve
    su lote —si el proceso muere, lo escrito esta escrito—.

    LA CABECERA ES LA CONDICION DE REANUDAR, y es lo que hace que esto no sea una trampa: una
    cosecha de `medicina@2` NO es una de `medicina@3` (el 18-sep cambiaron tres reglas del
    validador y el prompt entero). Si la cabecera no coincide, el checkpoint se DESCARTA y se
    vuelve a pagar todo, que es lo correcto. Un chunk que quedo con `error` se reintenta: se
    guarda para no perder el rastro, pero no cuenta como hecho.
    """

    def __init__(self, ruta, cabecera: dict, log=None):
        self.ruta = Path(ruta)
        self.cabecera = cabecera
        self._log = log or logging.getLogger(__name__)
        self._hechos: dict = {}
        self._fh = None

    # ── lectura ──────────────────────────────────────────────────────────────────────
    def leer(self) -> dict:
        """`{chunk_id: extraccion}` de lo ya pagado y util. Vacio si no hay o no sirve."""
        if not self.ruta.exists():
            return {}
        try:
            with open(self.ruta, encoding="utf-8") as fh:
                primera = fh.readline()
                if not primera.strip():
                    return {}
                cabecera = json.loads(primera)
                if any(cabecera.get(k) != v for k, v in self.cabecera.items()):
                    eventos.emitir(self._log, "extraccion_checkpoint_descartado",
                                   libro_id=self.cabecera.get("libro_id"),
                                   motivo="cabecera_distinta",
                                   tenia=json.dumps(cabecera, ensure_ascii=False)[:200])
                    return {}
                hechos = {}
                for linea in fh:
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        dato = json.loads(linea)
                    except ValueError:
                        # La ULTIMA linea puede estar cortada a la mitad: es exactamente el caso
                        # que este archivo existe para cubrir (el proceso murio escribiendo). Se
                        # descarta esa linea y se conserva todo lo anterior.
                        eventos.emitir(self._log, "extraccion_checkpoint_linea_rota",
                                       libro_id=self.cabecera.get("libro_id"))
                        continue
                    cid = dato.get("chunk_id")
                    if cid and not dato.get("error"):
                        hechos[cid] = dato
                self._hechos = hechos
                return dict(hechos)
        except OSError as e:
            self._log.warning(f"no se pudo leer el checkpoint {self.ruta.name}: {e}")
            return {}

    # ── escritura ────────────────────────────────────────────────────────────────────
    def abrir(self, *, reanudando: bool) -> None:
        """Deja el archivo listo para escribir. Sin `reanudando`, lo pisa con su cabecera."""
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        if reanudando and self.ruta.exists():
            self._fh = open(self.ruta, "a", encoding="utf-8")
            return
        self._fh = open(self.ruta, "w", encoding="utf-8")
        self._fh.write(json.dumps(self.cabecera, ensure_ascii=False) + "\n")
        self._fh.flush()

    def anotar(self, extracciones: list) -> None:
        """Escribe las extracciones de un lote y hace `flush`: lo que se pago, queda."""
        if self._fh is None:
            return
        for e in extracciones:
            self._fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._fh.flush()

    def cerrar(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def borrar(self) -> None:
        """Se llama SOLO cuando la corrida termino completa: el JSON final ya es la verdad."""
        self.cerrar()
        try:
            self.ruta.unlink(missing_ok=True)
        except OSError as e:
            self._log.warning(f"no se pudo borrar el checkpoint {self.ruta.name}: {e}")


def ruta_de_checkpoint(directorio, libro_id: str) -> Path:
    return Path(directorio) / f"{libro_id}{SUFIJO_CHECKPOINT}"


def cabecera_de_checkpoint(tx: Taxonomia, libro_id: str, model_id: str) -> dict:
    """Lo que tiene que coincidir para que reanudar sea legitimo. Ver `Checkpoint`."""
    return {"libro_id": libro_id, "profile": tx.perfil, "model_id": model_id,
            "taxonomia_forma": TAXONOMIA_FORMA}


# ══════════════════════════════════════════════════════════════════════════════════════
# LA CANONICALIZACION
# ══════════════════════════════════════════════════════════════════════════════════════

def chunks_de(extracciones: list) -> list:
    """Los ids de los fragmentos que ESTE artefacto cubre, en orden y sin repetir.

    ES EL ALCANCE DEL REEMPLAZO (R1, 22-sep-2026) y por eso no se deduce de las entidades ni de
    las relaciones: un fragmento que con `medicina@6` no deja NADA tiene que borrar igual lo que
    habia dejado con `@2`, y ese fragmento no aparece en ninguna de las dos listas. El unico
    lugar donde consta que se lo re-extrajo es la lista de `extractions` del artefacto.
    """
    vistos, salida = set(), []
    for ext in extracciones:
        cid = ext.get("chunk_id") or ""
        if cid and cid not in vistos:
            vistos.add(cid)
            salida.append(cid)
    return salida


def canonicalizar(tx: Taxonomia, extracciones: list, *, perfil: str | None = None) -> tuple:
    """Las menciones de todos los chunks, consolidadas en un set canonico de entidades.

    CADA RELACION SALE CON SU PROCEDENCIA (R1, 22-sep-2026, `docs/DISENO-procedencia-22sep.md`).
    Hasta hoy esta funcion se quedaba con las tres claves —`desde`, `relacion`, `hasta`— y tiraba
    de que fragmento venia cada afirmacion y la `evidencia` que `medicina@5` habia pagado, medido
    y verificado. Sin eso el grafo no sabe de donde salio ninguna de sus 204.000 aristas y
    recargar un fragmento SUMA a lo viejo en vez de reemplazarlo. Ahora cada relacion lleva
    `procedencia`: una entrada por fragmento que la afirma, con `chunk`, `libro`, `perfil` y
    `evidencia`. `cargar` las escribe en la arista.

    `perfil` pisa el `tx.perfil` para el linaje, y hay que pasarlo cuando se carga un artefacto
    LEIDO DE DISCO: lo que hay que anotar en la arista es el perfil que PRODUJO la extraccion
    (`profile` del artefacto), no el que esta activo en la instancia que la sube.

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

    `min_name_length`/`max_name_length` TAMBIEN SE APLICAN ACA (18-sep-2026), y no es redundante
    con `validar`: el camino `--upload` (`extract_entities.py`) canonicaliza un
    `extracted/*.json` LEIDO DE DISCO, sin volver a validarlo, y los artefactos viejos se
    escribieron con otros limites. Sin esto, re-subir un artefacto de julio puede crear nodos con
    un nombre de un caracter que el validador de hoy no dejaria entrar. Un nombre fuera de rango
    no deja rastro en `descartes` porque este paso no recibe chunk ni libro: el rastro es del
    validador, que es donde el descarte se puede atribuir.
    """
    entidades: dict = {}
    sinonimo_de: dict = {}

    def _admisible(nombre: str) -> bool:
        return bool(nombre) and tx.min_nombre <= len(nombre) <= tx.max_nombre

    for ext in extracciones:
        chunk_id = ext.get("chunk_id", "")
        for ent in ext.get("entidades") or []:
            nombre = ent["nombre"]
            if not _admisible(nombre):
                continue
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

    vistas, relaciones = {}, []
    for ext in extracciones:
        chunk_id = ext.get("chunk_id", "")
        libro = ext.get("libro_id") or ""
        for rel in ext.get("relaciones") or []:
            desde = sinonimo_de.get(rel["desde"], rel["desde"])
            hasta = sinonimo_de.get(rel["hasta"], rel["hasta"])
            clave = (desde, rel["relacion"], hasta)
            fila = vistas.get(clave)
            if fila is None:
                fila = {"desde": desde, "relacion": rel["relacion"], "hasta": hasta,
                        "procedencia": []}
                vistas[clave] = fila
                relaciones.append(fila)
            # UN FRAGMENTO, UNA PROCEDENCIA (R1, 22-sep-2026). Si el mismo chunk afirma dos veces
            # la misma tripleta, la segunda no agrega linaje: agregaria una entrada que despues
            # habria que sacar dos veces al reemplazar, y las cuatro listas de la arista tienen
            # que quedar alineadas posicion a posicion.
            if any(p["chunk"] == chunk_id for p in fila["procedencia"]):
                continue
            fila["procedencia"].append({
                "chunk": chunk_id, "libro": libro, "perfil": perfil or tx.perfil,
                "evidencia": str(rel.get("evidencia") or "")[:MAX_EVIDENCIA]})

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

#: LAS CUATRO LISTAS DE LA ARISTA (R1, 22-sep-2026, `docs/DISENO-procedencia-22sep.md`). Son
#: PARALELAS: la entrada `i` de las cuatro habla del MISMO fragmento. `chunks` es la clave del
#: reemplazo —y por eso no se acota nunca—, `libros` y `perfiles` dicen de que fuente y con que
#: perfil entro esa afirmacion, y `evidencias` es la oracion que la sostiene.
CLAVES_PROCEDENCIA = ("chunks", "libros", "perfiles", "evidencias")

#: CUANTAS EVIDENCIAS SE GUARDAN POR ARISTA, y el numero sale de medir y no de estimar (22-sep):
#: en `extracted-etapa0/` —798 chunks de cuatro libros, `medicina@6`— la arista mas afirmada lo
#: es por 4 fragmentos, la mediana por 1, ninguna llega a 13; la evidencia mide 72 caracteres de
#: mediana y 240 de maximo (el tope del contrato). O sea que este tope NO muerde en la practica
#: y esta para el caso patologico: una arista que cien fragmentos afirmen guardaria 24 KB de
#: texto en una sola propiedad. Pasado el tope la entrada existe igual, con la evidencia VACIA:
#: las cuatro listas siguen alineadas y `chunks` sigue COMPLETO, que es lo unico que el
#: reemplazo necesita para ser exacto.
TOPE_EVIDENCIAS = 12

#: LA MARCA DE LO HEREDADO (R1). El discriminador de verdad es `e.chunks IS NULL` —no hace falta
#: escribir nada en el grafo para saber cual arista es vieja—; esta marca es OPCIONAL, la pone el
#: guion de limpieza con su fecha, y la carga la SACA cuando una extraccion nueva vuelve a
#: afirmar esa arista: la adopcion. Una arista adoptada deja de ser heredada porque ya hay un
#: fragmento que la sostiene.
MARCA_HEREDADA = "heredado"


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


def cypher_relaciones(tx: Taxonomia, desde_label: str, tipo_rel: str, hasta_label: str) -> str:
    """El MERGE de un lote de aristas de un mismo (label, tipo, label), CON PROCEDENCIA.

    Era `MERGE (a)-[:TIPO]->(b)` y nada mas: `keys(r) = []` para las 204.000 relaciones del
    grafo, medido el 21-sep. Lo que agrega (R1, 22-sep-2026):

      · las cuatro listas paralelas de `CLAVES_PROCEDENCIA`, con una entrada por fragmento;
      · IDEMPOTENCIA POR FRAGMENTO: antes de concatenar saca de la arista las entradas de los
        fragmentos que vienen en este lote, asi que cargar dos veces lo mismo deja lo mismo y
        no una lista con todo repetido. Las entradas de OTROS fragmentos no se tocan: una
        arista que dos libros afirman no pierde la mitad porque se recargo uno;
      · `REMOVE e.origen`: la ADOPCION. Una arista heredada que esta extraccion vuelve a
        afirmar deja de ser heredada, y por eso despues de re-extraer el corpus lo que siga sin
        `chunks` es exactamente lo que ninguna extraccion actual sostiene.

    El tope de `evidencias` se aplica aca y no en Python porque la lista CRECE entre cargas: lo
    que hay que acotar es el acumulado, no lo que aporta este lote.
    """
    np = tx.name_property
    return (
        "\n                UNWIND $batch AS r\n"
        f"                MATCH (a:{desde_label} {{{np}: r.desde}})\n"
        f"                MATCH (b:{hasta_label} {{{np}: r.hasta}})\n"
        f"                MERGE (a)-[e:{tipo_rel}]->(b)\n"
        "                WITH e, r, [i IN range(0, size(coalesce(e.chunks, [])) - 1)\n"
        "                            WHERE NOT e.chunks[i] IN r.chunks] AS quedan\n"
        "                WITH e, r,\n"
        "                     [i IN quedan | e.chunks[i]] + r.chunks         AS nc,\n"
        "                     [i IN quedan | e.libros[i]] + r.libros         AS nl,\n"
        "                     [i IN quedan | e.perfiles[i]] + r.perfiles     AS nf,\n"
        "                     [i IN quedan | e.evidencias[i]] + r.evidencias AS ne\n"
        "                SET e.chunks = nc, e.libros = nl, e.perfiles = nf,\n"
        "                    e.evidencias = [i IN range(0, size(ne) - 1)\n"
        "                                    | CASE WHEN i < $tope THEN ne[i] ELSE '' END]\n"
        "                REMOVE e.origen, e.marcado_el\n                ")


def cypher_relaciones_pelado(tx: Taxonomia, desde_label: str, tipo_rel: str,
                             hasta_label: str) -> str:
    """El MERGE DE SIEMPRE, sin tocar la procedencia. Es el camino de una relacion que llega SIN
    linaje —armada a mano, o de un artefacto que no paso por `canonicalizar`—.

    POR QUE NO SE LE INVENTA UNA PROCEDENCIA VACIA. Una arista con `chunks: []` mentiria dos
    veces: diria que tiene linaje (`chunks IS NOT NULL`) y ningun fragmento la podria reemplazar
    nunca, porque no hay entrada que sacarle. La invariante que vale la pena sostener es
    `chunks IS NULL` <=> "no se sabe de donde salio", y esta funcion la respeta. Si la arista ya
    existia con procedencia, este MERGE no se la toca.
    """
    np = tx.name_property
    return (f"\n                UNWIND $batch AS r\n"
            f"                MATCH (a:{desde_label} {{{np}: r.desde}})\n"
            f"                MATCH (b:{hasta_label} {{{np}: r.hasta}})\n"
            f"                MERGE (a)-[:{tipo_rel}]->(b)\n                ")


def procedencia_en_lista(procedencia, libro_id: str, tx: Taxonomia) -> dict:
    """Las entradas de `canonicalizar` pasadas a las cuatro listas PARALELAS de la arista.

    El libro y el perfil de una entrada que no los trae se completan con los de la corrida: un
    artefacto viejo no escribia `libro_id` por extraccion, y es preferible anotar el libro que
    se esta cargando —que es cierto— a dejar la entrada sin fuente. Sin procedencia devuelve
    `{}` y el llamador usa el MERGE pelado.
    """
    if not procedencia:
        return {}
    listas: dict = {clave: [] for clave in CLAVES_PROCEDENCIA}
    for p in procedencia:
        listas["chunks"].append(p.get("chunk") or "")
        listas["libros"].append(p.get("libro") or libro_id)
        listas["perfiles"].append(p.get("perfil") or tx.perfil)
        listas["evidencias"].append(p.get("evidencia") or "")
    return listas


def cypher_purga_relaciones(tipo_rel: str) -> tuple:
    """Las DOS sentencias que le sacan a las aristas lo que aportaron ciertos fragmentos.

    Van en este orden y son dos porque Cypher no borra condicionalmente sin APOC:

      1. la arista cuyos fragmentos son TODOS del lote que se recarga se va entera (si la
         extraccion nueva la vuelve a afirmar, el MERGE la crea de nuevo un paso despues);
      2. la que ademas tiene fragmentos de afuera solo pierde las entradas del lote.

    EL PREFILTRO ES EL LIBRO, y no es cosmetico: `$chunks` puede tener 13.000 ids (Farreras) y
    `x IN $chunks` es lineal, asi que sin el `WITH` que fuerza el orden se evaluaria contra cada
    arista del tipo. Con el prefiltro solo se paga sobre las aristas que ya nombran a este libro.

    `$todos` en true ignora `$chunks` y saca TODO lo que este libro habia aportado. Lo usa el
    borrado de una fuente (`carga.borrar_libro`): ahi los fragmentos se van del grafo, y una
    entrada que apunta a un chunk que ya no existe volveria inmortal a la arista —ningun
    reemplazo futuro podria sacarsela, porque nadie va a volver a cargar ese fragmento—.
    """
    # La entrada `i` se purga si es DE ESTE LIBRO y (se borra el libro entero, o su fragmento
    # esta en el lote que se recarga). Nombrar el libro y no solo el chunk hace la regla
    # explicita en vez de depender de que el id del chunk lleve adentro el del libro.
    purgada = ("e.libros[i] = $libro AND ($todos OR e.chunks[i] IN $chunks)")
    indices = "range(0, size(coalesce(e.chunks, [])) - 1)"
    prefijo = (f"\n                MATCH ()-[e:{tipo_rel}]->()\n"
               "                WHERE $libro IN coalesce(e.libros, [])\n"
               "                WITH e\n")
    borrar = (prefijo +
              f"                WHERE all(i IN {indices} WHERE {purgada})\n"
              "                DELETE e\n                ")
    podar = (prefijo +
             f"                WHERE any(i IN {indices} WHERE {purgada})\n"
             f"                WITH e, [i IN {indices} WHERE NOT ({purgada})] AS quedan\n"
             "                SET e.chunks = [i IN quedan | e.chunks[i]],\n"
             "                    e.libros = [i IN quedan | e.libros[i]],\n"
             "                    e.perfiles = [i IN quedan | e.perfiles[i]],\n"
             "                    e.evidencias = [i IN quedan | e.evidencias[i]]\n                ")
    return borrar, podar


#: Las `MENCIONA` de los fragmentos que se recargan. Se borran antes de volver a escribirlas: un
#: fragmento que con el perfil nuevo ya no nombra a una entidad no puede seguir apuntandola.
#: No hace falta procedencia en este arco porque el chunk ES el extremo de origen.
CYPHER_PURGA_MENCIONA = (
    "\n                UNWIND $batch AS cid\n"
    "                MATCH (c:Chunk {id: cid})-[m:MENCIONA]->()\n"
    "                DELETE m\n                ")


def cargar(write, tx: Taxonomia, entidades: list, relaciones: list, libro_id: str,
           *, dev_mode: bool = False, chunks: list | None = None) -> dict:
    """Sube entidades, `MENCIONA` y relaciones al grafo. DEVUELVE LAS VIOLACIONES.

    El `except Exception: print(...)` por lote del codigo viejo se tragaba los errores de
    escritura y el paso reportaba lo CANONICALIZADO: con el grafo caido a mitad, `extract`
    informaba "N entidades" y en el grafo no habia ninguna. Ahora cada lote que no entra deja
    una fila en `violaciones` y el llamador decide (hoy: metrica de degradado, sin hacer
    fallar la ingesta, que es opcional por contrato).

    `dev_mode` por defecto **False** (era True): el sufijo `Dev` escribe un grafo paralelo que
    ninguna consulta mira. Se conserva porque `promote_dev_entities` lo usa.

    EL REEMPLAZO POR FRAGMENTO (R1, 22-sep-2026). Con `chunks` —la lista de fragmentos que el
    artefacto cubre, o sea `chunks_de(extracciones)`— la carga SUSTITUYE lo que esos fragmentos
    habian aportado en vez de sumarse a ello: borra sus `MENCIONA` y les saca a las aristas las
    entradas de procedencia de esos fragmentos, borrando la arista que se queda sin ninguna.
    Recien despues escribe lo nuevo. Correr la misma carga dos veces deja el grafo igual.

    SIN `chunks` la conducta es la de siempre —aditiva—, y es a proposito: un llamador que no
    sabe que fragmentos cubre su artefacto no puede afirmar que los esta reemplazando. Los tres
    caminos de produccion (el CLI de `ingest.py`, los dos `--upload` de los extractores) SI lo
    pasan, y hay un test que lo fija. **Con `dev_mode` tampoco se reemplaza**: ver el comentario
    de la purga.

    LO QUE EL REEMPLAZO NO HACE, dicho antes de que alguien lo suponga: no borra NODOS. Una
    entidad que ningun fragmento vuelve a nombrar se queda sin `MENCIONA` y sin aristas, pero el
    nodo sigue ahi. Borrar nodos es irreversible y puede alcanzar entidades que otra cosa del
    grafo usa (`ES_UN` a ATC/SNOMED, los DAG); va en el guion de limpieza que corre una persona,
    no en la ingesta.
    """
    sufijo = "Dev" if dev_mode else ""
    violaciones: list = []
    creadas = mencionadas = relacionadas = 0

    def _escribir_purga(cypher, parametros, contexto):
        try:
            write(cypher, parametros)
        except Exception as e:
            violaciones.append({"contexto": contexto, "n": len(parametros.get("chunks") or
                                                               parametros.get("batch") or []),
                                "error": f"{type(e).__name__}: {str(e)[:200]}"})

    def _escribir(cypher, parametros, contexto):
        try:
            write(cypher, parametros)
            return True
        except Exception as e:
            violaciones.append({"contexto": contexto, "n": len(parametros.get("batch") or []),
                                "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return False

    # ── EL REEMPLAZO, ANTES DE ESCRIBIR NADA ──────────────────────────────────────────
    # Va primero y no despues por una razon que cuesta cara si se invierte: si se purgara
    # DESPUES del MERGE, la purga se llevaria puesto lo que este mismo lote acaba de escribir
    # (sus entradas tambien son "de estos fragmentos") y la carga terminaria con el grafo vacio.
    # `dev_mode` NO REEMPLAZA, y hay que decir por que porque el default invita al error. El
    # sufijo `Dev` es de los LABELS; la purga entra por el TIPO de la arista y por el chunk, que
    # son los mismos en los dos grafos. O sea que una carga a `PatologiaDev` con `chunks` se
    # llevaria puesta la procedencia --y las `MENCIONA`-- del grafo de PRODUCCION, que es
    # exactamente lo contrario de para que existe el grafo paralelo.
    if chunks and not dev_mode:
        for i in range(0, len(chunks), LOTE):
            _escribir_purga(CYPHER_PURGA_MENCIONA, {"batch": chunks[i:i + LOTE]},
                            "purga:MENCIONA")
        # TODOS los tipos del perfil, no solo los que trae este artefacto: una arista que `@2`
        # dejo de un tipo que `@6` ya no produce tambien es "lo que este fragmento habia dicho".
        # El sufijo `Dev` es de los LABELS, no de los tipos de relacion (la carga nunca lo puso
        # en la arista): la purga tiene que nombrar el mismo tipo que el MERGE.
        for tipo_rel in tx.relaciones:
            for cypher in cypher_purga_relaciones(tipo_rel):
                _escribir_purga(cypher, {"libro": libro_id, "chunks": chunks, "todos": False},
                                f"purga:{tipo_rel}")

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
        if not (desde_l and hasta_l):
            continue
        listas = procedencia_en_lista(rel.get("procedencia"), libro_id, tx)
        grupos.setdefault((desde_l, rel["relacion"], hasta_l, bool(listas)), []).append(
            {"desde": rel["desde"], "hasta": rel["hasta"], **listas})

    for (desde_l, tipo_rel, hasta_l, con_linaje), items in grupos.items():
        cypher = (cypher_relaciones(tx, desde_l, tipo_rel, hasta_l) if con_linaje
                  else cypher_relaciones_pelado(tx, desde_l, tipo_rel, hasta_l))
        for i in range(0, len(items), LOTE):
            lote = items[i:i + LOTE]
            if _escribir(cypher, {"batch": lote, "tope": TOPE_EVIDENCIAS},
                         f"relaciones:{desde_l}-[{tipo_rel}]->{hasta_l}"):
                relacionadas += len(lote)

    return {"entities": creadas, "menciona": mencionadas, "relations": relacionadas,
            "violaciones": violaciones}


# ══════════════════════════════════════════════════════════════════════════════════════
# LO HEREDADO — las aristas que se cargaron cuando la procedencia no existia
# ══════════════════════════════════════════════════════════════════════════════════════
#
# EL PROBLEMA, dicho sin vueltas. Las 204.000 relaciones que el grafo tenia el 21-sep-2026 no
# llevan ninguna propiedad, asi que NO SE PUEDEN ATRIBUIR A UN LIBRO: no hay dato que diga cual
# de los 160 las produjo. El `fuente_libro` de la entidad es el ULTIMO libro que la cargo, no
# todos; el vecindario no dice nada; el label tampoco. Por eso «re-extraer un libro y borrar las
# relaciones sin procedencia DE ESE LIBRO» no se puede programar: la frase nombra un conjunto
# que no existe en el grafo.
#
# LO QUE SI SE PUEDE, y es lo que estas funciones arman (`docs/DISENO-procedencia-22sep.md` §4):
# una arista heredada se puede borrar cuando TODOS los libros que mencionan a alguno de sus dos
# extremos ya volvieron a cargarse con procedencia. Razon: para que un libro X hubiera producido
# esa arista, X tuvo que extraer al menos uno de los extremos en alguno de sus fragmentos, y eso
# deja `(:Book X)-[:CONTAINS]->(:Chunk)-[:MENCIONA]->(extremo)`. Si todos esos libros ya pasaron
# por la carga nueva y la arista SIGUE sin `chunks`, ninguna extraccion actual la sostiene.
#
# NADA DE ESTO CORRE SOLO. Son sentencias para el guion `testing/heredadas.py`, que las ejecuta
# una persona con snapshot hecho; borrar es irreversible.


def cypher_censo_procedencia(tipo_rel: str) -> str:
    """Cuantas aristas de un tipo tienen linaje, cuantas son heredadas y cuantas estan marcadas.
    No escribe: es el `--informe` del guion y lo unico que hace falta para decidir."""
    return (f"\n                MATCH ()-[e:{tipo_rel}]->()\n"
            "                RETURN count(e) AS total,\n"
            "                       count(CASE WHEN e.chunks IS NULL THEN 1 END) AS heredadas,\n"
            "                       count(CASE WHEN e.origen = $origen THEN 1 END) AS marcadas\n"
            "                ")


def cypher_marcar_heredadas(tipo_rel: str) -> str:
    """Le pone fecha y nombre a lo viejo. ES ADITIVO Y REVERSIBLE (un `REMOVE` lo deshace) y no
    hace falta para saber cual arista es vieja —`e.chunks IS NULL` ya lo dice—: sirve para dejar
    escrito CUANDO se hizo el corte, y para que la adopcion sea visible (la carga saca la marca
    en cuanto una extraccion nueva vuelve a afirmar esa arista)."""
    return (f"\n                MATCH ()-[e:{tipo_rel}]->()\n"
            "                WHERE e.chunks IS NULL AND e.origen IS NULL\n"
            "                SET e.origen = $origen, e.marcado_el = $fecha\n                ")


def cypher_heredadas_cerradas(tipo_rel: str, *, borrar: bool) -> str:
    """Las heredadas cuyo CIERRE DE MENCIONES esta entero dentro de `$reextraidos`.

    Con `borrar=False` las cuenta y no toca nada; con `borrar=True` las borra, que es
    irreversible. Un extremo sin una sola `MENCIONA` deja el cierre vacio y la arista NO entra:
    la regla se calla cuando no sabe, que es la unica forma de que no tenga falsos positivos.
    """
    final = "DELETE e" if borrar else "RETURN count(e) AS n"
    return (f"\n                MATCH (a)-[e:{tipo_rel}]->(b)\n"
            "                WHERE e.chunks IS NULL\n"
            "                WITH e, [a, b] AS extremos\n"
            "                UNWIND extremos AS x\n"
            "                MATCH (l:Book)-[:CONTAINS]->(:Chunk)-[:MENCIONA]->(x)\n"
            "                WITH e, collect(DISTINCT l.id) AS libros\n"
            "                WHERE size(libros) > 0\n"
            "                  AND all(lid IN libros WHERE lid IN $reextraidos)\n"
            f"                {final}\n                ")
