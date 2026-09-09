"""La estrategia de procesamiento de un dominio: como se parsea y como se chunkea.

POR QUE EXISTE (9-sep-2026). El perfil de dominio (`nomos-contracts/profiles/*.yaml`) decidia
UNA sola cosa: que entidades extrae el LLM. Todo lo demas del pipeline estaba clavado a
medicina:

  - los patrones de estructura vivian en un diccionario indexado por `libro_id`, o sea que
    "Farreras numera sus capitulos asi" era codigo, y cada libro nuevo con estructura propia
    pedia un commit;
  - los tamaños de chunk (280 palabras, solape 60, padres de 3) eran constantes de modulo.

Y 280 palabras esta bien para prosa medica y MAL para una norma, donde el articulo ES la unidad
y partirlo destruye la cita (decision de Ivan, 4-ago-2026). Un pipeline que sirve a varios
dominios no puede tener esos numeros escritos en el codigo.

QUE HACE. `Estrategia` junta esos parametros en un objeto que viaja por el pipeline. El default
(`POR_DEFECTO`) son EXACTAMENTE los valores de medicina que ya estaban, asi que sin perfil el
comportamiento es identico al de siempre; hay tests que lo fijan. `desde_perfil()` los pisa con
lo que declare el YAML del dominio.

LO QUE TODAVIA NO ESTA (proximo corte, ver NOMOS_PIPELINE_STATE.md §12.3): el prefijo del texto
que se embebe y el modelo de nodos de la carga. El primero es delicado porque cambiar el texto
cambia el vector: exige un valor nuevo de `embedding_forma` y una re-corrida, no un cambio de
constante.
"""
from dataclasses import dataclass, field, replace

# ── Valores de medicina, los que rigen desde el principio ──────────────────────
# Vivian como constantes en parseo.py; son el default para que nada cambie sin perfil.
TARGET_SIZE = 280       # palabras objetivo por child chunk
MIN_SIZE = 150          # minimo aceptable
MAX_SIZE = 380          # maximo antes de forzar corte
OVERLAP_SIZE = 60       # palabras de solape entre chunks
PARENT_WINDOW = 3       # children por parent chunk
MAX_PARENT_WORDS = 1200 # maximo de palabras por parent

# Patrones de estructura. `_default` es el generico; las demas claves son fuentes concretas
# cuya numeracion no matchea el generico (se descubrieron una por una parseando).
PATRONES_MEDICINA = {
    "farreras-2020": {
        "capitulo": [
            r'^SECCIÓN\s+[IVXLCDM]+\b',
            r'^Capítulo\s+\d+',
            r'^CAPÍTULO\s+\d+',
        ],
        "seccion": [
            r'^\d+\.\d+[\s\.]+[A-ZÁÉÍÓÚÑ]',    # 23.4 Glaucoma...
            r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,60}$', # MODELOS DE REGRESIÓN (línea sola en mayúsculas)
        ],
    },
    "garcia-feijoo-2012": {
        "capitulo": [
            r'PA\s*R\s*T\s*E\s+\d+',             # PA R T E 1 : BÁSICO
            r'^PARTE\s+\d+',
        ],
        "seccion": [
            r'^\d+\s*\|\s*.+',                    # 1 | Embriología. Desarrollo...
            r'^\d+\.\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{3,}',  # 1. Embriología... (requiere palabra real, no "3. Mixto.")
        ],
    },
    "diamante-orl": {
        "capitulo": [
            r'^SECCIÓN\s+[IVXLCDM]+',
            r'^Sección\s+[IVXLCDM]+',
            r'^CAPÍTULO\s+\d+',
        ],
        "seccion": [
            r'^\d+\.\s+[A-ZÁÉÍÓÚÑ]',
            r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,50}$',
        ],
    },
    "_default": {
        "capitulo": [
            r'^SECCIÓN\s+[IVXLCDM]+',
            r'^CAPÍTULO\s+\d+',
            r'^Capítulo\s+\d+',
            r'^PARTE\s+\d+',
        ],
        "seccion": [
            r'^\d+\.\d+[\s\.]+[A-ZÁÉÍÓÚÑ]',
            r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,60}$',
        ],
    },
}


@dataclass(frozen=True)
class Estrategia:
    """Como procesar el material de UN dominio. Inmutable: se construye una vez por corrida."""

    # ── parseo ──
    patrones: dict = field(default_factory=lambda: dict(PATRONES_MEDICINA))
    # ── chunkeo ──
    target_size: int = TARGET_SIZE
    min_size: int = MIN_SIZE
    max_size: int = MAX_SIZE
    overlap: int = OVERLAP_SIZE
    parent_window: int = PARENT_WINDOW
    max_parent_words: int = MAX_PARENT_WORDS

    def patrones_de(self, source_id: str) -> dict:
        """Los patrones de una fuente: los suyos si los tiene declarados, si no los del dominio."""
        return self.patrones.get(source_id) or self.patrones.get("_default") or {"capitulo": [], "seccion": []}

    def con(self, **cambios) -> "Estrategia":
        """Copia con algunos parametros cambiados (util en tests y en overrides por fuente)."""
        return replace(self, **cambios)


POR_DEFECTO = Estrategia()


def desde_perfil(perfil: dict | None) -> Estrategia:
    """Estrategia declarada por un perfil de dominio (`profiles/<dominio>.yaml`).

    Formato esperado (todo opcional; lo que falte queda en el default):

        parse:
          structure_patterns:            # patrones del DOMINIO
            capitulo: [...]
            seccion: [...]
          por_fuente:                    # excepciones por fuente concreta
            farreras-2020:
              capitulo: [...]
        chunk:
          target_size: 280
          min_size: 150
          max_size: 380
          overlap: 60
          parent_window: 3
          max_parent_words: 1200

    Un perfil sin estas secciones devuelve el default: agregarlas es opt-in.
    """
    if not perfil:
        return POR_DEFECTO

    parse = perfil.get("parse") or {}
    patrones = dict(PATRONES_MEDICINA)
    if parse.get("structure_patterns"):
        patrones["_default"] = {
            "capitulo": list(parse["structure_patterns"].get("capitulo") or []),
            "seccion": list(parse["structure_patterns"].get("seccion") or []),
        }
    for fuente, pats in (parse.get("por_fuente") or {}).items():
        patrones[fuente] = {
            "capitulo": list(pats.get("capitulo") or []),
            "seccion": list(pats.get("seccion") or []),
        }

    chunk = perfil.get("chunk") or {}
    numeros = {
        clave: int(chunk[clave])
        for clave in ("target_size", "min_size", "max_size", "overlap", "parent_window", "max_parent_words")
        if clave in chunk
    }
    estrategia = Estrategia(patrones=patrones, **numeros)
    _validar(estrategia)
    return estrategia


def _validar(e: Estrategia) -> None:
    """Falla cerrado: una combinacion incoherente produce chunks basura y se descubre tarde,
    con el corpus ya cargado."""
    if not 0 < e.min_size <= e.target_size <= e.max_size:
        raise ValueError(
            f"chunk: tiene que cumplirse 0 < min_size ({e.min_size}) <= target_size "
            f"({e.target_size}) <= max_size ({e.max_size})"
        )
    if not 0 <= e.overlap < e.min_size:
        raise ValueError(
            f"chunk: overlap ({e.overlap}) tiene que ser menor que min_size ({e.min_size}); "
            "si no, un chunk es casi todo solape del anterior"
        )
    if e.parent_window < 1:
        raise ValueError(f"chunk: parent_window ({e.parent_window}) tiene que ser al menos 1")
    if e.max_parent_words < e.target_size:
        raise ValueError(
            f"chunk: max_parent_words ({e.max_parent_words}) no puede ser menor que target_size "
            f"({e.target_size}): el padre no entraria ni un hijo"
        )
