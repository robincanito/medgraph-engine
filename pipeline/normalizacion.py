"""Las normalizaciones de texto del repo: UNA POR PROPOSITO (R3, 22-sep-2026).

POR QUE EXISTE. El censo de T3 (`docs/DISENO-evidencia-pubmed-T3-21sep.md` §6.2) encontro SIETE
funciones que pliegan texto y difieren entre si en NFKD contra NFD, minusculas si o no, puntuacion y
plural. Buscando las siete aparecieron DOS mas que el censo no vio —`pipeline/parseo.sin_acentos` y
`eval/_fusion.plano`—, y una de ellas es justo la mitad del contrato mas caro del repo: lo que
escribe `text_busqueda` en el grafo tiene que plegar IGUAL que lo que despues lo consulta
(`services/vector._normalize`). Dos copias de esa regla que divergen dejan el indice full-text
buscando lo que nadie indexo.

LA DECISION: no se unifican las nueve en una. Se unifican **por proposito**, porque plegar para
COMPARAR NOMBRES no es lo mismo que plegar para INDEXAR TEXTO ni que desacentuar CONSERVANDO
POSICIONES. Quedan cuatro funciones y cada diferencia esta escrita:

    sin_acentos(t)              NFKD, fuera los combinantes, CONSERVA la caja
    para_busqueda(t)            sin_acentos + minusculas          <- el contrato con `*_busqueda`
    plegar(t)                   para_busqueda + puntuacion a espacio
    sin_acentos_posicional(t)   NFD, fuera los Mn                 <- un caracter por caracter
    slug(t)                     ASCII puro, `^[a-z0-9-]+$`        <- identificadores, no texto

EL ORDEN DE `lower()` NO ERA COSMETICO, y es el hallazgo de unificar. Cuatro de las nueve hacian
`NFKD(lower(t))` y las otras `lower(NFKD(t))`. No dan lo mismo: medido sobre 41.952 codepoints,
**506 difieren**, y todos en la misma direccion — NFKD descompone en una letra MAYUSCULA y el
`lower()` de antes ya paso, asi que la salida "plegada" se queda con una mayuscula adentro. Son
caracteres que un PDF medico escupe de verdad: `℃` -> `°C` (y no `°c`), `℉`, `№` -> `No`, `℞`, las
letras modificadoras en superindice (`ᴬ`, `ᴮ`) y el alfabeto matematico de doble trazo (`ℂ`, `ℕ`).
En quien lo usa para tokenizar con `[a-z0-9]+` esa mayuscula **desaparece del resultado**: la
palabra pierde una letra. Por eso la forma unica es `lower()` DESPUES, que es la que ya tenian
`vector._normalize` y `parseo.normalize_for_search` —o sea que **el contrato con el grafo no se
mueve**: lo que cambia es que los otros cuatro se le acercan.

LO QUE **NO** SE UNIFICA, y por que:

  · `sin_acentos_posicional` (NFD, `category != "Mn"`), de `despegar_titulos`. Ahi el largo del
    texto plegado se usa como INDICE dentro del token (`pos_y = len(_sin_tildes(izquierda))`), asi
    que el plegado tiene que ser uno-a-uno por caracter. NFKD no lo es: `ﬁ` -> `fi` (1 -> 2) y
    `½` -> `1⁄2` (1 -> 3) corren las posiciones y la "y" pegada se separa en el lugar equivocado.
    NFD hace solo la descomposicion canonica y conserva el largo. Es la unica que no es un plegado
    para comparar: es un plegado para MEDIR.
  · El plural. `mesh._singular` distingue "-es/-iones" de "-is/-us" y no mutila, porque lo que sale
    de ahi se le MUESTRA a quien lee (`sin_traducir`) y viaja a PubMed; `eval/_fusion.raiz` corta la
    "s" a lo bruto porque se aplica a los DOS lados de una comparacion y es simetrica. Unificarlas
    seria elegir entre mutilar un aviso o cambiar la regla que decide que nodos se fusionan.
  · `slug`. Tiene `encode("ascii", "ignore")`, que se come lo que no es latino en vez de dejarlo
    pasar. Para un identificador (`^[a-z0-9-]+$`) eso es el requisito; para buscar texto seria
    perder el 100 % de una consulta en cirilico o griego.
  · Las palabras vacias. `mesh._CONECTORES`, `_fusion.VACIAS` y `calibrar_cobertura.VACIAS` son tres
    listas distintas porque responden tres preguntas distintas (que conector no viaja a PubMed, que
    token no cuenta para decir que dos entidades son la misma, que palabra no acredita cobertura).
    No son normalizacion: son vocabulario, y viven con su dueño.

`unicodedata.normalize("NFC", ...)` de `services/classifier.extract_text` tampoco entra: eso no
pliega nada, RECOMPONE el texto que sale del PDF antes de mandarlo al modelo.
"""
from __future__ import annotations

import re
import unicodedata

#: Todo lo que no es letra ASCII ni digito. Se aplica DESPUES de plegar, asi que a esta altura ya no
#: quedan tildes que perder: lo que cae es puntuacion, espacios repetidos y lo no latino.
_NO_ALFANUM = re.compile(r"[^a-z0-9]+")


def sin_acentos(texto: str) -> str:
    """El texto sin tildes ni dieresis, CONSERVANDO la caja.

    Es la mitad de `para_busqueda` que tambien necesita `parseo.detect_structure` para decidir si una
    linea es un encabezado (18-sep-2026, T5): ahi la comparacion se hace sobre el texto plegado
    —`CAPITULO 268` tiene que entrar por `^CAPITULO\\s+\\d+`— pero el titulo que se GUARDA es el
    original, con sus tildes.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", texto or "")
                   if not unicodedata.combining(c))


def para_busqueda(texto: str) -> str:
    """Sin acentos y en minusculas. **Es el contrato con los campos `*_busqueda` del grafo.**

    Lo que `pipeline/parseo.normalize_chunks` escribe en `text_busqueda`, `titulo_seccion_busqueda` y
    `titulo_capitulo_busqueda` sale de aca, y lo que `services/vector.search_keyword` le pregunta al
    indice full-text tambien. Si las dos puntas divergen, la busqueda lexica busca una forma que
    nadie indexo — y no falla: devuelve menos, en silencio. Por eso es UNA funcion y no dos iguales.

    `lower()` va DESPUES de descomponer, a proposito: ver el encabezado del modulo (506 codepoints).
    """
    return sin_acentos(texto).lower()


def plegar(texto: str) -> str:
    """`para_busqueda` + toda la puntuacion convertida en un espacio, sin espacios de sobra.

    La forma para COMPARAR nombres y terminos: "sindrome uremico-hemolitico" y "sindrome uremico
    hemolitico" tienen que caer en lo mismo, y "Na+/K+-ATPasa" en "na k atpasa". Encima de esto
    `eval/_fusion` agrega las grafias y el plural, y `services/mesh` su propio singular: eso es
    vocabulario de cada dominio, no plegado.
    """
    return _NO_ALFANUM.sub(" ", para_busqueda(texto)).strip()


def sin_acentos_posicional(texto: str) -> str:
    """Sin tildes UNO A UNO: el resultado tiene un caracter por cada caracter de la entrada.

    NFD y no NFKD porque acá el LARGO del resultado se usa como indice (`despegar_titulos`), y la
    descomposicion de compatibilidad cambia el largo (`ﬁ` -> `fi`, `½` -> `1⁄2`). No usarla para
    comparar: para eso estan `para_busqueda` y `plegar`, que ademas normalizan la caja.
    """
    return "".join(c for c in unicodedata.normalize("NFD", texto or "")
                   if unicodedata.category(c) != "Mn")


def slug(texto: str, tope: int = 60) -> str:
    """Texto -> identificador `^[a-z0-9-]+$`.

    ENDURECIDO EL 15-sep-2026 (hallazgo 7). Antes desacentuaba con seis `re.sub` escritos a mano
    —agudas, graves y la eñe— y NADA MAS: una dieresis, un circunflejo o una `ç` caian en el
    `[^a-z0-9]+` y se volvian un guion, asi que "Vratnica" daba `vr-tnica`. Con NFKD el acento se
    separa de la letra y `encode("ascii", "ignore")` se lo lleva, que es la misma regla para todos
    los diacriticos en vez de una lista que siempre le falta uno.

    NO es `plegar`: el `encode("ascii", "ignore")` **borra** lo que no es latino en vez de dejarlo
    pasar. Para un identificador eso es lo que se quiere; para buscar texto seria perder la consulta.

    El corte en 60 se queda: el `libro_id` viaja ADENTRO del id de cada unidad
    (`^[a-z0-9-]+_v2_[0-9]{5}$`, chunk/v1) y un titulo entero no es un identificador.
    """
    plano = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", plano.lower().strip()).strip("-")[:tope]
