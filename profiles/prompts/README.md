# `prompts/` — las plantillas de extracción

Un archivo por dominio, referenciado desde `extraction.prompt_template` del perfil y resuelto
**relativo al directorio del perfil**. Hasta el 14-sep-2026 los tres perfiles apuntaban acá y
**el directorio no existía** en ninguno de los tres repos (contrato, `medgraph/`,
`medgraph-engine/`): un campo del contrato apuntando al vacío desde el 9-sep.

## Cómo se renderiza

No es `str.format`. El renderizador (`medgraph/pipeline/extraccion.py::_render`) sustituye
**sólo** los marcadores conocidos —`{` + minúsculas/`_` + `}`— y deja intacta cualquier otra
llave. Por eso el ejemplo de JSON del final se escribe con llaves simples y legibles, y no
duplicadas como exigía el `.format()` del código viejo.

| marcador | qué renderiza |
|---|---|
| `{n}` | cuántos fragmentos van en este prompt |
| `{fragments}` | los fragmentos armados (`--- FRAGMENTO i ---`, contexto y texto recortado a `extraction.max_chars_fragmento`) |
| `{entities}` | los ids de `entities`, separados por coma, **en el orden del YAML** |
| `{entities_desc}` | una línea `- id: desc` por entidad |
| `{relations}` | los ids de `relations` con `extraer` ≠ `false`, separados por coma |
| `{relations_desc}` | una línea `- ID: a \| b -> c \| d` por relación extraíble, con sus `from`/`to`, y —si la relación declara `ejemplo`— una segunda línea indentada `    ej: …` (18-sep-2026) |
| `{rules}` | `extraction.rules`, verbatim |
| `{libro}` | el título de la fuente (los tres prompts de hoy lo llevan adentro de cada fragmento y no usan este marcador) |

Reglas de la casa:

1. **El/los salto(s) de línea final(es) del archivo se descartan.** Un archivo de texto termina
   en newline y un prompt no; sin esto la plantilla de medicina no podría ser byte a byte la del
   código que extrajo las 159K entidades.
2. **Un marcador desconocido falla cerrado**, al construir la taxonomía y antes de pagar una
   llamada: `{entidades}` (en castellano) sería un prompt que le pide al modelo una llave literal.
3. **La taxonomía no se escribe a mano en la plantilla.** Es exactamente el defecto que la
   decisión E vino a curar: la lista de tipos estaba escrita cinco veces en el repo vivo. Si una
   plantilla nombra un tipo, lo hace como ejemplo dentro de una regla, nunca como vocabulario.

## Los tres usan `{entities_desc}` y `{relations_desc}` (medicina, desde el 18-sep-2026)

Hasta `medicina@2`, `medicina.md` **era**, carácter por carácter, `extract_entities_fast.BATCH_PROMPT`
con tres agujeros, y por eso publicaba la taxonomía con `{entities}` y `{relations}`: listas planas
de ids. Tenía que serlo, porque el corpus de 159K entidades salió de ese prompt exacto y un render
que difiriera en una coma mezclaba dos cosechas en el mismo grafo sin que nada lo dijera.

**El juez mostró lo que esa compatibilidad costaba** (`medgraph/docs/E-juez-linea-de-base-14sep.md`,
14-sep-2026): `j3_relacion` 2,08 sobre 5. Con `{relations}` el modelo recibía trece ids separados
por coma —`CAUSADA_POR, SE_MANIFIESTA_CON, …`— sin una palabra de qué significan ni de para qué
lado apuntan, y con `{entities}` recibía once ids sin sus `desc`, o sea sin los ejemplos que el
perfil escribe al lado de cada tipo. El defecto más frecuente que el juez nombró fue la relación
invertida, que es exactamente lo que un prompt así no puede evitar.

Desde `medicina@3` los tres dominios usan la forma buena: `{entities_desc}` (la descripción de
cada tipo) y `{relations_desc}` (los extremos permitidos **más** el `ejemplo` que fija la
dirección). El prompt de medicina **ya no es** el que extrajo el corpus, y eso está declarado en
el encabezado de `medicina.yaml`: el testigo byte a byte de v1 sigue congelado en
`medgraph/tests/golden/extraccion/prompt_medicina_v1.txt` y un test exige que la diferencia sea
sólo la declarada.
