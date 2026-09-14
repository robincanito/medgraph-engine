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
| `{relations_desc}` | una línea `- ID: a \| b -> c \| d` por relación extraíble, con sus `from`/`to` |
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

## Por qué medicina usa `{entities}` y los otros dos `{entities_desc}`

`medicina.md` **es**, carácter por carácter, `extract_entities_fast.BATCH_PROMPT` con tres
agujeros. Tiene que serlo: el corpus de 159K entidades salió de ese prompt exacto, y un render
que difiera en una coma mezcla dos cosechas en el mismo grafo sin que nada lo diga. Ese prompt
publica los tipos como una lista plana, así que medicina usa `{entities}`.

`derecho.md` y `generico.md` no tienen corpus con el que ser compatibles, así que usan la forma
mejor: `{entities_desc}`, que le da al modelo la descripción de cada tipo, y `{relations_desc}`,
que le da los extremos permitidos de cada relación.
