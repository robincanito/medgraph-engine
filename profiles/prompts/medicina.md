Eres un extractor de entidades medicas. Analiza los siguientes {n} fragmentos de texto medico y extrae TODAS las entidades y relaciones medicas de CADA fragmento.

REGLAS:
{rules}

TIPOS DE ENTIDAD:
{entities}

TIPOS DE RELACION:
{relations}

{fragments}

Responde SOLO con JSON valido. El JSON debe tener un array "resultados" con {n} elementos, uno por fragmento, en orden:
{"resultados": [{"chunk_index": 0, "entidades": [{"nombre": "...", "tipo": "...", "sinonimos": ["..."]}], "relaciones": [{"desde": "...", "relacion": "...", "hasta": "..."}]}, ...]}
