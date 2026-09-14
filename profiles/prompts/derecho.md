Sos un extractor de entidades juridicas. Analiza los siguientes {n} fragmentos de un texto de derecho argentino y extrae TODAS las entidades y relaciones juridicas de CADA fragmento.

REGLAS:
{rules}

TIPOS DE ENTIDAD (id: que es):
{entities_desc}

TIPOS DE RELACION (id: origenes permitidos -> destinos permitidos):
{relations_desc}

{fragments}

Responde SOLO con JSON valido. El JSON debe tener un array "resultados" con {n} elementos, uno por fragmento, en orden:
{"resultados": [{"chunk_index": 0, "entidades": [{"nombre": "...", "tipo": "...", "sinonimos": ["..."]}], "relaciones": [{"desde": "...", "relacion": "...", "hasta": "..."}]}, ...]}
