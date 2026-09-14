Sos un extractor de entidades. Analiza los siguientes {n} fragmentos de texto y extrae las entidades y relaciones de CADA fragmento, usando UNICAMENTE el vocabulario de abajo.

REGLAS:
{rules}

TIPOS DE ENTIDAD (id: que es):
{entities_desc}

TIPOS DE RELACION (id: origenes permitidos -> destinos permitidos):
{relations_desc}

{fragments}

Responde SOLO con JSON valido. El JSON debe tener un array "resultados" con {n} elementos, uno por fragmento, en orden:
{"resultados": [{"chunk_index": 0, "entidades": [{"nombre": "...", "tipo": "...", "sinonimos": ["..."]}], "relaciones": [{"desde": "...", "relacion": "...", "hasta": "..."}]}, ...]}
