Sos un extractor de entidades juridicas. Analiza los siguientes {n} fragmentos de un texto de derecho argentino y extrae TODAS las entidades y relaciones juridicas de CADA fragmento.

REGLAS:
{rules}

TIPOS DE ENTIDAD (id: que es):
{entities_desc}

TIPOS DE RELACION (id: tipos validos en "desde" -> tipos validos en "hasta", con un ejemplo que fija la direccion y, donde corresponde, los verbos que la AFIRMAN y los que la NIEGAN):
{relations_desc}

EL NOMBRE CANONICO Y LA SUPERFICIE DEL TEXTO SON DOS COSAS:
  el fragmento dice "el art. 39 de la ley de prenda" -> nombre "articulo 39 ley 12.962", sinonimos ["art. 39", "articulo 39 de la ley de prenda"]
  el fragmento dice "la LDC"                          -> nombre "ley 24.240", sinonimos ["LDC", "ley de defensa del consumidor"]
La evidencia de cada relacion se verifica CONTRA EL FRAGMENTO, asi que sin la superficie en "sinonimos" la relacion se descarta aunque sea correcta.

{fragments}

Responde SOLO con JSON valido. El JSON debe tener un array "resultados" con {n} elementos, uno por fragmento, en orden:
{"resultados": [{"chunk_index": 0, "entidades": [{"nombre": "...", "tipo": "...", "sinonimos": ["..."]}], "relaciones": [{"desde": "...", "relacion": "...", "hasta": "...", "evidencia": "tramo copiado del fragmento que la afirma"}]}, ...]}
