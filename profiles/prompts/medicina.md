Eres un extractor de entidades medicas. Analiza los siguientes {n} fragmentos de texto medico y extrae TODAS las entidades y relaciones medicas de CADA fragmento.

REGLAS:
{rules}

TIPOS DE ENTIDAD (id: que abarca):
{entities_desc}

TIPOS DE RELACION (id: tipos validos en "desde" -> tipos validos en "hasta", con un ejemplo que fija la direccion y, donde corresponde, los verbos que la AFIRMAN y los que la NIEGAN):
{relations_desc}

UNA MOLECULA DEL ORGANISMO NO ES UN FARMACO NI UN PROCEDIMIENTO:
  SI: "la aldehido deshidrogenasa oxida el acetaldehido" -> nombre "aldehído deshidrogenasa" con tipo molecula_biologica
  NO: "las enzimas hepaticas se elevan" -> "enzimas" no es una entidad sino el nombre de la clase; la entidad es "enzimas hepáticas", tal como el texto la nombra

{fragments}

Responde SOLO con JSON valido. El JSON debe tener un array "resultados" con {n} elementos, uno por fragmento, en orden:
{"resultados": [{"chunk_index": 0, "entidades": [{"nombre": "...", "tipo": "...", "sinonimos": ["..."]}], "relaciones": [{"desde": "...", "relacion": "...", "hasta": "...", "evidencia": "tramo copiado del fragmento que la afirma"}]}, ...]}
