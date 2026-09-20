"""Servicio de busqueda: semantica, full-text y hybrid con RRF."""

import logging

from dotenv import load_dotenv
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from pipeline.parseo import NO_CONTENIDO
from services.graph import query
from services.settings import get_settings

load_dotenv()

_client = None

# Pool size para retrieval inicial (se fusionan despues).
#
# 60 -> 150 el 10-sep-2026, MEDIDO con eval/eval_retrieval.py sobre 101 consultas reales (corridas
# `base2` vs `pool150` en eval/resultados/). Cuatro indicadores apuntaron al mismo lado, y el que
# no es ruido es el tercero:
#   · consultas SIN un solo pasaje relevante en el pool: 3 -> 1 de 101
#   · concentracion (precision@top_k / precision@pool): 1.33 -> 1.45
#   · 17 consultas mejoran contra 9 que empeoran (75 sin cambio)
#   · contaminacion por `cie10` en el top-10: 7 -> 5 consultas
# Costo: +54 ms en la mediana y +60 ms en p95 (~6%), plano en todos los percentiles.
# `precision@top_k` sube apenas +0.009: eso SOLO es ruido, no es el argumento.
RETRIEVAL_POOL = 150

# Constante k para Reciprocal Rank Fusion.
#
# SE PROBO BAJARLA A 20 el 10-sep-2026 Y SALIO PEOR: 8 consultas mejoran contra 19 que empeoran,
# concentracion 1.33 -> 1.27, y la contaminacion por `cie10` subio de 7 a 9 consultas (corrida
# `rrf20`). Queda en 60 con la hipotesis refutada anotada, para no volver a gastar la medicion.
#
# POR QUE FALLO, que es lo util: se sospechaba que k=60 con un pool de 60 "aplana" el ranking --el
# puesto 1 pesa apenas el doble que el 60-- y que bajarla haria discriminar mas. Pasa lo contrario.
# Con k alto lo que decide es la MULTIPLICIDAD (aparecer en varios de los rankings que se fusionan),
# que es justamente la señal robusta. Bajar k le sube el peso al puesto 1 de CADA ranking por
# separado, o sea que amplifica el error de cada uno --incluido el de BM25, que pone primero al
# indice de codigos CIE-10-- en vez de corregirlo.
RRF_K = 60

# EL TOPE DE `top_k`, EN UN SOLO LUGAR (16-sep-2026). Hasta hoy `/search/*` aceptaba cualquier
# entero. Un `top_k: 5000` no rompe nada, pero recupera el pool entero y devuelve un documento
# completo en una respuesta HTTP. El tope se declara una sola vez y lo leen todas las puertas que
# exponga el despliegue.
TOPE_TOP_K = 20

# Embedding config

# VERTEX AI, no API key (migrado 18-ago-2026).
# ESTA ES LA RUTA DE CONSULTA: cada busqueda semantica embebe el texto de la pregunta, o sea que
# depende de un proveedor externo EN CADA REQUEST. De ahi salen dos obligaciones de contrato:
#   · el proveedor de embeddings es una dependencia dura de la busqueda semantica. Si deja de
#     responder -cuota agotada, credenciales vencidas, region equivocada-, la busqueda no degrada:
#     falla. Quien opere esto tiene que tratar esa cuota como parte del servicio.
#   · un chequeo de vida que no embebe nada NO mide esta ruta. Una sonda que solo mira el proceso
#     puede decir "sano" con la busqueda caida; si se quiere detectar, hay que sondear un embedding.
#
# OJO CON LA LOCATION: gemini-embedding-2 no existe en us-central1 (404).
# Vive en `global`.
_s = get_settings()
GCP_PROJECT = _s.gcp_project
GCP_LOCATION = _s.gcp_location


def get_embedding_client():
    global _client
    if _client is None:
        # Cliente unico via el pipeline canonico (Vertex, location global). Ver
        # pipeline/embeddings.py: gemini-embedding-2 NO existe en us-central1.
        from pipeline import embeddings
        _client = embeddings.crear_cliente(GCP_PROJECT, GCP_LOCATION)
    return _client


def generate_embedding(text: str) -> list:
    """Embedding de la consulta: misma llamada y mismo modelo que la ingesta.

    Delegado a pipeline.embeddings (6-sep-2026): un contenido por llamada, dict
    por texto, techo de 2000 chars — el contrato que rompio en junio y el unico
    que Vertex acepta, escrito una sola vez.
    """
    from pipeline import embeddings

    return embeddings.generate_embeddings(get_embedding_client(), [text])[0]


# =============================================================================
#  FILTROS DE FUENTE — el `WHERE` se arma UNA vez (T3.a, §3.A+E, 16-sep-2026)
# =============================================================================
#
# QUE PROBLEMA CIERRA. El modelo de peticion era `query`, `top_k` y **un** `libro_id`: para acotar
# la busqueda a un subconjunto de fuentes habia que hacer una llamada por fuente y pegar los
# resultados a mano. Con filtros de lista eso es UNA llamada.
#
# POR QUE UN SOLO ARMADOR Y NO UN `WHERE` por Cypher: hasta hoy habia CUATRO consultas de unidad
# (semantica con y sin fuente, lexica con y sin fuente) con la condicion de no-contenido copiada en
# las cuatro. Agregar cinco filtros a mano ahi es garantizar que dentro de un mes una de las cuatro
# quede sin alguno y la busqueda mienta en silencio para una sola de las rutas —el modo de falla
# mas caro que tiene este archivo (ver el BUG del 2026-07-30 en `search_keyword`)—. Ahora las
# cuatro se colapsaron en DOS (una semantica, una lexica) y las dos arman su `WHERE` con
# `_filtros_where`.
CLAVES_FILTRO = ("libro_ids", "collections", "areas", "tipos", "excluir")

#: Los tres filtros que viven en el `:Book` y no en el `:Chunk`: `b.collections` (una lista de
#: etiquetas), `b.area` y `b.source_kind`. Son las mismas propiedades que publica el catalogo de
#: fuentes, con el mismo `coalesce` por default.
FILTROS_DE_FUENTE = (
    ("collections", "$f_collections",
     "any(x IN coalesce(b.collections, []) WHERE x IN $f_collections)"),
    ("areas", "$f_areas", "b.area IN $f_areas"),
    # `coalesce` y no `b.source_kind` pelado: el catalogo publica `bibliografia` para las fuentes
    # que no tienen la propiedad, asi que filtrar por `tipos: ["bibliografia"]` tiene que encontrar
    # tambien a esas. Sin el coalesce, el filtro contradice al catalogo que lo ofrece.
    ("tipos", "$f_tipos", "coalesce(b.source_kind, 'bibliografia') IN $f_tipos"),
)

# CUANTO SE INFLA EL POOL DEL INDICE VECTORIAL SEGUN EL FILTRO.
#
# El indice de Neo4j NO sabe filtrar por propiedad: `db.index.vector.queryNodes` devuelve los K
# vecinos GLOBALES y el `WHERE` recien corre despues (post-filtro). Con un filtro restrictivo y un
# K chico eso devuelve casi siempre VACIO — es el bug del 2026-07-30 que documenta la rama de
# `libro_id`. Los factores son dos porque los filtros no pesan igual:
#
#   · RESTRICTIVO (`libro_ids`, `collections`, `areas`, `tipos`): deja pasar una fraccion CHICA del
#     corpus (una fuente de 27 unidades sobre 90.000 es el 0,03 %). Factor 60 y piso de 3.000, que
#     es el numero que ya usaba la rama con `libro_id` y que esta medido en uso.
#   · `excluir` SOLO: saca una fraccion chica y deja pasar casi todo. Inflar a 3.000 seria pagar un
#     KNN sobre el corpus entero para descartar dos fuentes. Factor 4 con piso de 300: cubre el peor
#     caso realista (excluir las fuentes mas grandes, ~25 % de las unidades) con margen.
FACTOR_POOL_RESTRINGIDO = 60
POOL_RESTRINGIDO_MIN, POOL_RESTRINGIDO_MAX = 3000, 20000
FACTOR_POOL_EXCLUIR = 4
POOL_EXCLUIR_MIN, POOL_EXCLUIR_MAX = 300, 2000

#: Los filtros que ACOTAN (los que justifican inflar el pool hasta el techo).
FILTROS_RESTRICTIVOS = ("libro_ids", "collections", "areas", "tipos")

# `calidad` y `calidad_valor` viajan en CADA resultado desde el 16-sep-2026: una calidad por
# FUENTE no se puede usar para advertir sobre el pasaje que se esta citando, que es donde el aviso
# sirve. Las escribe el parseo por unidad; mientras un corpus viejo no tenga el backfill pueden
# venir `null`, y `null` significa "no medida", que es distinto de `ok` y hay que poder
# distinguirlo.
RETORNO_CHUNK = """
    RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
           c.page_end AS pag_fin, c.text AS texto, c.word_count AS palabras,
           c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
           c.tipo_contenido AS tipo, c.parent_id AS parent_id,
           c.calidad AS calidad, c.calidad_valor AS calidad_valor, score
"""


def _lista(valor) -> list:
    """Cualquier cosa -> lista de strings no vacios. Un string suelto es una lista de uno."""
    if valor is None:
        return []
    if isinstance(valor, str):
        valor = [valor]
    return [str(x).strip() for x in valor if str(x).strip()]


def normalizar_filtros(filtros: dict | None = None, libro_id: str | None = None) -> dict:
    """`filtros` canonico: solo las claves con contenido, todas listas de strings.

    `libro_id` SE MANTIENE COMO AZUCAR de `libro_ids: [x]` (16-sep-2026): es el parametro que los
    consumidores ya usan, y romperlo por estetica no vale la pena. Si llegan los dos, `libro_id` se
    AGREGA a la lista: es un atajo, no una segunda dimension, y nadie manda las dos formas
    queriendo una interseccion.
    """
    salida = {c: _lista((filtros or {}).get(c)) for c in CLAVES_FILTRO}
    if libro_id and libro_id not in salida["libro_ids"]:
        salida["libro_ids"] = [*salida["libro_ids"], libro_id]
    return {c: v for c, v in salida.items() if v}


def _filtros_where(filtros: dict) -> tuple[list, dict]:
    """(clausulas, params) de los filtros de fuente. EL unico lugar donde se escribe ese `WHERE`.

    Las clausulas se devuelven sueltas para que cada Cypher las pegue despues de SU condicion base
    (la de no-contenido), que es la unica que no depende de la peticion.

    `EXISTS { MATCH (b:Book {id: c.libro_id}) ... }` y no un `MATCH` en el cuerpo: las tres
    propiedades de fuente viven en el `:Book` y el `:Chunk` guarda `libro_id` como string (no hay
    relacion obligatoria: `CONTAINS` se teje aparte, `carga.py:154-161`). Un subquery de existencia
    se puede pegar a un `WHERE` armado por partes; un `MATCH` obligaria a reescribir las dos
    consultas enteras segun que filtros vinieron, que es exactamente la duplicacion que se sacó.
    """
    clausulas, params = [], {}
    if filtros.get("libro_ids"):
        clausulas.append("c.libro_id IN $f_libro_ids")
        params["f_libro_ids"] = filtros["libro_ids"]
    if filtros.get("excluir"):
        clausulas.append("NOT c.libro_id IN $f_excluir")
        params["f_excluir"] = filtros["excluir"]

    de_fuente = []
    for clave, nombre_param, expresion in FILTROS_DE_FUENTE:
        if filtros.get(clave):
            de_fuente.append(expresion)
            params[nombre_param.lstrip("$")] = filtros[clave]
    if de_fuente:
        # Los tres se combinan con AND: `areas: [pediatria], tipos: [apunte]` es "apuntes DE
        # pediatria", que es lo que espera cualquiera que use dos facetas a la vez.
        clausulas.append("EXISTS { MATCH (b:Book {id: c.libro_id}) WHERE "
                         + " AND ".join(de_fuente) + " }")
    return clausulas, params


def _where(clausulas: list) -> str:
    base = "NOT coalesce(c.tipo_contenido, 'body') IN $excluidos"
    return " AND ".join([base, *clausulas])


def _pool_semantico(top_k: int, filtros: dict) -> int:
    """Cuantos vecinos pedirle al indice ANTES del post-filtro. Ver los factores de arriba."""
    if any(filtros.get(c) for c in FILTROS_RESTRICTIVOS):
        return min(max(top_k * FACTOR_POOL_RESTRINGIDO, POOL_RESTRINGIDO_MIN),
                   POOL_RESTRINGIDO_MAX)
    if filtros.get("excluir"):
        return min(max(top_k * FACTOR_POOL_EXCLUIR, POOL_EXCLUIR_MIN), POOL_EXCLUIR_MAX)
    # LO QUE NO ES CONTENIDO SE FILTRA DESPUES del indice (13-sep-2026): el indice vectorial
    # devuelve los K vecinos y no sabe filtrar por propiedad. Si se pidieran exactamente top_k y
    # despues se sacaran las referencias y el indice CIE-10, quedarian menos de top_k. Se piden de
    # mas -el no-contenido es ~3% del corpus, pero en una consulta puntual llego a ser 4 de 10- y
    # se corta despues. Mismo principio que los dos factores de arriba, con menos margen.
    return top_k + max(50, top_k // 2)


def search_semantic(query_text: str, top_k: int = 40, libro_id: str = None,
                    filtros: dict = None) -> list:
    """Búsqueda semántica con vector KNN sobre embeddings."""
    f = normalizar_filtros(filtros, libro_id)
    embedding = generate_embedding(query_text)
    clausulas, params = _filtros_where(f)

    return query(f"""
    CALL db.index.vector.queryNodes('chunk_embeddings', $pool, $embedding)
    YIELD node AS c, score
    WHERE {_where(clausulas)}
    {RETORNO_CHUNK}
    ORDER BY score DESC
    LIMIT $top_k
    """, {"embedding": embedding, "top_k": top_k, "pool": _pool_semantico(top_k, f),
          "excluidos": list(NO_CONTENIDO), **params})


def _normalize(text: str) -> str:
    """Sin acentos, lowercase — los campos *_busqueda del índice v2 están normalizados así."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# La sintaxis del QueryParser clásico de Lucene. Ninguno de estos caracteres puede llegar crudo
# desde una consulta de usuario (ver `_terminos_lucene`).
_SINTAXIS_LUCENE = str.maketrans({c: " " for c in '+-&|!(){}[]^"~*?:\\/'})


def _terminos_lucene(texto: str) -> str:
    """Consulta de usuario -> términos planos para Lucene, sin sintaxis que la reinterprete.

    BUG 2026-09-10, encontrado por eval/eval_retrieval.py: **37 de las 101 consultas del conjunto
    de evaluación traían caracteres de sintaxis Lucene**, y el texto iba crudo al parser. Dos modos
    de falla, y el segundo es el peligroso:

      · `/` y `[ ]` -> excepción del parser. `search_hybrid` la atrapa, tira la mitad keyword y
        sigue con semántica sola. Al menos queda registrado (`fallas` -> `degradado`).
      · `:` -> NO da error: Lucene lo lee como `campo:término`. "sindrome: condensacion pulmonar"
        se convierte en "buscá 'condensacion' en el campo 'sindrome'" -- campo que no existe, así
        que ese término se pierde y sólo queda "pulmonar". La búsqueda devuelve resultados
        plausibles de la fuente equivocada sin una sola señal de que algo salió mal. Eso es peor
        que el error: mentira silenciosa.

    POR QUÉ REEMPLAZAR POR ESPACIO Y NO ESCAPAR CON `\\`: el índice usa el analizador
    `standard-no-stop-words`, cuyo tokenizer (UAX#29) ya descarta la puntuación AL INDEXAR -- los
    tokens guardados nunca tienen `:` ni `/` ni `-`. Entonces partir en espacios es exactamente la
    tokenización del índice: "anti-inflamatorios" -> "anti inflamatorios", que es el par de tokens
    que el índice realmente contiene. Escapar dejaría que el analizador los borre igual, pero
    agrega los problemas del `\\` (final de cadena, doble escape) sin ganar nada.
    """
    return " ".join(texto.translate(_SINTAXIS_LUCENE).split())


def search_keyword(query_text: str, top_k: int = 40, libro_id: str = None,
                   filtros: dict = None) -> list:
    """Búsqueda full-text con scoring BM25 sobre los campos normalizados del Chunk (Lucene via Neo4j)."""
    if not query_text.strip():
        return []

    # Lucene query: OR entre términos para mayor recall,
    # el scoring BM25 se encarga de rankear mejor los que tienen más matches.
    # Normalizada porque el índice v2 indexa text_busqueda (sin acentos, lowercase).
    lucene_query = _terminos_lucene(_normalize(query_text))  # OR implícito en Lucene
    if not lucene_query:
        # La consulta era pura sintaxis ("///", "?!"). No hay nada que buscar, y mandarla vacía
        # hace explotar el parser.
        return []

    f = normalizar_filtros(filtros, libro_id)
    clausulas, params = _filtros_where(f)
    # BUG 2026-07-30: el Cypher usaba $libro_id pero el parametro nunca se agregaba a `params`
    # -> ParameterMissing, error tragado por search_hybrid, y el filtro por fuente degradaba en
    # silencio a semantica-sola. Hoy los params SALEN del mismo armador que las clausulas, asi que
    # una clausula sin su parametro es imposible por construccion: esa es la mitad del valor de
    # tener un solo lugar.
    params.update({"query": lucene_query, "top_k": top_k, "excluidos": list(NO_CONTENIDO)})

    return query(f"""
    CALL db.index.fulltext.queryNodes('busqueda_chunks_v2', $query)
    YIELD node AS c, score
    WHERE {_where(clausulas)}
    {RETORNO_CHUNK}
    ORDER BY score DESC
    LIMIT $top_k
    """, params)


def _rrf_fusion(rankings: list[list], k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion: combina múltiples rankings en uno solo.

    Para cada resultado en cada ranking, calcula score = 1/(k + rank).
    Suma los scores de todos los rankings donde aparece cada chunk.
    Chunks que aparecen en múltiples rankings suben al top.
    """
    scores = {}
    chunk_data = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking):
            chunk_id = result["id"]
            rrf_score = 1.0 / (k + rank + 1)

            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                chunk_data[chunk_id] = result
            scores[chunk_id] += rrf_score

    # Ordenar por score RRF combinado
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids:
        result = chunk_data[chunk_id]
        result["rrf_score"] = round(scores[chunk_id], 6)
        results.append(result)

    return results


# =============================================================================
#  GARANTIA POR COLECCION (T3.b, §3.B) y AGRUPACION POR PADRE (T4, §4.4)
# =============================================================================

# EL PISO DE LA GARANTIA, con nombre y con numero (16-sep-2026).
#
# QUE TIENE QUE DECIDIR. `garantizar` reserva lugares del `top_k` para las fuentes que el cliente
# nombró. Sin piso eso es un cheque en blanco: la pasada restringida SIEMPRE devuelve algo --el KNN
# vectorial no tiene umbral, le pidas lo que le pidas te da sus vecinos mas cercanos-- asi que una
# coleccion que no tiene una linea del tema igual se comeria la mitad del `top_k` con su unidad
# menos lejana. Lo que tiene que pasar es lo contrario: si la coleccion no tiene nada del tema, el
# cupo queda VACIO y la respuesta lo dice.
#
# POR QUE EL PISO NO PUEDE SER SOBRE EL `rrf_score`, que es lo primero que uno intenta (y lo que
# la primera version hizo, hasta que un test lo refuto): **RRF es por RANGO**. El primero de
# CUALQUIER ranking vale 1/(60+1), sea el mejor pasaje del corpus o el menos malo de un documento
# de cuatro paginas que no habla del tema. O sea que la pasada restringida siempre produce un
# candidato con el score maximo posible y un piso sobre `rrf_score` no distingue nada; normalizarlo
# por la cantidad de rankings tampoco, porque el problema no es la escala sino que el rango no mide
# PARECIDO.
#
# QUE SI MIDE PARECIDO: el score CRUDO de la ruta que lo trajo, comparado con el mejor que esa
# MISMA ruta consiguio en todo el corpus. Son dos comparaciones "de igual a igual":
#   · lexica  — BM25 del candidato garantizado contra el mejor BM25 de la pasada general. Mismo
#     indice y misma consulta: los BM25 son comparables entre si.
#   · semantica — coseno contra el mejor coseno de la pasada general.
# Alcanza con pasar UNA de las dos: un pasaje que el corpus entero no tiene y esta fuente si, entra
# por la ruta en la que se destaque.
#
# EL NUMERO ES 0,5: "al menos la mitad de bueno que lo mejor que el corpus podia ofrecer, medido
# con la misma vara". Es una RAZON entre dos scores de la misma ruta, asi que no depende de la
# escala del motor ni del modelo de embeddings -- que es justo lo que permite elegirlo sin una
# medicion previa. Si un despliegue quiere una garantia mas laxa o mas estricta, este es el numero.
PISO_GARANTIA_FRACCION = 0.5

#: Los dos modos de agrupacion. `None` = no agrupar, y es el default de las rutas HTTP: un
#: consumidor que ya existe espera un pasaje por lugar. Lo prende quien escribe un documento.
MODOS_AGRUPACION = ("padre", "pagina")

CYPHER_FUENTES_DE_COLECCION = """
MATCH (b:Book)
WHERE any(x IN coalesce(b.collections, []) WHERE x IN $collections)
RETURN b.id AS id
ORDER BY b.id
"""

def cupo_por_defecto(top_k: int) -> int:
    """La mitad del `top_k`, minimo 1. Es el mismo numero que usa el cupo implicito de la capa de
    bibliografia para las fuentes que sugiere el router; se nombra para que la garantia explicita
    del cliente y la implicita del router no puedan divergir."""
    return max(1, top_k // 2)


def resolver_garantia(garantizar: dict | None) -> list:
    """`{collections|libro_ids}` -> la lista de `libro_id` garantizados, sin repetidos.

    Las colecciones se resuelven contra el `:Book` (una consulta barata sobre ~170 nodos) porque
    el `:Chunk` no las tiene: el filtro por coleccion es siempre una propiedad de la fuente.
    """
    if not garantizar:
        return []
    ids = _lista(garantizar.get("libro_ids"))
    colecciones = _lista(garantizar.get("collections"))
    if colecciones:
        ids += [f["id"] for f in query(CYPHER_FUENTES_DE_COLECCION,
                                       {"collections": colecciones}) if f.get("id")]
    vistos, salida = set(), []
    for x in ids:
        if x not in vistos:
            vistos.add(x)
            salida.append(x)
    return salida


def clave_de_agrupacion(resultado: dict, modo: str | None):
    """La identidad de un resultado a los efectos de deduplicar. Sin agrupar es su `id`."""
    if modo == "padre":
        # Un chunk sin padre es su propio grupo: no se puede colapsar con nadie.
        return ("padre", resultado.get("parent_id") or resultado.get("id"))
    if modo == "pagina":
        return ("pagina", resultado.get("libro"), resultado.get("pag_inicio"))
    return ("unidad", resultado.get("id"))


def agrupar_resultados(items: list, modo: str | None) -> list:
    """Colapsa por padre o por pagina, conservando el MEJOR `rrf_score` de cada grupo.

    MEDIDO sobre 224 respuestas reales (16-sep-2026): **ningun `id` se repite** dentro de un
    `top_k` --la fusion RRF deduplica antes de cortar-- pero la PAGINA si (130 lugares de 672 en la
    muestra mas grande). Con unidades de ~280 palabras hay unas 5 por pagina y el solape entre
    consecutivas es del tamaño de la ventana de solape del chunker (solape mediano medido: 19 % en
    8-gramas, ninguno >= 50 %). O sea que no son duplicados: son pasajes distintos de un mismo
    pasaje largo. Quedarse con el primero de cada pagina TIRA material real; agrupar no tira nada.

    EL TEXTO NO SE REEMPLAZA POR EL DEL PADRE, y conviene saber por que antes de "mejorarlo". Se
    hacia hasta el 20-sep-2026 --habia un `texto_de_padres` que traia la ventana entera-- con una
    intencion razonable: dar el pasaje largo en vez de un recorte suelto. Lo tumbo la medicion
    sobre una instancia real, con 87.222 chunks con padre:

        padre 4.714 caracteres de mediana  ·  hijo 1.829  ·  tope de entrega 1.100

        del hijo llegaban 372 caracteres de media, y en el **62,8 %** de los casos no llegaba
        NADA del pasaje que matcheo

    La causa es que el texto se recorta AGUAS ABAJO y siempre DESDE EL PRINCIPIO: si la ventana
    mide cuatro veces el tope, sus primeros 1.100 caracteres casi nunca contienen al hijo. Lo que
    se entregaba era texto vecino de la misma ventana: mismo tema, misma pagina, y no lo que se
    busco. Es el peor modo de falla de un buscador, porque no se ve.

    Anclar la ventana en el hijo en vez de tomarla desde el principio se evaluo y se descarto con
    el numero: solo el 8,7 % de los hijos es mas corto que el tope, asi que anclar agregaba 22
    caracteres de contexto en el 5,8 % de los casos. Identico a no reemplazar, con codigo de mas.

    Lo que agrupar SI hace, y es lo que vale: DEDUPLICAR Y LIBERAR LUGARES. Cinco pasajes de la
    misma ventana ocupan uno y los otros cuatro se llenan con material distinto. Cada
    representante lleva `unidades` (los hijos que lo trajeron, con su score y su pagina) para que
    la cita siga siendo exacta.
    """
    if modo not in MODOS_AGRUPACION:
        return [dict(r) for r in items]

    grupos: dict = {}
    orden: list = []
    for r in items:
        clave = clave_de_agrupacion(r, modo)
        if clave not in grupos:
            grupos[clave] = []
            orden.append(clave)
        grupos[clave].append(r)

    salida = []
    for clave in orden:
        hijos = sorted(grupos[clave], key=lambda x: x.get("rrf_score") or 0, reverse=True)
        representante = dict(hijos[0])
        representante["agrupado_por"] = modo
        representante["unidades"] = [{"id": h.get("id"), "rrf_score": h.get("rrf_score"),
                                      "pag_inicio": h.get("pag_inicio")} for h in hijos]
        salida.append(representante)
    return sorted(salida, key=lambda x: x.get("rrf_score") or 0, reverse=True)


def fusionar_con_cupo(general: list, garantizados: list, top_k: int, cupo: int | None = None,
                      clave=None, max_por_fuente: int | None = None) -> list:
    """Hasta `cupo` lugares para `garantizados`, el resto de `general`, dedup, orden por rrf_score.

    LA UNICA IMPLEMENTACION (16-sep-2026). El mecanismo ya existia en la capa de bibliografia,
    para las fuentes que sugiere el router, pero vivia ahi adentro y el cliente no podia nombrar
    las suyas. Vive aca y quien lo necesite lo llama: dos copias de una regla de seleccion
    divergen, y esta ya habia costado una medicion (la del 2026-07-30 que la hizo nacer).

    POR QUE EL CUPO Y NO UN PESO. En RRF gana la MULTIPLICIDAD: un chunk de un documento corto
    aparece en las N sub-consultas globales y acumula N x ~0,016, mientras la pasada restringida a
    una fuente aporta UN ranking. Ningun peso razonable cierra esa brecha; reservar lugares si.

    `max_por_fuente` es un tope de diversidad, APAGADO por defecto: una fuente que tiene los ocho
    mejores pasajes de un tema los merece, y recortarla sin pedirlo seria empeorar la respuesta
    para que se vea mas variada.
    """
    clave = clave or (lambda r: r.get("id"))
    cupo = cupo_por_defecto(top_k) if cupo is None else max(0, cupo)
    elegidos, vistos, por_fuente = [], set(), {}

    def tomar(candidatos, limite):
        for r in candidatos:
            if len(elegidos) >= limite:
                return
            k = clave(r)
            if k in vistos:
                continue
            if max_por_fuente is not None and por_fuente.get(r.get("libro"), 0) >= max_por_fuente:
                continue
            vistos.add(k)
            por_fuente[r.get("libro")] = por_fuente.get(r.get("libro"), 0) + 1
            elegidos.append(r)

    tomar(garantizados, min(cupo, top_k))
    tomar(general, top_k)
    # Reordenar el conjunto final por score, para no dejar un bloque artificial arriba.
    return sorted(elegidos, key=lambda x: x.get("rrf_score") or 0, reverse=True)[:top_k]


def procedencia_de(top: list, pool: list, garantizados: int = 0,
                   fuentes_garantizadas: list | None = None) -> dict:
    """De donde salio lo que se devuelve, y de donde salio lo que se MIRO (16-sep-2026).

    Es una auditoria de procedencia gratis: los dos conteos ya se calculaban para la telemetria
    del evento `retrieval`, solo que morian en el log. `pool` es lo que decide si una fuente "no
    aparece porque el ranking la entierra" (esta en el pool, no en el top_k) o "no aparece porque
    no tiene material" (no esta en ninguno de los dos), que son dos problemas con dos curas
    opuestas y muy faciles de confundir desde afuera.
    """
    def contar(items):
        cuenta: dict = {}
        for r in items:
            libro = r.get("libro")
            if libro:
                cuenta[libro] = cuenta.get(libro, 0) + 1
        return dict(sorted(cuenta.items(), key=lambda kv: (-kv[1], kv[0])))

    salida = {"top_k": contar(top), "pool": contar(pool),
              "garantizados": garantizados, "candidatos_unicos": len(pool)}
    if fuentes_garantizadas is not None:
        # Se dice SIEMPRE que hubo `garantizar`, aunque el cupo haya quedado vacio: "pedi estas
        # tres fuentes y entraron cero" es informacion, y callarla deja al cliente creyendo que la
        # garantia funciono.
        salida["fuentes_garantizadas"] = list(fuentes_garantizadas)
    return salida


# LAS EXCEPCIONES QUE SIGNIFICAN "EL GRAFO NO CONTESTA" (16-sep-2026). No es una lista de errores
# feos: es la frontera entre "no se pudo preguntar" y "se pregunto y no hay". Las relanza
# `services/graph.query` despues de agotar sus reintentos. Cualquier otra cosa --el parser de la
# consulta, una cuota del proveedor de embeddings, un TimeoutError-- es una ruta rota con el grafo
# vivo, y eso se degrada, no se corta.
#
# `ConnectionError`/`OSError` NO estan a proposito: `graph.query` ya las reintenta y las relanza tal
# cual, y desde afuera no se puede distinguir un socket del grafo de uno del proveedor de
# embeddings. Si en un despliegue concreto hace falta, se agrega aca y en ningun otro lado.
EXCEPCIONES_DE_GRAFO = (ServiceUnavailable, SessionExpired)


def _es_de_grafo(e: BaseException) -> bool:
    return isinstance(e, EXCEPCIONES_DE_GRAFO)


def _falla(ruta: str, e: BaseException) -> dict:
    """Una falla de ruta, en la forma que viaja en la respuesta.

    `grafo` es lo que decide el codigo HTTP corriente arriba, asi que se calcula UNA vez y aca:
    quien consume la respuesta no tiene por que conocer la jerarquia de excepciones del driver.
    """
    return {"ruta": ruta, "error": type(e).__name__, "grafo": _es_de_grafo(e)}


def _a_relanzar(caidas: list) -> BaseException | None:
    """La primera excepcion de grafo, o None si ninguna lo es."""
    return next((e for e in caidas if _es_de_grafo(e)), None)


def _mejor_score(ranking: list) -> float:
    """El score crudo del primero de un ranking (viene ordenado por score desc). 0 si esta vacio."""
    return (ranking[0].get("score") or 0) if ranking else 0.0


def _por_id(ranking: list) -> dict:
    """id -> score crudo de ESA ruta. Es lo que el piso compara de igual a igual."""
    return {r.get("id"): (r.get("score") or 0) for r in ranking}


def pasa_el_piso(cid, puntajes_lexicos: dict, puntajes_semanticos: dict,
                 mejor_lexico: float, mejor_semantico: float) -> bool:
    """¿Este candidato garantizado se gano su lugar? Ver `PISO_GARANTIA_FRACCION`.

    Basta con pasar UNA de las dos rutas: una guia corta puede no tener el vocabulario del tratado
    (BM25 bajo) y ser exactamente el pasaje que se buscaba (coseno alto), o al reves.
    """
    lexico = puntajes_lexicos.get(cid)
    if lexico is not None and lexico >= PISO_GARANTIA_FRACCION * mejor_lexico:
        return True
    semantico = puntajes_semanticos.get(cid)
    return semantico is not None and semantico >= PISO_GARANTIA_FRACCION * mejor_semantico


def search_hybrid(query_text: str, top_k: int = 8, libro_id: str = None,
                  filtros: dict = None, garantizar: dict = None,
                  agrupar: str = None, max_por_fuente: int = None) -> dict:
    """Búsqueda híbrida con query rewriting + full-text + semántica + RRF.

    1. Query preprocessing (expansión sinónimos, clasificación intención, decomposición)
    2. Para cada sub-query: full-text top POOL + dense top POOL
    3. RRF fusion de todos los rankings
    4. Devuelve top_k resultados finales

    LO QUE EL CLIENTE PUEDE PEDIR (16-sep-2026):

      · `filtros` — `{libro_ids, collections, areas, tipos, excluir}`, todas listas y todas
        opcionales. ACOTAN la búsqueda; no tocan el ranking. `libro_id` sigue andando como azúcar.
      · `garantizar` — `{collections | libro_ids, cupo}`: además de la pasada general se corre UNA
        pasada restringida a esas fuentes, y hasta `cupo` lugares del `top_k` se les reservan **si
        pasan el piso** (`PISO_GARANTIA_FRACCION`). Si la colección no tiene nada del tema el cupo
        queda vacío y `procedencia.garantizados` lo dice.
      · `agrupar` — `"padre"` / `"pagina"` / `None`: colapsa los hits de un mismo `:ParentChunk` (o
        de una misma página) en UNO, con el texto del padre y la lista de hijos que lo trajeron.
        Los lugares liberados se llenan con grupos distintos. Por defecto `None`: un consumidor que
        ya existe espera un pasaje por lugar.
      · `max_por_fuente` — tope de diversidad, apagado por defecto.

    Y LO QUE LA RESPUESTA AGREGA: `procedencia` (conteo por fuente en el `top_k` y en el pool, una
    auditoría de procedencia gratis) y, por resultado, `calidad`/`calidad_valor`.

    QUE DEVUELVE CUANDO ALGO SE CAE (16-sep-2026). Hasta hoy los cuatro primeros casos de la tabla
    eran indistinguibles desde afuera --`results: []` y un 200--, o sea que "el corpus no tiene
    material sobre esto" y "no se pudo consultar el corpus" llegaban iguales al consumidor.

    | Caso                                   | `degradado` | `fallas`        | Devuelve / lanza        |
    |----------------------------------------|-------------|-----------------|-------------------------|
    | Todo anduvo, hay material              | `None`      | `[]`            | dict con `results`      |
    | Todo anduvo, no hay material           | `None`      | `[]`            | dict con `results: []`  |
    | Una ruta caída, la otra trajo algo     | `"parcial"` | la ruta caída   | dict con `results`      |
    | Cayeron rutas, ninguna es del grafo    | `"total"`   | las caídas      | dict con `results: []`  |
    | Cayeron TODAS y alguna es del grafo    | —           | —               | **lanza** la del grafo  |

    La última fila es la que cambia el contrato: `search_hybrid` LANZA en vez de devolver un vacío
    mudo, y decide QUIEN LLAMA (`routers/search.py` lo convierte en 503 con `Retry-After`). Se lanza
    sólo si fallaron TODAS las rutas: con una viva hay material que devolver, y eso vale más que el
    código de error.

    `fallas` es aditivo: `[{"ruta": "keyword"|"semantic", "error": "<NombreDeExcepcion>",
    "grafo": bool}]`.
    """
    from services.query import preprocess

    # 1. Preprocess query
    processed = preprocess(query_text)
    sub_queries = processed["sub_queries"]

    f = normalizar_filtros(filtros, libro_id)

    all_rankings = []
    total_keyword = 0
    total_semantic = 0

    # 2. Multi-retrieval EN PARALELO: N keyword (una por sub-query) + 1 semantica.
    # Antes corrian en secuencia (hasta 4 viajes seriales a Neo4j/Gemini).
    from concurrent.futures import ThreadPoolExecutor

    from services import telemetria

    # Las dos rutas de busqueda se tragan sus errores para degradar en vez de romper. Bien, pero
    # sin registrarlo, una caida del grafo devuelve `results: []` — indistinguible de "no hay
    # nada sobre esto". El cliente entonces contesta de su propio conocimiento SIN AVISAR que la
    # bibliografia no se consulto, que en una herramienta cuya promesa es "cita libro y pagina"
    # es un fallo de correccion, no de disponibilidad. Se anotan para poder distinguirlas.
    fallas: list = []
    # Y LAS EXCEPCIONES TAL CUAL, no solo su nombre (16-sep-2026): si todas las rutas se cayeron y
    # alguna fue del grafo, hay que RELANZAR una --con su mensaje-- en vez de fabricar una nueva.
    # El nombre en `fallas` es para el que lee la respuesta; el objeto es para el que arma el 503.
    caidas: list = []
    # Cuantas rutas se dispararon: N sub-queries lexicas + 1 semantica. Es el denominador de "se
    # cayeron TODAS", que es lo unico que justifica cortar en vez de degradar.
    rutas_totales = len(sub_queries) + 1

    # LA PASADA GARANTIZADA SE RESUELVE ANTES DE ABRIR EL POOL (T3.b): `resolver_garantia` es una
    # consulta al `:Book` y, si el grafo no contesta, cae en el mismo lugar que las demas rutas en
    # vez de tumbar la busqueda entera. Sin fuentes resueltas no hay pasada restringida y el cupo
    # queda vacio, que es exactamente lo que el diseño pide para "la coleccion no existe".
    fuentes_garantizadas: list = []
    if garantizar:
        try:
            fuentes_garantizadas = resolver_garantia(garantizar)
        except Exception as e:
            logging.error(f"No se pudieron resolver las fuentes garantizadas: {type(e).__name__}")
            fallas.append(_falla("garantia", e))

    filtros_garantia = None
    if fuentes_garantizadas:
        # La garantia ACOTA: los otros filtros siguen valiendo (una garantia de coleccion dentro de
        # un `excluir` no puede resucitar lo excluido), pero `libro_ids` pasa a ser el de la
        # garantia --intersecado con el que ya venia, si venia--.
        previos = f.get("libro_ids")
        ids = [x for x in fuentes_garantizadas if x in previos] if previos else fuentes_garantizadas
        filtros_garantia = {**f, "libro_ids": ids} if ids else None

    # Los rankings se guardan TAMBIEN por ruta (no solo fusionados): el piso de la garantia
    # compara scores crudos de la MISMA ruta, y `_rrf_fusion` los pierde de vista.
    rankings_lexicos: list = []
    ranking_semantico: list = []
    g_lexico: list = []
    g_semantico: list = []
    n_hilos = len(sub_queries) + 1 + (2 if filtros_garantia else 0)

    with ThreadPoolExecutor(max_workers=n_hilos) as pool:
        kw_futures = [(sq, pool.submit(search_keyword, sq, RETRIEVAL_POOL, None, f))
                      for sq in sub_queries]
        sem_future = pool.submit(search_semantic, processed["expandida"], RETRIEVAL_POOL, None, f)
        if filtros_garantia:
            g_kw = pool.submit(search_keyword, sub_queries[0], RETRIEVAL_POOL, None, filtros_garantia)
            g_sem = pool.submit(search_semantic, processed["expandida"], RETRIEVAL_POOL, None,
                                filtros_garantia)

        for sq, fut in kw_futures:
            try:
                kw = fut.result()
                if kw:
                    all_rankings.append(kw)
                    rankings_lexicos.append(kw)
                    total_keyword += len(kw)
            except Exception as e:
                logging.error(f"Keyword search failed for '{sq}': {type(e).__name__}: {str(e)[:100]}")
                fallas.append(_falla("keyword", e))
                caidas.append(e)

        try:
            sem = sem_future.result()
            if sem:
                all_rankings.append(sem)
                ranking_semantico = sem
                total_semantic += len(sem)
        except Exception as e:
            logging.error(f"Semantic search failed: {type(e).__name__}: {str(e)[:100]}")
            fallas.append(_falla("semantic", e))
            caidas.append(e)

        if filtros_garantia:
            # Las dos rutas de la pasada restringida NO cuentan para `rutas_totales`: que se caiga
            # la garantia deja la busqueda general en pie, y eso es un hueco, no un corte.
            for etiqueta, futuro in (("garantia:keyword", g_kw), ("garantia:semantic", g_sem)):
                try:
                    r = futuro.result()
                except Exception as e:
                    logging.error(f"Pasada garantizada '{etiqueta}': {type(e).__name__}")
                    fallas.append(_falla(etiqueta, e))
                    continue
                if not r:
                    continue
                if etiqueta.endswith("keyword"):
                    g_lexico = r
                else:
                    g_semantico = r

    if not all_rankings:
        if fallas:
            telemetria.registrar("mcp_degradado", tool="search_hybrid", capa="retrieval",
                                 rutas_caidas=[f"{f_['ruta']}:{f_['error']}" for f_ in fallas],
                                 detalle="sin resultados PORQUE fallo el retrieval, no porque no haya material")
        # EL VACIO DEJA DE SER LA UNICA SALIDA (16-sep-2026). Si se cayeron TODAS las rutas y al
        # menos una fue del grafo, quien llama tiene que poder contestar 503: devolver `results: []`
        # con 200 obliga al consumidor a adivinar. Se relanza DESPUES de registrar la telemetria.
        rutas_caidas = [f_ for f_ in fallas if not str(f_["ruta"]).startswith("garantia")]
        if len(rutas_caidas) == rutas_totales and (exc := _a_relanzar(caidas)) is not None:
            raise exc
        return {
            "keyword_count": 0,
            "semantic_count": 0,
            "intencion": processed["intencion"],
            "query_expandida": processed["expandida"],
            "results": [],
            "procedencia": procedencia_de([], [], 0,
                                          fuentes_garantizadas if garantizar else None),
            # SUBE CON LA RESPUESTA, no se queda solo en el log (10-sep-2026): quien pregunta
            # tiene derecho a saber que el corpus no se pudo consultar. Sin esto, una lista vacia
            # por caida se lee igual que "no hay material".
            "degradado": "total" if fallas else None,
            "fallas": fallas,
        }

    if fallas:
        # Degradacion PARCIAL: una ruta respondio y la otra no. Hay resultados, pero peores que
        # los normales, y sin esto nadie se entera nunca.
        telemetria.registrar("mcp_degradado", tool="search_hybrid", capa="retrieval_parcial",
                             rutas_caidas=[f"{f_['ruta']}:{f_['error']}" for f_ in fallas])

    # 3. RRF fusion de todos los rankings
    fused = _rrf_fusion(all_rankings)

    # 4. La pasada garantizada, con su piso (T3.b). El ORDEN entre los garantizados lo da RRF (es
    # bueno para ordenar), pero QUIEN entra lo decide el piso sobre los scores crudos de cada ruta:
    # ver `PISO_GARANTIA_FRACCION` para por que el rango no sirve de umbral.
    garantizados: list = []
    rankings_garantia = [r for r in (g_lexico, g_semantico) if r]
    if rankings_garantia:
        mejor_lexico = max((_mejor_score(rk) for rk in rankings_lexicos), default=0.0)
        mejor_semantico = _mejor_score(ranking_semantico)
        puntajes_lexicos, puntajes_semanticos = _por_id(g_lexico), _por_id(g_semantico)
        garantizados = [r for r in _rrf_fusion(rankings_garantia)
                        if pasa_el_piso(r.get("id"), puntajes_lexicos, puntajes_semanticos,
                                        mejor_lexico, mejor_semantico)]
        # Un chunk que TAMBIEN salio en la pasada general se queda con el score de la general, que
        # es mas alto (mas rankings donde sumar). Sin esto el mejor resultado de la busqueda podria
        # aparecer ultimo por el solo hecho de estar garantizado, que es lo contrario de lo pedido.
        por_id = {r.get("id"): r for r in fused}
        for r in garantizados:
            gemelo = por_id.get(r.get("id"))
            if gemelo is not None and (gemelo.get("rrf_score") or 0) > (r.get("rrf_score") or 0):
                r["rrf_score"] = gemelo["rrf_score"]
            r["garantizado"] = True

    # 5. Agrupacion (T4) ANTES de cortar: los lugares que libera un grupo se llenan con el
    # siguiente candidato DISTINTO, que es el punto. Agrupar despues del corte solo achicaria la
    # respuesta.
    candidatos = agrupar_resultados(fused, agrupar)
    candidatos_g = agrupar_resultados(garantizados, agrupar) if garantizados else []

    # 6. Seleccion final: cupo para los garantizados, resto de la general, dedup por la clave de
    # agrupacion (dos representantes del mismo padre son el mismo lugar), orden por rrf_score.
    cupo = None
    if garantizar and garantizar.get("cupo"):
        cupo = int(garantizar["cupo"])
    results = fusionar_con_cupo(candidatos, candidatos_g, top_k, cupo,
                                clave=lambda r: clave_de_agrupacion(r, agrupar),
                                max_por_fuente=max_por_fuente)

    # EL POOL ES LA UNION DE LO QUE SE MIRO: la pasada general mas la garantizada. Se cuenta sobre
    # las unidades crudas y no sobre los grupos, porque "cuantos pasajes de esta fuente entraron al
    # pool" es la pregunta que se quiere contestar y agrupar la contestaria mas chica de lo que es.
    ids_general = {r.get("id") for r in fused}
    pool_total = fused + [r for r in garantizados if r.get("id") not in ids_general]
    procedencia = procedencia_de(results, pool_total,
                                 sum(1 for r in results if r.get("garantizado")),
                                 fuentes_garantizadas if garantizar else None)

    # CUANTO MATERIAL SE ESTA TIRANDO (paso 0 de MEDGRAPH_RERANKER_PLAN.md, 10-sep-2026).
    # Se recuperan hasta RETRIEVAL_POOL por ranking y se cortan `top_k`: hoy eso es descartar
    # ~92-96% de lo traido sin volver a mirarlo. Un reranker solo sirve si lo relevante ESTA en el
    # pool pero mal rankeado; si ya viene en el top_k, no hay nada que ganar. Estos tres numeros
    # son lo que permite decidirlo sin comprometerse a nada.
    #
    # `rrf_top` va porque `RRF_K = 60` con un pool de 60 aplana el ranking -- el puesto 1 pesa
    # 1/61 y el 60 pesa 1/120, o sea que el primero vale apenas el doble que el ultimo-- y lo que
    # termina decidiendo es la MULTIPLICIDAD. Ver la distribucion es lo que confirma o refuta eso.
    #
    # `n_candidatos_unicos` SALE DE `procedencia` (16-sep-2026): el conteo se hacia aca y moria en
    # el log mientras el cliente, que es quien necesita la auditoria de procedencia, no lo veia.
    # Ahora se calcula UNA vez y los dos leen lo mismo.
    telemetria.registrar(
        "retrieval",
        n_sub_queries=len(sub_queries),
        n_rankings=len(all_rankings),
        n_candidatos_unicos=procedencia["candidatos_unicos"],
        top_k=top_k,
        descartados=max(0, procedencia["candidatos_unicos"] - len(results)),
        rrf_top=[round(r.get("rrf_score", 0), 5) for r in results[:5]],
        rrf_k=RRF_K,
        pool=RETRIEVAL_POOL,
        garantizados=procedencia["garantizados"],
        agrupado_por=agrupar,
    )

    return {
        "keyword_count": total_keyword,
        "semantic_count": total_semantic,
        "intencion": processed["intencion"],
        "query_expandida": processed["expandida"],
        "results": results,
        "procedencia": procedencia,
        # "parcial": hay resultados, pero una de las dos rutas de busqueda no contesto, asi que
        # son peores que los normales. Vale decirlo aunque no este vacio.
        "degradado": "parcial" if fallas else None,
        "fallas": fallas,
    }
