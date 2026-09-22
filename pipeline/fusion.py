"""EL CYPHER QUE FUNDE DOS NODOS EN UNO. Una sola copia (R3, 22-sep-2026).

POR QUE EXISTE. El repo tenia DOS implementaciones de "fundir entidades duplicadas y transferirles
las aristas": `eval/fusionar_entidades.py` (la del piloto del 21-sep, con su Cypher a mano) y
`dedup_entities.py` (el guion por acentos, que nunca se corrio). Dos copias de la regla que decide
que nodos se unen en el grafo es el bug de `nomos_pipeline_duplicado` aplicado a lo IRREVERSIBLE:
la que se arregla no es la que se corre. Desde R3 las dos llaman a `sentencias_de_grupo`.

LO QUE ESTA COPIA ARREGLA Y NINGUNA DE LAS DOS HACIA — la deuda que R1 dejo anotada
(`docs/DISENO-procedencia-22sep.md` §5): **la procedencia de la arista no se fundia.**

  · `dedup_entities` hacia `MERGE (canon)-[:TIPO]->(t)` y no copiaba **ninguna** propiedad: el
    `libro`, el `chunk`, el `perfil` y la `evidencia` del duplicado se perdian enteros.
  · `fusionar_entidades` hacia `ON CREATE SET nr += properties(r)`: salvaba el caso de la arista
    nueva y perdia el otro — cuando el canonico YA tenia esa arista, la procedencia del duplicado
    se iba con el nodo. Y ese es justo el caso frecuente: dos grafias del mismo concepto sostenidas
    por dos libros distintos.

Medido el 22-sep contra produccion: hoy **cero** aristas tienen `chunks`, asi que fundir sin esto
no rompe nada TODAVIA. Pero R1 acaba de construir el linaje y la primera re-extraccion lo va a
escribir; fundir despues de eso, con cualquiera de las dos versiones viejas, lo borraba en silencio.
Por eso se arregla antes de correr nada, que es el orden que pidio la tanda.

COMO SE FUNDEN LAS CUATRO LISTAS. `chunks`, `libros`, `perfiles` y `evidencias` son PARALELAS: la
entrada `i` de las cuatro habla del mismo fragmento (§1 del diseño de procedencia). Asi que no se
pueden unir por separado —eso las desalinearia— sino por INDICE: se recorre `chunks` del duplicado,
se saltean los fragmentos que el canonico ya tiene, y de los que quedan se copian las cuatro
entradas juntas. El salteo es lo que hace la fusion idempotente y lo que impide que un fragmento
cuente dos veces cuando dos duplicados afirman la misma arista.

LO QUE NO SE TOCA SI NO HAY PROCEDENCIA (`WHERE size(ch) > 0`). Una arista heredada se reconoce por
`r.chunks IS NULL` (§3.2 del diseño): escribirle una lista vacia la haria pasar por "re-extraida" y
`testing/heredadas.py` dejaria de encontrarla. Asi que cuando no hay nada que fundir, no se escribe.

LAS DOS REGLAS DE SIEMPRE, que este modulo NO decide (las decide quien arma el grupo): nunca entre
labels distintos y nunca por modificador. Ver `eval/_fusion.clave_fusion`.
"""
from __future__ import annotations

import re

#: Un tipo de relacion se interpola en el Cypher (no se puede parametrizar), asi que se valida
#: contra lo que el grafo puede tener como tipo. Cualquier cosa rara aborta el grupo entero.
TIPO_VALIDO = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: Las cuatro listas paralelas de `docs/DISENO-procedencia-22sep.md` §1. El orden importa: `chunks`
#: es la que manda (es la que se recorre y la que decide que entradas entran).
PROCEDENCIA = ("chunks", "libros", "perfiles", "evidencias")

#: El bloque que funde la procedencia, comun a las dos direcciones. `ch0..ev0` son las listas que el
#: canonico ya tiene (despues del `ON CREATE SET`, o sea que si la arista se acaba de crear ya son
#: las del duplicado y el `reduce` no agrega nada); `ch..ev` son las del duplicado concatenadas.
_FUNDIR_PROCEDENCIA = """
    WITH nr, ch, li, pe, ev,
         coalesce(nr.chunks, []) AS ch0, coalesce(nr.libros, []) AS li0,
         coalesce(nr.perfiles, []) AS pe0, coalesce(nr.evidencias, []) AS ev0
    WHERE size(ch) > 0
    WITH nr, ch, li, pe, ev, ch0, li0, pe0, ev0,
         reduce(a = {vistos: ch0, idx: []}, i IN range(0, size(ch) - 1) |
                CASE WHEN ch[i] IN a.vistos THEN a
                     ELSE {vistos: a.vistos + ch[i], idx: a.idx + i} END) AS acu
    SET nr.chunks     = ch0 + [i IN acu.idx | ch[i]],
        nr.libros     = li0 + [i IN acu.idx | coalesce(li[i], '')],
        nr.perfiles   = pe0 + [i IN acu.idx | coalesce(pe[i], '')],
        nr.evidencias = ev0 + [i IN acu.idx | coalesce(ev[i], '')]"""

#: Las cuatro listas del lote de duplicados, concatenadas, mas las propiedades de la PRIMERA arista
#: (las que se copian si la arista hay que crearla). Se agrega por (canonico, otro extremo) para que
#: cada arista destino se toque en UNA sola fila: leer `nr.chunks` en una fila y escribirlo en otra
#: del mismo query es lo que no se puede hacer sin depender del plan de ejecucion.
_JUNTAR = """
    WITH c, o, collect(r) AS rs
    WITH c, o, rs, head([x IN rs | properties(x)]) AS props,
         reduce(a = [], x IN rs | a + coalesce(x.chunks, []))     AS ch,
         reduce(a = [], x IN rs | a + coalesce(x.libros, []))     AS li,
         reduce(a = [], x IN rs | a + coalesce(x.perfiles, []))   AS pe,
         reduce(a = [], x IN rs | a + coalesce(x.evidencias, [])) AS ev"""


def cypher_reapuntar(tipo: str, *, entrante: bool) -> str:
    """El Cypher que mueve al canonico las aristas `tipo` de los duplicados, en una direccion.

    `ON CREATE SET nr += props` y no `SET`: si el canonico ya tenia esa arista, sus propiedades
    ganan (es la regla del piloto y no cambia). Lo que SI cambia es que la procedencia no se decide
    por esa regla —no es una propiedad que "gane" una punta, es un registro de quien afirmo que— y
    por eso se funde aparte, siempre.

    No hay `DELETE r`: las aristas del duplicado se van con el `DETACH DELETE` del final, que es lo
    que tambien se lleva las que colapsan en un bucle (contadas en el plan).
    """
    patron = ("MATCH (o)-[r:{t}]->(d)" if entrante else "MATCH (d)-[r:{t}]->(o)").format(t=tipo)
    destino = f"MERGE (o)-[nr:{tipo}]->(c)" if entrante else f"MERGE (c)-[nr:{tipo}]->(o)"
    return f"""
    MATCH (c) WHERE elementId(c) = $canon
    {patron}
    WHERE elementId(d) IN $dups AND NOT elementId(o) IN $grupo{_JUNTAR}
    {destino}
      ON CREATE SET nr += props{_FUNDIR_PROCEDENCIA}"""


#: `freq` y `sinonimos` SE CALCULAN DENTRO de la transaccion, a partir de los duplicados que todavia
#: existen: si se pasaran ya sumados y la transaccion se repitiera, `freq` contaria doble. El
#: `+ [null]` es para que el UNWIND no se coma la fila cuando no hay ni un sinonimo.
CYPHER_CANONICO = """
    MATCH (c) WHERE elementId(c) = $canon
    OPTIONAL MATCH (d) WHERE elementId(d) IN $dups
    WITH c, collect(d) AS ds
    WITH c, reduce(t = coalesce(c.freq, 0), d IN ds | t + coalesce(d.freq, 0)) AS freq,
         coalesce(c.sinonimos, []) + [d IN ds | coalesce(d.nombre, d.name)] +
         reduce(acc = [], d IN ds | acc + coalesce(d.sinonimos, [])) + [null] AS crudos
    UNWIND crudos AS s
    WITH c, freq, collect(DISTINCT s) AS sins
    SET c.freq = freq,
        c.sinonimos = [x IN sins WHERE x IS NOT NULL AND x <> coalesce(c.nombre, c.name, '')]"""

#: DETACH y no DELETE: lo unico que puede quedarle colgado a un duplicado son las aristas que
#: colapsan (el otro extremo esta en el grupo), y esas se pierden a proposito.
CYPHER_BORRAR = """
    MATCH (d) WHERE elementId(d) IN $dups
    DETACH DELETE d"""


def sentencias_de_grupo(canonico: str, duplicados: list[str], tipos) -> list[dict]:
    """Las sentencias de UN grupo, en orden, para correr en UNA transaccion.

    Funcion PURA (no toca la base): el test las lee sin grafo, y el banco las corre contra un Neo4j
    de verdad (`tests/test_escenarios/test_familia_e_fusion.py`).

    Un grupo a medio fusionar es peor que un grupo sin fusionar, asi que las sentencias van juntas y
    el llamador las corre en una sola transaccion.
    """
    tipos = sorted(set(tipos))
    malos = [t for t in tipos if not TIPO_VALIDO.match(t)]
    if malos:
        raise ValueError(f"tipo de relacion inesperado {malos!r}: no se interpola en el Cypher")
    p = {"canon": canonico, "dups": list(duplicados),
         "grupo": list(duplicados) + [canonico]}
    sent = []
    for tipo in tipos:
        sent.append({"nombre": "reapuntar_salientes", "tipo": tipo, "params": p,
                     "cypher": cypher_reapuntar(tipo, entrante=False)})
        sent.append({"nombre": "reapuntar_entrantes", "tipo": tipo, "params": p,
                     "cypher": cypher_reapuntar(tipo, entrante=True)})
    sent.append({"nombre": "canonico", "tipo": None, "params": p, "cypher": CYPHER_CANONICO})
    sent.append({"nombre": "borrar_dups", "tipo": None, "params": p, "cypher": CYPHER_BORRAR})
    return sent


def transaccion_con(driver, base: str):
    """`transaccion(sentencias)` sobre ESE driver: todas las sentencias en UNA transaccion.

    Recibe el driver por parametro para que el banco de QA (`tests/harness/grafo.py`, un Neo4j en
    Docker) pueda validar el Cypher DE VERDAD, y para que ningun guion de fusion tenga escrito
    adentro a que base se conecta.
    """

    def transaccion(sentencias: list[dict]) -> None:
        def _correr(tx):
            for s in sentencias:
                tx.run(s["cypher"], s["params"])

        with driver.session(database=base) as sesion:
            sesion.execute_write(_correr)

    return transaccion
