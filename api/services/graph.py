"""Servicio de consultas al grafo Neo4j. Operaciones predefinidas, no queries abiertas.

LA CONEXION SALE DE `services/settings.py` (14-sep-2026) y no de cuatro `os.getenv` propios: un
valor leido en dos lugares diverge en silencio, y aca ademas el `.strip()` del saneador de bordes
es lo que salva a quien pegue una password con un `
` invisible al final.

QUE SE PODO DE ESTE ARCHIVO EN EL ESPEJO OSS: las consultas de `:UP`, `:Tema`, `:Fuente`,
`:Actividad` y `:Documento` (temas por unidad problematica, actividades de catedra y su material).
Ningun script de este repo escribe esos nodos —son de la cursada que administra la instancia
privada—, asi que aca eran funciones que devolvian [] contra cualquier grafo. Lo que queda son las
consultas sobre lo que este repo SI crea: chunks (pipeline/carga.py), entidades del perfil activo
(extract_entities.py), ontologia (ontology.py) y DAGs (load_dags.py).
"""

import logging
import threading
import time

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from services.settings import get_settings

_s = get_settings()
URI = _s.neo4j_uri
USER = _s.neo4j_username
PASSWORD = _s.neo4j_password
DATABASE = _s.neo4j_database

_keepalive_started = False

_driver = None


def _keepalive_loop():
    """Thread que mantiene la conexión a Neo4j viva con un ping cada 45 segundos."""
    while True:
        time.sleep(45)
        try:
            driver = get_driver()
            with driver.session(database=DATABASE) as session:
                session.run("RETURN 1")
        except Exception as e:
            logging.warning(f"Keep-alive ping failed: {e}")
            _reset_driver()


def get_driver():
    global _driver, _keepalive_started
    if _driver is None:
        _driver = GraphDatabase.driver(
            URI, auth=(USER, PASSWORD),
            max_connection_lifetime=300,
            max_connection_pool_size=10,
            connection_acquisition_timeout=30,
            connection_timeout=15,
        )
        if not _keepalive_started:
            t = threading.Thread(target=_keepalive_loop, daemon=True)
            t.start()
            _keepalive_started = True
            logging.info("Neo4j keep-alive thread started (45s interval)")
    return _driver


def _reset_driver():
    """Fuerza recrear el driver si la conexión se perdió."""
    global _driver
    if _driver:
        try:
            _driver.close()
        except Exception:
            pass
    _driver = None


def query(cypher: str, params: dict = None, retries: int = 2) -> list:
    for attempt in range(retries + 1):
        try:
            with get_driver().session(database=DATABASE) as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
        except (ServiceUnavailable, SessionExpired, OSError) as e:
            logging.warning(f"Neo4j query retry {attempt+1}/{retries+1}: {e}")
            _reset_driver()
            if attempt == retries:
                raise


def write(cypher: str, params: dict = None, retries: int = 2):
    """Ejecuta una query de escritura con retry."""
    for attempt in range(retries + 1):
        try:
            with get_driver().session(database=DATABASE) as session:
                session.execute_write(lambda tx: tx.run(cypher, params or {}))
                return
        except (ServiceUnavailable, SessionExpired, OSError) as e:
            logging.warning(f"Neo4j write retry {attempt+1}/{retries+1}: {e}")
            _reset_driver()
            if attempt == retries:
                raise


# === PATHOLOGY ===

def get_pathology(nombre: str) -> list:
    return query("""
    MATCH (p:Patologia)
    WHERE toLower(p.nombre) CONTAINS toLower($nombre)
    OPTIONAL MATCH (p)-[:SE_DIAGNOSTICA_CON]->(dx:MetodoDx)
    OPTIONAL MATCH (p)-[:SE_TRATA_CON]->(tx)
    OPTIONAL MATCH (p)-[:PRESENTA]->(s:Signo)
    OPTIONAL MATCH (p)-[:PREVALENTE_EN]->(ge:GrupoEtario)
    OPTIONAL MATCH (t:Tema)-[:TRATA]->(p)
    OPTIONAL MATCH (t)-[:CONTENIDO_EN]->(f:Fuente)
    RETURN p.nombre AS nombre, p.definicion AS definicion,
           p.cie10 AS cie10, p.via_transmision AS via_transmision,
           collect(DISTINCT dx.nombre) AS metodos_dx,
           collect(DISTINCT s.nombre) AS signos,
           collect(DISTINCT {titulo: f.titulo, id: f.id}) AS fuentes,
           collect(DISTINCT ge.nombre) AS grupos_etarios,
           collect(DISTINCT t.nombre) AS temas
    """, {"nombre": nombre})


def get_differential(nombre: str) -> list:
    return query("""
    MATCH (p:Patologia)
    WHERE toLower(p.nombre) CONTAINS toLower($nombre)
    MATCH (p)-[:PRESENTA]->(s:Signo)<-[:PRESENTA]-(other:Patologia)
    WHERE p <> other
    RETURN other.nombre AS patologia, other.definicion AS definicion,
           collect(DISTINCT s.nombre) AS signos_compartidos
    ORDER BY size(collect(DISTINCT s.nombre)) DESC
    """, {"nombre": nombre})


# === PROCEDURES ===

def get_procedure(nombre: str) -> list:
    return query("""
    MATCH (proc:Procedimiento)
    WHERE toLower(proc.nombre) CONTAINS toLower($nombre)
    OPTIONAL MATCH (proc)-[:EVALUA]->(param:Parametro)
    OPTIONAL MATCH (proc)-[:PUEDE_DETECTAR]->(hall:Signo)
    OPTIONAL MATCH (proc)-[:EVALUA_PUNTO]->(punto:Signo)
    OPTIONAL MATCH (proc)-[:EVALUA_GANGLIO]->(gang:Signo)
    RETURN proc.nombre AS nombre, proc.definicion AS definicion,
           proc.indicaciones AS indicaciones, proc.insumos AS insumos,
           proc.pasos_totales AS pasos_totales, proc.fuente_gus AS fuente_gus,
           collect(DISTINCT {nombre: param.nombre, normal: param.valores_normales}) AS parametros,
           collect(DISTINCT hall.nombre) AS hallazgos_detectables,
           collect(DISTINCT punto.nombre) AS puntos_dolorosos,
           collect(DISTINCT gang.nombre) AS ganglios
    """, {"nombre": nombre})


# === STATS ===

def get_stats() -> dict:
    nodos = query("MATCH (n) RETURN labels(n)[0] AS tipo, count(n) AS cantidad ORDER BY cantidad DESC")
    rels = query("MATCH ()-[r]->() RETURN type(r) AS tipo, count(r) AS cantidad ORDER BY cantidad DESC")
    return {
        "total_nodos": sum(n["cantidad"] for n in nodos),
        "total_relaciones": sum(r["cantidad"] for r in rels),
        "nodos": {n["tipo"]: n["cantidad"] for n in nodos},
        "relaciones": {r["tipo"]: r["cantidad"] for r in rels}
    }


def db_state() -> str:
    """`connected` | `down`, para GET /admin/v1/health.

    NUNCA `sleeping`: ese estado del contrato es para una instancia cuyo grafo vive apagado a
    proposito y se despierta con una consulta real. Este repo no administra infraestructura, asi
    que un grafo que no contesta es un grafo caido y se dice asi.
    """
    try:
        query("RETURN 1", retries=0)
        return "connected"
    except Exception as e:
        logging.warning(f"Neo4j no responde: {e}")
        return "down"
