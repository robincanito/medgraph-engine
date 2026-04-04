"""Servicio de consultas al grafo Neo4j. Operaciones predefinidas, no queries abiertas."""

import os
import logging
import threading
import time
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE")

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


# === TOPICS ===

def get_topics_by_up(up_id: str) -> list:
    return query("""
    MATCH (t:Tema)-[:PERTENECE_A]->(up:UP {id: $up_id})
    OPTIONAL MATCH (t)-[:CONTENIDO_EN]->(f:Fuente)
    OPTIONAL MATCH (t)-[:AREA_DE]->(e:Especialidad)
    RETURN t.nombre AS tema, t.descripcion AS descripcion,
           e.nombre AS especialidad,
           collect(DISTINCT {titulo: f.titulo, id: f.id}) AS fuentes
    ORDER BY e.nombre, t.nombre
    """, {"up_id": up_id})


def get_related_topics(up_id: str) -> list:
    return query("""
    MATCH (up1:UP {id: $up_id})<-[:PERTENECE_A]-(t1:Tema)-[:TRATA]->(concepto)
    MATCH (concepto)<-[:TRATA]-(t2:Tema)-[:PERTENECE_A]->(up2:UP)
    WHERE up1 <> up2
    RETURN t1.nombre AS tema_origen, concepto.nombre AS concepto_compartido,
           t2.nombre AS tema_relacionado, up2.id AS up_relacionada, up2.nombre AS up_nombre
    """, {"up_id": up_id})


def get_topic_detail(tema_nombre: str) -> dict:
    results = query("""
    MATCH (t:Tema {nombre: $nombre})
    OPTIONAL MATCH (t)-[:TRATA]->(p:Patologia)
    OPTIONAL MATCH (p)-[:SE_DIAGNOSTICA_CON]->(dx:MetodoDx)
    OPTIONAL MATCH (p)-[:SE_TRATA_CON]->(tx)
    OPTIONAL MATCH (p)-[:PRESENTA]->(s:Signo)
    OPTIONAL MATCH (t)-[:CONTENIDO_EN]->(f:Fuente)
    OPTIONAL MATCH (t)-[:INCLUYE]->(proc:Procedimiento)
    RETURN t.nombre AS tema, t.descripcion AS descripcion,
           collect(DISTINCT {nombre: p.nombre, definicion: p.definicion, cie10: p.cie10}) AS patologias,
           collect(DISTINCT dx.nombre) AS metodos_dx,
           collect(DISTINCT s.nombre) AS signos,
           collect(DISTINCT {titulo: f.titulo, id: f.id}) AS fuentes,
           collect(DISTINCT proc.nombre) AS procedimientos
    """, {"nombre": tema_nombre})
    return results[0] if results else None


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

# === ACTIVITIES ===

def get_activity(nombre: str) -> list:
    """Busca actividades (TP, Seminario, Taller, Acreditacion) por nombre, titulo o temas.

    Busca por frase completa y por palabras individuales (>= 4 chars) para
    maximizar matches. Ej: "rinoscopia anterior" matchea titulo "Rinoscopia".
    """
    # Primero intento con la frase completa
    results = query("""
    MATCH (a:Actividad)
    WHERE toLower(a.nombre) CONTAINS toLower($nombre)
       OR toLower(a.titulo) CONTAINS toLower($nombre)
       OR toLower(a.id) CONTAINS toLower($nombre)
    OPTIONAL MATCH (a)-[:ABORDA]->(t:Tema)
    OPTIONAL MATCH (a)-[:PRACTICA]->(proc:Procedimiento)
    OPTIONAL MATCH (a)-[:TIENE_DOCUMENTO]->(d:Documento)
    OPTIONAL MATCH (a)-[:PERTENECE_A]->(up:UP)
    RETURN a.id AS id, a.nombre AS nombre, a.titulo AS titulo, a.tipo AS tipo,
           up.nombre AS unidad_problematica, up.id AS up_id,
           collect(DISTINCT t.nombre) AS temas,
           collect(DISTINCT proc.nombre) AS procedimientos,
           collect(DISTINCT {id: d.id, nombre: d.nombre, tipo: d.tipo, archivo: d.archivo}) AS documentos
    """, {"nombre": nombre})

    if results:
        return results

    # Fallback: buscar por cada palabra individual (>= 4 chars)
    words = [w for w in nombre.lower().split() if len(w) >= 4]
    if not words:
        return []

    # Buscar actividades cuyo titulo contenga alguna de las palabras
    for word in words:
        results = query("""
        MATCH (a:Actividad)
        WHERE toLower(a.titulo) CONTAINS toLower($word)
           OR toLower(a.nombre) CONTAINS toLower($word)
        OPTIONAL MATCH (a)-[:ABORDA]->(t:Tema)
        OPTIONAL MATCH (a)-[:PRACTICA]->(proc:Procedimiento)
        OPTIONAL MATCH (a)-[:TIENE_DOCUMENTO]->(d:Documento)
        OPTIONAL MATCH (a)-[:PERTENECE_A]->(up:UP)
        RETURN a.id AS id, a.nombre AS nombre, a.titulo AS titulo, a.tipo AS tipo,
               up.nombre AS unidad_problematica, up.id AS up_id,
               collect(DISTINCT t.nombre) AS temas,
               collect(DISTINCT proc.nombre) AS procedimientos,
               collect(DISTINCT {id: d.id, nombre: d.nombre, tipo: d.tipo, archivo: d.archivo}) AS documentos
        """, {"word": word})
        if results:
            return results

    # Último fallback: buscar en temas que aborda la actividad
    for word in words:
        results = query("""
        MATCH (a:Actividad)-[:ABORDA]->(t:Tema)
        WHERE toLower(t.nombre) CONTAINS toLower($word)
        OPTIONAL MATCH (a)-[:PRACTICA]->(proc:Procedimiento)
        OPTIONAL MATCH (a)-[:TIENE_DOCUMENTO]->(d:Documento)
        OPTIONAL MATCH (a)-[:PERTENECE_A]->(up:UP)
        RETURN a.id AS id, a.nombre AS nombre, a.titulo AS titulo, a.tipo AS tipo,
               up.nombre AS unidad_problematica, up.id AS up_id,
               collect(DISTINCT t.nombre) AS temas,
               collect(DISTINCT proc.nombre) AS procedimientos,
               collect(DISTINCT {id: d.id, nombre: d.nombre, tipo: d.tipo, archivo: d.archivo}) AS documentos
        """, {"word": word})
        if results:
            return results

    return []


def list_activities(up_id: str = None, tipo: str = None) -> list:
    """Lista actividades filtradas por UP y/o tipo."""
    where_clauses = []
    params = {}

    if up_id:
        where_clauses.append("up.id = $up_id")
        params["up_id"] = up_id
    if tipo:
        where_clauses.append("a.tipo = $tipo")
        params["tipo"] = tipo

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    return query(f"""
    MATCH (a:Actividad)-[:PERTENECE_A]->(up:UP)
    {where}
    OPTIONAL MATCH (a)-[:ABORDA]->(t:Tema)
    RETURN a.id AS id, a.nombre AS nombre, a.titulo AS titulo, a.tipo AS tipo,
           up.id AS up_id, collect(DISTINCT t.nombre) AS temas
    ORDER BY a.nombre
    """, params)


def get_activity_material(activity_id: str) -> list:
    """Devuelve el contenido completo de los documentos vinculados a una actividad."""
    return query("""
    MATCH (a:Actividad {id: $id})-[:TIENE_DOCUMENTO]->(d:Documento)
    RETURN d.id AS doc_id, d.nombre AS nombre, d.tipo AS tipo,
           d.archivo AS archivo, d.texto AS texto, d.palabras AS palabras
    ORDER BY d.tipo, d.nombre
    """, {"id": activity_id})


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
