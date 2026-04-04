"""Conexión a Neo4j y operaciones base."""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE")


def get_driver():
    return GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def run_query(query: str, params: dict = None):
    """Ejecuta una query Cypher y retorna los resultados."""
    with get_driver() as driver:
        with driver.session(database=DATABASE) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]


def run_write(query: str, params: dict = None):
    """Ejecuta una query de escritura."""
    with get_driver() as driver:
        with driver.session(database=DATABASE) as session:
            session.execute_write(lambda tx: tx.run(query, params or {}))


def test_connection():
    """Verifica conexión a Neo4j."""
    try:
        with get_driver() as driver:
            driver.verify_connectivity()
            print("Conexión exitosa a Neo4j AuraDB")
            return True
    except Exception as e:
        print(f"Error de conexión: {e}")
        return False


if __name__ == "__main__":
    test_connection()
