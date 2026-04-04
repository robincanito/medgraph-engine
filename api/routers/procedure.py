"""Endpoints de procedimientos."""

from fastapi import APIRouter
from services import graph

router = APIRouter(prefix="/procedure", tags=["procedure"])


@router.get("/{nombre}")
async def get_procedure(nombre: str):
    """Detalle de un procedimiento: pasos, insumos, parámetros, hallazgos."""
    results = graph.get_procedure(nombre)
    if not results:
        return {"error": f"Procedimiento '{nombre}' no encontrado"}
    return {"results": results, "count": len(results)}
