"""Endpoints de patologías."""

from fastapi import APIRouter
from services import graph

router = APIRouter(prefix="/pathology", tags=["pathology"])


@router.get("/{nombre}")
async def get_pathology(nombre: str):
    """Todo sobre una patología: definición, dx, tto, signos, fuentes."""
    results = graph.get_pathology(nombre)
    if not results:
        return {"error": f"Patología '{nombre}' no encontrada"}
    return {"results": results, "count": len(results)}


@router.get("/{nombre}/differential")
async def get_differential(nombre: str):
    """Diagnóstico diferencial basado en signos compartidos."""
    results = graph.get_differential(nombre)
    return {"patologia": nombre, "diferenciales": results, "count": len(results)}
