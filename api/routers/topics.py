"""Endpoints de temas y contenido."""

from fastapi import APIRouter
from services import graph

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("/{up_id}")
async def get_topics(up_id: str):
    """Obtener temas de una UP con fuentes bibliográficas."""
    topics = graph.get_topics_by_up(up_id)
    return {"up_id": up_id, "topics": topics, "count": len(topics)}


@router.get("/{up_id}/related")
async def get_related(up_id: str):
    """Temas relacionados con otras UPs/materias."""
    related = graph.get_related_topics(up_id)
    return {"up_id": up_id, "related": related, "count": len(related)}


@router.get("/{tema}/detail")
async def get_detail(tema: str):
    """Detalle completo de un tema: patologías, dx, tto, fuentes."""
    detail = graph.get_topic_detail(tema)
    if not detail:
        return {"error": f"Tema '{tema}' no encontrado"}
    return detail
