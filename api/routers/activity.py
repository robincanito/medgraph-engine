"""Endpoints de actividades: TPs, Seminarios, Talleres, Acreditaciones."""

from fastapi import APIRouter, Query
from services import graph

router = APIRouter(prefix="/activity", tags=["activity"])


@router.get("/search/{nombre}")
async def search_activity(nombre: str):
    """Buscar actividad por nombre, titulo o ID. Ej: 'TP N7', 'Otoscopia', 'Seminario N4'."""
    results = graph.get_activity(nombre)
    if not results:
        return {"error": f"Actividad '{nombre}' no encontrada", "sugerencia": "Probá con 'TP', 'Seminario', 'Taller' + numero o tema"}
    return {"results": results, "count": len(results)}


@router.get("/list")
async def list_activities(
    up_id: str | None = Query(None, description="Filtrar por UP (ej: COURSE-UNIT2)"),
    tipo: str | None = Query(None, description="Filtrar por tipo: tp, seminario, taller, acreditacion, tutoria"),
):
    """Listar todas las actividades, opcionalmente filtradas por UP y/o tipo."""
    results = graph.list_activities(up_id=up_id, tipo=tipo)
    return {"results": results, "count": len(results)}


@router.get("/{activity_id}/material")
async def get_material(activity_id: str):
    """Obtener el contenido completo de los documentos vinculados a una actividad."""
    results = graph.get_activity_material(activity_id)
    if not results:
        return {"error": f"No hay material vinculado a '{activity_id}'"}
    return {"activity_id": activity_id, "documentos": results, "count": len(results)}


@router.get("/doc/{doc_nombre}")
async def get_document(doc_nombre: str):
    """Buscar un documento por nombre. Ej: 'Otoscopia', 'Bioseguridad', 'ECG'."""
    results = graph.query("""
    MATCH (d:Documento)
    WHERE toLower(d.nombre) CONTAINS toLower($nombre)
    RETURN d.id AS id, d.nombre AS nombre, d.tipo AS tipo,
           d.archivo AS archivo, d.palabras AS palabras, d.texto AS texto
    ORDER BY d.nombre
    """, {"nombre": doc_nombre})
    if not results:
        return {"error": f"Documento '{doc_nombre}' no encontrado"}
    return {"results": results, "count": len(results)}
