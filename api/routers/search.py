"""Endpoints de búsqueda: full-text, semántica e híbrida con RRF."""

from fastapi import APIRouter
from pydantic import BaseModel
from services import vector

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    libro_id: str | None = None


@router.post("/semantic")
async def semantic_search(req: SearchRequest):
    """Búsqueda por significado usando embeddings (vector KNN)."""
    results = vector.search_semantic(req.query, req.top_k, req.libro_id)
    return {"query": req.query, "results": results, "count": len(results)}


@router.post("/keyword")
async def keyword_search(req: SearchRequest):
    """Búsqueda full-text con scoring BM25 (Lucene via Neo4j)."""
    results = vector.search_keyword(req.query, req.top_k, req.libro_id)
    return {"query": req.query, "results": results, "count": len(results)}


@router.post("/hybrid")
async def hybrid_search(req: SearchRequest):
    """Búsqueda híbrida: full-text + semántica fusionadas con Reciprocal Rank Fusion. Usar esta por defecto."""
    result = vector.search_hybrid(req.query, req.top_k, req.libro_id)
    return {
        "query": req.query,
        "keyword_count": result["keyword_count"],
        "semantic_count": result["semantic_count"],
        "intencion": result.get("intencion", "general"),
        "query_expandida": result.get("query_expandida", req.query),
        "results": result["results"],
        "count": len(result["results"])
    }
