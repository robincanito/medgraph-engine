"""Endpoints administrativos: ingesta de libros."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from services.ingest import run_ingest, get_job

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest")
async def ingest_libro(
    background_tasks: BackgroundTasks,
    libro_id: str = Form(...),
    titulo: str = Form(...),
    autor: str = Form(""),
    pdf: UploadFile = File(...),
):
    """Sube un PDF y arranca el pipeline de ingesta en background."""
    existing = get_job(libro_id)
    if existing and existing["status"] == "running":
        raise HTTPException(409, f"Ya hay una ingesta en curso para {libro_id}")

    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF")

    # Save to temp
    pdf_path = f"/tmp/{libro_id}.pdf"
    content = await pdf.read()
    with open(pdf_path, "wb") as f:
        f.write(content)

    # Run in background
    background_tasks.add_task(run_ingest, libro_id, pdf_path, titulo, autor)

    return {"job_id": libro_id, "status": "started"}


@router.get("/ingest/{job_id}")
async def ingest_status(job_id: str):
    """Consulta el estado de una ingesta en curso."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    return job
