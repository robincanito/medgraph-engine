"""Endpoints administrativos: ingesta de libros."""

import re

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from services.ingest import run_ingest, get_job

router = APIRouter(prefix="/admin", tags=["admin"])

# UN `libro_id` VALIDO, Y ES EL PATRON DEL CONTRATO (13-sep-2026). El schema `chunk/v1` de
# nomos-contracts declara `libro_id: "^[a-z0-9-]+$"`: lo que entra por esta puerta tiene que ser
# lo mismo que lo que sale al grafo. Se agrega solo un techo de largo, porque el id se convierte
# en un nombre de archivo.
#
# POR QUE IMPORTA: `libro_id` viene de un formulario y mas abajo se concatena a una ruta de disco.
# Sin validar, un id con `..` escribia el PDF subido FUERA del directorio temporal. El patron no
# admite punto ni barra, asi que no hay traversal posible.
LIBRO_ID = re.compile(r"^[a-z0-9-]{1,64}$")


@router.post("/ingest")
async def ingest_libro(
    background_tasks: BackgroundTasks,
    libro_id: str = Form(...),
    titulo: str = Form(...),
    autor: str = Form(""),
    pdf: UploadFile = File(...),
):
    """Sube un PDF y arranca el pipeline de ingesta en background."""
    # PRIMERO EL GUARD, antes de consultar el job y antes de tocar el disco.
    if not LIBRO_ID.match(libro_id):
        raise HTTPException(
            422, "libro_id invalido: se espera el patron del contrato ^[a-z0-9-]+$ "
                 "(minusculas, digitos y guiones, hasta 64 caracteres)")

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
