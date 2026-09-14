# Image of the MedGraph Engine API. BUILD CONTEXT = THE REPO ROOT, not api/.
#
# Why it lives here and not in api/ (14-sep-2026): the API imports `pipeline/` — the retrieval
# filters by the content types the parser declares, and the admin/v1 descriptor is built from the
# active profile in `profiles/`. With the context inside api/ neither of those directories made it
# into the image, and the container died on `from pipeline import perfiles`.
#
#   docker build -t medgraph-engine-api .
#   docker run --rm -p 8080:8080 --env-file .env medgraph-engine-api
FROM python:3.13-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The API at /app (so `main:app` resolves) and the pipeline next to it, importable as
# `pipeline.parseo` without touching sys.path.
COPY api/ /app/
COPY pipeline/ /app/pipeline/
COPY profiles/ /app/profiles/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
