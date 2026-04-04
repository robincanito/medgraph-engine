# MedGraph

**A knowledge graph engine that transforms textbooks and documents into an intelligent, queryable system with semantic search, reasoning flows, and ontological classification.**

Originally built for medical education, but the pipeline is domain-agnostic — it works with any field where knowledge is structured in books: law, engineering, biology, chemistry, or any academic discipline. Bring your own books, build your own knowledge graph.

---

## The Problem

Students and professionals deal with thousands of pages across dozens of textbooks. Information is fragmented across multiple sources with no way to ask cross-cutting questions and get a unified answer backed by exact page citations from multiple books.

## The Solution

MedGraph parses PDFs (textbooks, papers, course materials), chunks them semantically, generates vector embeddings, extracts entities and relationships using LLMs, maps them to standard ontologies, and exposes everything through an intelligent API that a language model can query.

It's not a search engine. It's a **knowledge graph with structured reasoning**.

> While the examples and built-in ontologies (ATC/SNOMED) are medical, the core pipeline — parsing, chunking, embedding, entity extraction, and graph construction — works with any domain.

---

## Requirements

| Requirement | What it is | Cost | Where to get it |
|---|---|---|---|
| **Python 3.10+** | Runtime | Free | [python.org](https://python.org) |
| **Docker** (optional) | For running Neo4j locally | Free | [docker.com](https://docker.com) |
| **Neo4j** | Graph database | Free | Docker (local) or [AuraDB Free](https://neo4j.com/cloud/aura-free/) |
| **Google AI Studio API key** | For embeddings and entity extraction | Free | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **Your own PDFs** | The content you want to process | — | Your textbooks, papers, course materials |

No credit card required. Everything runs locally.

### Neo4j: Local vs Cloud

| | Docker (local) | AuraDB Free (cloud) |
|---|---|---|
| Cost | Free | Free |
| Limits | None | 200K nodes, 400K relationships |
| Access | Your PC only | From anywhere |
| Setup | `docker-compose up -d` | Create account at neo4j.io |
| URI | `bolt://localhost:7687` | `neo4j+s://xxx.databases.neo4j.io` |
| Best for | Getting started, development | Sharing, production |

### Google AI Studio API Key

The pipeline uses Google's Gemini models for embeddings, entity extraction, and query routing. All you need is a free API key:

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key to your `.env` file

No GCP project, no credit card, no billing account required. The free tier is enough to process several books.

---

## Quick Start (5 minutes)

### Step 1 — Clone and configure

```bash
git clone https://github.com/robincanito/medgraph-engine.git
cd medgraph-engine
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your credentials:

```
# Get your free API key at https://aistudio.google.com/apikey
GCP_API_KEY=your-google-ai-studio-key

# If using Docker (local Neo4j):
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme-local-password

# If using AuraDB Free (cloud Neo4j):
# NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=your-auradb-password
```

### Step 2 — Start Neo4j

**Option A — Docker (recommended):**
```bash
docker-compose up -d
# Wait ~30 seconds for Neo4j to be healthy
# Web UI available at http://localhost:7474
```

**Option B — AuraDB Free:**
1. Create a free instance at [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free/)
2. Copy the connection URI and password to your `.env`

### Step 3 — Run the pipeline

```bash
# Place a PDF in the examples/ folder
python quickstart.py
```

`quickstart.py` runs the full pipeline automatically:

1. Checks your environment and dependencies
2. Creates Neo4j schema and indexes
3. Parses your PDF (text extraction, structure detection)
4. Generates 280-word chunks with overlap
5. Creates vector embeddings (Gemini 3072d)
6. Extracts entities with LLM (requires Vertex AI — skipped if not configured)
7. Verifies everything works

### Step 4 — Start the API

```bash
cd api && uvicorn main:app --reload
# Open http://localhost:8000/docs
```

### Step 5 — Connect your LLM

See the [MedGraph Client](https://github.com/robincanito/medgraph-client-oss) for connecting Claude Code, ChatGPT, Ollama, or OpenClaw to your instance.

---

## Deploy to Cloud (optional)

If you want your API accessible from anywhere (other devices, bots, OpenClaw via WhatsApp), you can deploy to Google Cloud Run:

### Prerequisites

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed
- A GCP project (free tier works)
- Neo4j on AuraDB Free (Docker local won't work with Cloud Run — it needs a remote database)

### Steps

```bash
# 1. Login to Google Cloud
gcloud auth login

# 2. Deploy the API
cd api
gcloud run deploy medgraph-api \
  --source . \
  --project your-gcp-project \
  --region us-central1 \
  --set-env-vars="NEO4J_URI=neo4j+s://your-auradb.databases.neo4j.io,NEO4J_USERNAME=neo4j,NEO4J_PASSWORD=your-password,NEO4J_DATABASE=neo4j,API_KEY=your-api-key,GCP_API_KEY=your-ai-studio-key,ENVIRONMENT=production"

# 3. Done — Cloud Run gives you a public URL
# Example: https://medgraph-api-xxxxx.us-central1.run.app
```

Your API is now live. Point the [MedGraph Client](https://github.com/robincanito/medgraph-client-oss) to that URL and query from anywhere.

> **Tip:** For production, use [Google Secret Manager](https://cloud.google.com/secret-manager) instead of passing credentials as env vars. Replace `--set-env-vars` with `--set-secrets` for each sensitive value.

---

## Architecture

```
Layer 1: DOCUMENTAL    Chunks with vector embeddings (semantic search)
Layer 2: SEMANTIC      Full-text BM25 + RRF fusion + query rewriting
Layer 3: GRAPH         Typed entities + relationships (CAUSED_BY, TREATED_WITH, etc.)
Layer 4: REASONING     DAGs: directed clinical flows (symptom -> diagnosis -> treatment)
Layer 5: ONTOLOGY      ATC drug classification + SNOMED clinical terminology
```

Each layer enriches the previous one. An intelligent router (Gemini) analyzes each query and activates the relevant layers automatically.

### System Diagram

```
User Question
    |
    v
GEMINI ROUTER (analyzer)
    |
    +---> ONTOLOGY -----> ATC hierarchy / SNOMED classification
    |
    +---> GRAPH --------> Entity relationships (CAUSED_BY, TREATED_WITH, etc.)
    |
    +---> BIBLIOGRAPHY -> Hybrid search: vector (3072d) + BM25 + RRF fusion
    |
    +---> ACTIVITIES ---> Academic activities (labs, seminars, practical work)
    |
    +---> DAGS ---------> Clinical reasoning flows (pathways + clinical decisions)
    |
    v
UNIFIED RESPONSE (structured JSON with sources)
```

---

## What You Can Build

The engine scales with the content you feed it. As a reference, a deployment with 17 textbooks produces:

| Metric | Example at scale |
|---|---|
| Semantic entities | 100K+ (pathologies, drugs, anatomy, signs, procedures) |
| Typed relationships | 1M+ (CAUSED_BY, TREATED_WITH, DIAGNOSED_WITH, etc.) |
| Text chunks | ~2,000 per book (280 words each, with overlap) |
| Ontology mappings | ATC drug hierarchy + SNOMED clinical terminology |
| Clinical DAGs | Custom reasoning flows per topic |
| Embedding dimensions | 3,072 (Gemini Embedding 2 Preview) |

Your instance starts empty. Each book you process through the pipeline adds thousands of entities and relationships automatically.

---

## Tech Stack

| Component | Technology |
|---|---|
| Knowledge Graph | Neo4j AuraDB |
| Embeddings | Google Gemini Embedding 2 Preview (3072 dims) |
| Entity Extraction | Gemini 3.1 Flash Lite |
| Query Router | Gemini 2.5 Flash |
| API | FastAPI on Google Cloud Run |
| Vector Search | Neo4j native vector index |
| Full-text Search | Neo4j native full-text index (BM25) |
| PDF Parsing | PyMuPDF + Google Cloud Vision (OCR) |
| Auth | Bearer token + Google Secret Manager |
| LLM Integration | MCP Server (Claude) + Custom GPT (ChatGPT) |

---

## Pipeline

The complete ingestion pipeline transforms a PDF into queryable knowledge:

```
PDF (textbook or course material)
  |
  v
1. PARSE (parser_v2.py)
   PyMuPDF extracts text, detects chapters/sections
   Cloud Vision API for scanned books (OCR)
  |
  v
2. CHUNK (parser_v2.py)
   280 words target, 60 words overlap
   Parent-child structure (parent = full section, children = chunks)
   Rich metadata: book, chapter, section, page range, word count
  |
  v
3. UPLOAD (upload_chunks.py)
   Batch upload to Neo4j as :Chunk and :ParentChunk nodes
  |
  v
4. VECTORIZE (vectorize.py)
   Generate embeddings with Gemini Embedding 2 Preview (3072 dims)
   Store as node property in Neo4j
  |
  v
5. EXTRACT ENTITIES (extract_entities.py)
   LLM reads each chunk and extracts:
   - Entities: pathologies, drugs, anatomy, signs, symptoms, procedures...
   - Relationships: CAUSED_BY, TREATED_WITH, DIAGNOSED_WITH...
   Canonicalize (deduplicate, merge synonyms, sum frequencies)
   Upload as typed nodes + MENTIONS relationships
  |
  v
6. MAP ONTOLOGY (ontology.py)
   Map drugs to ATC hierarchy (Anatomical Therapeutic Chemical)
   Map pathologies/anatomy/procedures to SNOMED-CT
   Create IS_A hierarchical relationships
```

---

## Entity Types

| Label | Count | Examples |
|---|---|---|
| Patologia | 36,021 | Otitis media, hypertension, lymphoma |
| EstructuraAnatomica | 12,850 | Tympanic membrane, cochlea, retina |
| Procedimiento | 10,449 | Otoscopy, ECG, chest X-ray |
| Hallazgo | 8,238 | Bulging TM, ST elevation |
| Farmaco | 7,070 | Amoxicillin, ibuprofen, oseltamivir |
| Signo | 5,945 | Fever, edema, cyanosis |
| Parametro | 5,298 | Blood pressure, heart rate, hemoglobin |
| MetodoDx | 5,222 | CBC, echocardiogram, audiometry |
| Agente | 4,135 | S. pneumoniae, EBV, Influenza A |
| GrupoFarmacologico | 3,993 | NSAIDs, beta-lactams, coxibs |
| Sintoma | 3,017 | Otalgia, headache, dyspnea |

## Relationship Types

`CAUSED_BY` `TREATED_WITH` `DIAGNOSED_WITH` `MANIFESTS_WITH` `PART_OF` `EVALUATES` `BELONGS_TO` `CAN_PRODUCE` `DIFFERENTIAL_OF` `ASSOCIATED_WITH` `RISK_FACTOR` `COMPLICATION_OF` `VARIANT_OF` `IS_A` `MENTIONS` `PATHWAY` `CLINICAL`

---

## API Endpoints

### Main (use this)

```bash
POST /query
# Intelligent unified query — analyzes the question and activates relevant layers
{"pregunta": "What NSAIDs are contraindicated in chronic kidney disease?", "top_k": 10}
```

### Specific

```bash
POST   /search/hybrid          # Hybrid search (semantic + BM25 + RRF)
GET    /topic/{topic}/comprehensive  # Full topic: graph + activities + bibliography
GET    /pathology/{name}       # Pathology details from graph
GET    /procedure/{name}       # Procedure details from graph
GET    /activity/search/{name} # Academic activity search
GET    /topic/{topic}/ontology # ATC/SNOMED hierarchical classification
GET    /topic/{topic}/pathways # Clinical reasoning DAGs
GET    /topic/{topic}/clinical # Clinical decision flows
```

---

## DAGs: Clinical Reasoning

DAGs (Directed Acyclic Graphs) add **sequence and clinical logic** on top of the knowledge graph. The graph knows that "OMA is connected to otalgia, otoscopy, amoxicillin, S. pneumoniae". The DAG adds the **order**: symptom -> exam -> finding -> diagnosis -> treatment -> follow-up.

Example — Otalgia management:

```
Otalgia -> Anamnesis (age, duration, fever, ENT history)
  -> Otoscopy
    -> IF swollen CAE, positive tragus sign -> Otitis externa
    -> IF bulging, erythematous, opaque TM -> Acute otitis media
    -> IF retracted TM, air-fluid level -> Effusive otitis media
    -> IF normal TM and CAE -> Referred otalgia
  -> Treatment per diagnosis
  -> Follow-up 48-72h
  -> Escalation if no improvement
```

---

## MCP Server (Claude Integration)

MedGraph includes an MCP server that connects directly to Claude (Anthropic's AI), giving it real-time access to the medical knowledge graph.

```bash
# Install and run
pip install fastmcp httpx
python mcp_server.py
```

Available tools: `medgraph_query`, `medgraph_search`, `medgraph_comprehensive`, `medgraph_pathology`, `medgraph_procedure`, `medgraph_activity`, `medgraph_pathways`, `medgraph_clinical`, `medgraph_cronograma`

---

## Getting Started

### Prerequisites

- Python 3.10+
- Neo4j AuraDB instance (or local Neo4j)
- Google Cloud Platform account (for Gemini embeddings + Vertex AI)

### Installation

```bash
git clone https://github.com/robincanito/medgraph.git
cd medgraph
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### Run the pipeline on your own books

```bash
# 1. Parse a PDF
python parser_v2.py my-textbook

# 2. Upload chunks to Neo4j
python upload_chunks.py my-textbook

# 3. Generate embeddings
python vectorize.py my-textbook

# 4. Extract entities with LLM
python extract_entities.py my-textbook

# 5. Map to ATC/SNOMED ontology
python ontology.py
```

### Run the API

```bash
cd api
uvicorn main:app --reload
```

### Deploy to Cloud Run

```bash
cd api
gcloud run deploy medgraph-api --source . --project your-project --region us-central1
```

---

## Important Note

**This repository contains the engine only.** Book data, extracted entities, parsed chunks, and the Neo4j database are not included due to copyright restrictions on medical textbooks. You must provide your own PDFs and run the pipeline to populate your own knowledge graph.

### Disclaimer

MedGraph is a software tool for processing and structuring text content. Users are solely responsible for ensuring they have the appropriate rights to any content they process through the system. The authors of MedGraph do not endorse, facilitate, or assume liability for copyright infringement or any unauthorized use of copyrighted materials.

---

## Security Considerations

MedGraph is designed for local development and personal use out of the box. **If you plan to deploy it to production or expose the API publicly, review the following:**

- **API authentication:** The API uses a single Bearer token (`API_KEY` env var). For production, implement proper auth (OAuth2, JWT, or API key rotation).
- **CORS:** The default config allows `localhost` origins only. Add your frontend domains explicitly — never use `allow_origins=["*"]` in production.
- **Rate limiting:** Included via `slowapi` (100 req/min default). Adjust for your expected load.
- **Neo4j credentials:** Use environment variables or a secret manager (e.g., Google Secret Manager, AWS Secrets Manager). Never hardcode credentials.
- **HTTPS:** Cloud Run and most hosting providers handle TLS automatically. If self-hosting, put the API behind a reverse proxy (nginx/Caddy) with SSL.
- **Input validation:** The API accepts user queries as strings. While they go to an LLM (not SQL), consider sanitizing inputs if exposing publicly.
- **Docker Compose:** The default Neo4j password is `changeme-local-password`. Change it immediately if exposing the database.
- **Dependency updates:** Run `pip audit` periodically to check for known vulnerabilities in dependencies.

This is not an exhaustive security review. For production deployments, conduct a proper security assessment based on your specific infrastructure and threat model.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
medgraph/
  api/                    # FastAPI application
    routers/              # API route handlers
    services/             # Business logic (graph, vector, analyzer)
    main.py               # App entry point
  dags/                   # Clinical reasoning flows (YAML)
  parser_v2.py            # PDF parser with structure detection
  upload_chunks.py        # Neo4j chunk uploader
  vectorize.py            # Embedding generator
  extract_entities.py     # LLM entity extractor
  ontology.py             # ATC/SNOMED mapper
  load_dags.py            # DAG loader
  db.py                   # Neo4j connection helper
  schema.py               # Database schema and indexes
  mcp_server.py           # MCP server for Claude
  dedup_entities.py       # Entity deduplication
  catalog.json            # Book metadata catalog
  requirements.txt        # Python dependencies
```

---

## Origin

Born from the frustration of studying across dozens of fragmented textbooks and the conviction that medical knowledge should be structured, connected, and queryable — not trapped in isolated PDFs.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPLv3).

You are free to use, modify, and distribute this software. If you deploy it as a network service, you must make your modified source code available to users of that service.

See [LICENSE](LICENSE) for the full text.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
