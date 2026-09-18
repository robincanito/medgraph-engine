# MedGraph Engine

**An ingestion pipeline that turns documents into a queryable knowledge graph: PDF → pages →
structure detected by a domain profile → overlapping chunks with provenance → an idempotent load
into Neo4j → embeddings generated under a single, explicit policy.**

Built for medical bibliography, but the pipeline is domain-agnostic: what to treat as a chapter,
what to treat as a section and how big a chunk should be are declared in a **profile**
(`profiles/<domain>.yaml`), not hardcoded. Bring your own documents.

It is not a search engine and not a wrapper around one. It is the part of a RAG/GraphRAG system
that nobody wants to rewrite: the parsing, the chunking, the provenance, the idempotent load and
the embedding policy — with the failure modes already paid for and written down.

> **This repo is a mirror of the pipeline.** `pipeline/` and `profiles/` are byte-for-byte copies
> of the private MedGraph repo (part of the Nomos Graph ecosystem). See
> [The pipeline is a mirror](#the-pipeline-is-a-mirror) before opening a PR against them.

---

## What it does

```
document (PDF)
  │
  ├─ 1. EXTRACT      pipeline/parseo.py — text per page, tables rebuilt in reading order
  │                  (PyMuPDF find_tables; a table becomes "value (COLUMN); value (COLUMN).")
  │
  ├─ 2. STRUCTURE    pipeline/parseo.py + pipeline/estrategia.py — chapter/section headings
  │                  detected with the patterns THE PROFILE declares. A page with two headings
  │                  is split into segments: the title belongs to the POSITION, not to the page.
  │
  ├─ 3. CHUNK        pipeline/parseo.py — target 280 words, 60 of overlap, cut on a sentence
  │                  boundary; parent chunks group 3 children. Every chunk carries libro_id,
  │                  page_start/page_end, chapter, section, content type and word count.
  │                  Reference lists and code indexes are typed as such so retrieval can skip
  │                  them (`parseo.NO_CONTENIDO`).
  │
  ├─ 4. LOAD         pipeline/carga.py — MERGE by id, never CREATE. Re-ingesting the same
  │                  document is a no-op; chunks that disappeared from the document are deleted
  │                  as orphans; a chunk whose text changed loses its embedding (so it gets
  │                  re-embedded and nothing else does); derived relationships (CHILD_OF,
  │                  SIGUE_A) are recomputed. Fails closed: an empty parse never empties a
  │                  loaded document.
  │
  └─ 5. EMBED        pipeline/embeddings.py — one embedding policy for every caller: canonical
                     prefix ("Capítulo: … Sección: … Tipo: … " + text), one content per call,
                     2000-char ceiling, 3 retries per batch with growing backoff, and a SECOND
                     PASS over whatever is still missing. What it reports is auditable:
                     embedded, missing, failed batches and **billable calls**.
```

The steps report through **named events** (`pipeline/eventos.py`) on top of the standard library
`logging`: fields instead of prose, so a run can be measured from the outside — which document,
which step, how long, how many paid calls. The vocabulary (`ingest_paso`, `embed_lote`,
`llm_llamada`) is declared and tested here, and `pipeline/embeddings.py` emits `embed_lote` on
every attempt (ok / retry / failure). `ingest_paso` is emitted by an orchestrator wrapping each
step — this repo does not ship one yet (see Known gaps); `eventos.paso` is the context manager to
use when you write yours.

Entity extraction (`extract_entities.py`), deduplication (`dedup_entities.py`), ontology mapping
(`ontology.py`) and clinical DAGs (`load_dags.py`, `dags/*.yaml`) are separate root scripts that
run after the load. They are this repo's own code, not part of the mirrored pipeline.

---

## Layout

```
pipeline/            ← the canonical implementation of each step (MIRROR — see below)
  parseo.py            extract, detect structure, chunk, normalize
  estrategia.py        Estrategia: chunk sizes + structure patterns of a domain
  perfiles.py          finds and loads profiles/<domain>.yaml (the only place that knows the path)
  carga.py             idempotent load, source (:Book) registration, deletion
  embeddings.py        canonical embedding text, provider call, retry/second-pass policy
  eventos.py           named events (logging only — no dependency on the private logger)
profiles/            ← domain profiles: medicina.yaml (default), generico.yaml
api/                 ← the FastAPI service: query routes + admin/v1 (read-only). It starts.
dags/                ← clinical reasoning flows (YAML)
tests/               ← the suite: pipeline regression + the API + repo health
  contracts/           vendored JSON Schemas of nomos-contracts (a test keeps them identical)
Dockerfile           ← the API image. Build context = the repo root (it needs pipeline/ too)
docker-compose.yml   ← local Neo4j + the API for development
```

### Root scripts: what is alive and what is legacy

Checked by grepping for who imports `pipeline/` (`tests/test_pipeline_parseo.py` keeps two of
these honest with an AST test):

| Script | Status | Notes |
|---|---|---|
| `parser_v2.py` | **alive — shim** | Re-exports `pipeline.parseo` (the same objects, not a copy) and keeps the `catalog.json`-driven CLI. Import it and you get the canonical parser. |
| `db.py`, `schema.py` | **alive** | Neo4j connection helper and schema/index creation. |
| `extract_entities.py`, `dedup_entities.py`, `ontology.py`, `load_dags.py` | **alive** | Post-load steps; no counterpart in `pipeline/`. LLM calls go through `google-genai`. |
| `mcp_server.py` | **gone (2026-09-17)** | It was a *stdio* MCP server talking HTTP to a separate API, with eleven instance-prefixed tools — three of them against routes this mirror no longer exposes. The MCP endpoint now lives **inside** the API: `POST /mcp`, see [The MCP endpoint](#the-mcp-endpoint). |
| `quickstart.py` | **alive, but** | Uses the legacy upload path and requires Neo4j before it parses anything (see Quick start). |
| `migrate_chunks.py` | **hybrid** | Re-exports `normalize_chunks`/`normalize_for_search` from `pipeline.parseo`, but its own uploader still does `CREATE` — i.e. it is **not** the idempotent load of `pipeline/carga.py`. |
| `vectorize.py` | **legacy** | Its own embedding loop, its own `EMBEDDING_MODEL`, its own batch size. Duplicates `pipeline/embeddings.py` and does not have the retry/second-pass policy or the paid-call accounting. |
| `upload_chunks.py` | **legacy** | Writes a subset of the chunk fields from v1 `parsed/*_chunks.json` files. |
| `parser.py` | **legacy** | The v1 parser, superseded by `pipeline/parseo.py`. |
| `search.py` | **legacy** | Keyword search over local parsed JSON, from before the graph existed. |

Legacy here means: *still in the tree, not deleted, not maintained, and not what you should call.*
For anything new, call `pipeline/` directly (four lines, see Quick start) — that is the code with
tests. (Heads-up for Windows: the root CLIs report progress with `print` and accented Spanish,
which raises `UnicodeEncodeError` on a cp1252 console — run them with `PYTHONIOENCODING=utf-8`.
`pipeline/` is on `logging`, except four progress lines still printed by `parse_pdf_v2`.)

---

## The pipeline is a mirror

`pipeline/*.py` and `profiles/*.yaml` are **byte-for-byte copies** of the private MedGraph repo,
where they are the single implementation used by both its CLI and its API. The private side has a
parity test that compares each file against the copy in this repo, so a divergence turns its CI
red.

Consequences, stated plainly:

- **Do not send a PR that edits `pipeline/` or `profiles/` here.** It would be overwritten by the
  next replication. Open an issue describing the bug (a failing test in `tests/` is the ideal
  form): the fix lands upstream and is replicated back.
- Their comments and docstrings are in Spanish, and they carry the history of real incidents
  (dates, what broke, what was measured, what was tried and rejected). That prose is the reason
  the files are copied instead of retyped — it is documentation, not noise. The tests in
  `tests/test_pipeline_*.py` are in Spanish for the same reason; everything user-facing
  (this README, `.env.example`, `CONTRIBUTING.md`) is in English.
- Replication is manual and one-directional (private → public), so this mirror can lag. What is
  missing today is listed under [Known gaps](#known-gaps-what-this-mirror-does-not-have-yet).

---

## Requirements

| Requirement | What for | Cost |
|---|---|---|
| **Python 3.13** | CI runs on 3.13; the pipeline itself works on 3.10+ | Free |
| **Neo4j 5** | The graph. Docker locally, or [AuraDB Free](https://neo4j.com/cloud/aura-free/) | Free |
| **Vertex AI** (a GCP project + ADC) | Embeddings: `pipeline/embeddings.crear_cliente` builds a `google-genai` client with `vertexai=True`, and `gemini-embedding-2` (3072 dims) lives in location `global`, not `us-central1` | Free tier covers a few books |
| **An AI Studio key** (`GCP_API_KEY`) | What the *other* LLM callers use: `extract_entities.py`, `ontology.py`, the API's router, and the legacy `vectorize.py` | Free |
| **Your own documents** | The content | — |

Two different auth paths, and it is not an accident: the mirrored embedding code was migrated to
Vertex (`gcloud auth application-default login`) when the same model started rejecting batched
requests through the AI Studio key. The v1.0 root scripts were left on the key.

Parsing and chunking need **neither** a graph nor a key: `pipeline/parseo.py` only needs PyMuPDF.
That is worth knowing before you set anything up — you can see what the engine makes of your
documents first.

```bash
git clone https://github.com/robincanito/medgraph-engine.git
cd medgraph-engine
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # pipeline + suite (ruff, pytest)
pip install -r api/requirements.txt              # to run the API (and its tests)
cp .env.example .env                             # then edit it
```

---

## Quick start

### Parse a document, no graph, no key

```python
from pipeline import parseo, perfiles

estrategia = perfiles.estrategia("medicina")          # or "generico"
children, parents = parseo.parse_pdf_v2("mydoc.pdf", "mydoc", estrategia)
parseo.normalize_chunks(children)

print(len(children), "chunks")
print(children[0]["page_start"], children[0]["titulo_capitulo"], children[0]["text"][:120])
```

That is the whole first half of the pipeline, and it is exactly what the test suite exercises.

### With a graph

```bash
docker compose up -d          # Neo4j on bolt://localhost:7687, UI on http://localhost:7474
python schema.py              # constraints and indexes
```

```python
from db import run_query, run_write
from pipeline import carga, embeddings, parseo, perfiles

children, parents = parseo.parse_pdf_v2("mydoc.pdf", "mydoc", perfiles.estrategia())
parseo.normalize_chunks(children)

carga.cargar_libro(run_write, "mydoc", children, parents)          # idempotent: re-run at will
carga.registrar_fuente(run_query, run_write, "mydoc", {"title": "My document"})

cliente = embeddings.crear_cliente("your-gcp-project", "global")   # Vertex AI + ADC; the model
                                                                   # lives in `global`, not us-central1
stats = embeddings.vectorizar_libro(run_query, run_write, cliente, "mydoc")
print(stats)   # {'embebidos': …, 'sin_vector': 0, 'lotes_fallidos': 0, 'llamadas': …, 'pasadas': 1}
```

`sin_vector` is the number that matters: if it is not 0, part of the document has no embedding and
is invisible to semantic search. The policy already retried and made a second pass; anything left
is a provider outage or an exhausted quota, and it is reported instead of swallowed.

### Your catalog

The root scripts (`parser_v2.py`, `vectorize.py`, `search.py`, `upload_chunks.py`,
`migrate_chunks.py`, `parser.py`) are driven by a **`catalog.json`** in the repo root: the list of
documents you ingested, one entry per source, keyed by the `id` you use everywhere else
(`libro_id`).

That file is **yours and is not in this repo** — it would be a list of someone else's library. What
ships is **`catalog.example.json`**, with two invented entries that show the shape. The scripts
*read* `catalog.json` if it exists and fall back to the example if it does not, so a fresh clone
runs instead of dying with a `FileNotFoundError`; they always *write* to `catalog.json`, so the
example is never overwritten. Copy it and edit:

```bash
cp catalog.example.json catalog.json
```

`libro_id` must match `^[a-z0-9-]+$` (the `chunk/v1` contract, and what the ingest endpoint
enforces).

### `quickstart.py`

`python quickstart.py` runs the older end-to-end script (PDF in `examples/` → parse → upload →
vectorize → extract entities). Two honest caveats:

- it **exits immediately without a `.env`**, and it creates the Neo4j schema (step 3 of 7) before
  it parses anything — so it cannot be used to try the parser offline; use the snippet above;
- its upload and vectorize steps go through the **legacy** scripts (`migrate_chunks.py`,
  `vectorize.py`), not through `pipeline/carga.py` and `pipeline/embeddings.py`. There is no
  orchestrator in this repo yet that wires the mirrored pipeline end to end (see Known gaps).

### The API

```bash
pip install -r api/requirements.txt
cd api && uvicorn main:app --reload      # http://localhost:8000/docs (with ENVIRONMENT=development)
```

It refuses to start without `API_KEY` — an empty key used to authorize every request that carried
no header — and every route except `/health` needs it: `Authorization: Bearer $API_KEY` (or
`X-API-Key`). Missing or wrong is a **401** with `WWW-Authenticate`.

**Query** — what the graph already holds:

| Route | What it answers | Needs |
|---|---|---|
| `POST /search/hybrid` | BM25 + vector, fused with RRF. The default. | Vertex (it embeds the question) |
| `POST /search/semantic`, `POST /search/keyword` | Each half on its own. | Vertex / nothing |
| `POST /query` | An LLM router decides which layers to activate and composes the answer. | `GCP_API_KEY` |
| `GET /topic/{t}/comprehensive` | Entities + bibliography + ontology in one call. | Vertex |
| `GET /topic/{t}/pathways`, `/clinical` | The DAGs loaded by `load_dags.py`. | — |
| `GET /topic/{t}/ontology` | ATC / SNOMED hierarchies (`ontology.py`). | — |
| `GET /pathology/{n}`, `/pathology/{n}/differential`, `/procedure/{n}` | Typed entities and what hangs off them. | — |
| `POST /mcp` | The MCP endpoint (Streamable HTTP): the same corpus as seven `mcp/v1` tools, for Claude, ChatGPT or any MCP client. See [The MCP endpoint](#the-mcp-endpoint). | depends on the tool |
| `GET /health`, `GET /stats` | Liveness (no credential) and node / relationship counts. | — |

**Administration** — `admin/v1`, the contract every Nomos graph implements so that a single console
can administer it without knowing the domain. The shape is pinned by the JSON Schemas vendored in
`tests/contracts/`, and every response in the suite is validated against them:

| Route | What it returns |
|---|---|
| `GET /admin/v1/descriptor` | Vocabulary, source fields, pipeline steps, capabilities. **Built from the active profile**, not from constants: `PROFILE=generico` changes the domain, the entity types and the relation types with no code change. |
| `GET /admin/v1/health` | `{status, graph, checked_at}`. 200 even when the graph is down — a health check that fails together with its dependency diagnoses nothing. |
| `GET /admin/v1/stats` | Counts by label and by relationship type, plus sources / units / units embedded. Cached for 10 minutes. |
| `GET /admin/v1/sources` | A page of sources: `q`, `kind`, `status`, `filter[<field>]`, `sort`, `limit`, `cursor`. |
| `GET /admin/v1/sources/{id}` | One source, or a 404 as `{detail, code}`. |

Read-only, and the descriptor says so: **every capability is published `false`**, each with a note
(`kind: not_built`, the reason, and what it would take to build it). The contract is explicit that
publishing a capability without an endpoint is lying to the console — so uploads, jobs,
classification, review, PATCH, DELETE, re-ingest, facets, the unit viewer and vectorize are all
declared off. Authentication is the single API key (`auth.schemes: ["api_key"]`): there is no Clerk
and no role model here, and the descriptor does not pretend otherwise.

A source is a `:Book` written by `carga.registrar_fuente` — **and also** a `libro_id` that exists
only in chunks, which is what you get if you call `carga.cargar_libro` and stop there. Those are
listed too, with a `status_detail` telling you to register them: an empty console over a full graph
would be worse than a missing field.

#### What this mirror does NOT expose, and why

The v1.0 copy of `api/` also carried the private instance's routes. They were removed on
2026-09-14, in the same batch that made the service start — an API that is smaller but starts
beats one that is bigger and answers with empty lists:

| Gone | Why |
|---|---|
| `GET /topics/{up_id}`, `/{up_id}/related`, `/{tema}/detail` | They read `:UP`, `:Tema` and `:Fuente` — a course's curriculum. No script in this repo writes those nodes. |
| `GET /activity/*` (list, search, material) | They read `:Actividad` and `:Documento` (lab sessions, seminars and their handouts). Same reason; the `ACTIVITIES` layer of `POST /query` went with them. |
| `POST /admin/ingest`, `GET /admin/ingest/{job_id}` | `services/ingest.py` had its **own** parser and a destructive upload (delete, then create), i.e. the exact opposite of `pipeline/carga.py`. Ingestion here runs through `pipeline/` from your own script. |
| `GET /chatgpt-schema` | It served a curated JSON file that is not in this repo. |
| `routers/clinical.py` | A dead duplicate of two routes in `comprehensive.py`; it was never mounted. |

`GET /topic/{t}/comprehensive` survived, minus its `:Tema` and activities sections.

### The MCP endpoint

The API **is** an MCP server: `POST /mcp`, Streamable HTTP, stateless. It implements `mcp/v1`, the
tool contract every Nomos graph serves (`nomos-contracts/mcp-v1.md`; the JSON Schema is vendored in
`tests/contracts/mcp-tools.schema.json` and the suite validates what a client actually receives
against it). Seven read-only tools, no instance prefix:

| Tool | What it does here |
|---|---|
| `search` | The hybrid search of `POST /search/hybrid`, with `filtros`, `garantizar`, `agrupar` and `procedencia`. |
| `fetch` | One passage by id, with its provenance metadata. |
| `list_sources` | The `admin/v1` catalogue: sources, passages, embedding coverage, status. |
| `search_entities` / `expand_concept` | The typed entity graph, one hop, filtered by relation type. |
| `deep_dive` | Dozens of passages grouped by facet, deduplicated — the tool for writing a document. |
| `unified_query` | The full router of `POST /query`: the analyser plus the corpus layer. |

`evidence` — the contract's one optional capability — is **not** published: it requires an external
source (the private instance asks PubMed) and this repo ships no adapter. Seven tools are the whole
contract.

**Authentication is the instance API key**, the same one the rest of the API takes, and `/mcp` is
not exempt from the middleware. There is no OAuth authorization server here, so the 401 carries no
`resource_metadata`: its body explains the credential instead. Consequently the `admin/v1` descriptor
publishes `capabilities.mcp: true` and `auth.mcp` **without** an `oauth` block — which the contract
reads as "this instance authenticates its MCP some other way". Putting an authorization server in
front (and publishing your own protected-resource metadata) is a deployment decision the contract
already contemplates; nothing in the code needs to change for the tools themselves.

```bash
# Claude Code, against a local API. Header support depends on your client version; if it cannot send
# one, put a proxy in front or use a client that can.
claude mcp add --transport http medgraph-engine http://localhost:8000/mcp \
  --header "X-API-Key: $API_KEY"

# Anything else: it is plain Streamable HTTP.
curl -s -X POST http://localhost:8000/mcp -H "X-API-Key: $API_KEY" \
     -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

`MCP_ALLOWED_HOSTS` matters once this runs behind a domain: the SDK validates the `Host` header
against it (DNS-rebinding protection) and answers `421 Invalid Host header` otherwise. Left empty it
derives the hosts from `PUBLIC_BASE_URL` plus localhost.

*(Until 2026-09-17 this section described `mcp_server.py`, a separate **stdio** server that talked
HTTP to a running API and published eleven tools named `medgraph_*` — three of them against routes
this mirror no longer exposes. It was deleted: the prefix forced every agent to be rewritten when
changing graphs, and a second copy of the tool surface is exactly what the `mcp/v1` contract exists
to stop. If your client cannot speak Streamable HTTP, the place for a stdio shim is the client side,
not a second tool surface in this repo.)*

---

## Domain profiles

A profile is what makes the pipeline domain-agnostic. It declares the entity/relation vocabulary
**and** how the material is parsed and chunked. Abridged from `profiles/generico.yaml` (the real
file also carries the extraction prompt rules and canonicalization settings, and is heavily
commented):

```yaml
profile: generico
version: 1
language: es

entities:
  - {id: concepto,      label: Concepto}
  - {id: procedimiento, label: Procedimiento}
  - {id: organizacion,  label: Organizacion}
relations:
  - {id: RELACIONADO_CON}
  - {id: PARTE_DE}

parse:                          # what counts as a heading — this is what governs step 2
  structure_patterns:
    capitulo:
    - ^SECCIÓN\s+[IVXLCDM]+
    - ^CAPÍTULO\s+\d+
    - ^Capítulo\s+\d+
    - ^PARTE\s+\d+
    seccion:
    - ^\d+\.\d+[\s\.]+[A-ZÁÉÍÓÚÑ]
    - ^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,60}$

chunk:                          # and this is what governs step 3
  target_size: 280
  min_size: 150
  max_size: 380
  overlap: 60
  parent_window: 3
  max_parent_words: 1200
```

Two profiles ship here: **`medicina`** (the default of this instance; its numbers *are* the
historical defaults of the code, and a test asserts that) and **`generico`** (domain-neutral, same
numbers, minimal vocabulary — so "other material" is an explicit choice rather than medicine by
accident).

To add a domain: drop `profiles/mydomain.yaml` next to them and pass
`perfiles.estrategia("mydomain")` to `parse_pdf_v2`. `tests/test_pipeline_estrategia.py` walks the
directory, so your profile is automatically checked (it loads, its regexes compile, its numbers are
coherent). Incoherent numbers — `overlap` at least as big as `min_size`, a parent that cannot hold
one child — raise at construction time, not after you have chunked 10,000 pages.

Why this matters: 280 words is right for medical prose and **wrong** for a statute, where the
article is the unit and splitting it destroys the citation. That is a profile change, not a code
change.

---

## Tests

```bash
pip install -r requirements.txt -r api/requirements.txt      # the second one only for the API tests
pytest                                                       # the whole suite
ruff check pipeline tests api parser_v2.py migrate_chunks.py # the lint gate (same as CI)
```

`tests/test_admin_v1.py` boots the API, so it needs `api/requirements.txt`. Without it that one
file **skips itself** with a message saying what to install — cloning this repo to use only the
pipeline should not hand you a red suite. CI exports `MEDGRAPH_ENGINE_REQUIRE_API=1`, and there
that skip is a failure: the tests that guard the API cannot go quiet because an install step broke.

A bare `pytest` run **never touches the network, a graph, or a key**. The embedding client is a
double, the graph is a pair of functions that record what they were asked to write, the PDFs are
generated inside the test with PyMuPDF (invented text, no copyright), and the sleep between
retries is injected — so the suite runs in well under a second.

| File | What it pins |
|---|---|
| `tests/test_pipeline_parseo.py` | Pure functions, the chunker, the two cures of 2026-09-13 (spaced-letter titles; title by position), reference/index detection, and a synthetic PDF parsed end to end into the canonical embedding text. |
| `tests/test_pipeline_embeddings.py` | The canonical prefix, one content per call, batch retries, misalignment safety, the second pass, and that `llamadas` counts **billable** calls. |
| `tests/test_pipeline_carga.py` | MERGE not CREATE, orphan deletion, embedding invalidation on text change, derived relationships, fail-closed guards, source registration and deletion. |
| `tests/test_pipeline_estrategia.py` | The historical default byte for byte, profiles overriding it, incoherent numbers failing closed, and every YAML in `profiles/` loading and compiling. |
| `tests/test_pipeline_eventos.py` | Event names, fields, reserved field names, errors re-raised, and that no module under `pipeline/` imports the private logger (it only gets `logging`). |
| `tests/test_admin_v1.py` | `admin/v1`: the descriptor and every source validate against the contract's schemas; the descriptor comes from the **active profile** (switching to `generico` changes it end to end); every capability published `false` explains why and how; 401 without a credential, `{detail, code}` on 404, 503 when the graph is down — and a `/admin/v1/health` that still answers 200. The graph is a double: no Neo4j, no network. |
| `tests/test_repo.py` | Repo health: every Python file parses, every dependency is declared, no credential-shaped string anywhere. |
| `tests/test_basic.py` | The v1.0 tests of the root scripts (entity extraction, deduplication) — kept because they are green and cover code nothing else covers. |

Markers are declared in `pytest.ini` for tests that would need a real provider (`integration`) or
a live Neo4j (`neo4j`). There are none today, and both are deselected by default: a marker exists
so that adding such a test later does not silently make `pytest` spend money.

CI (`.github/workflows/ci.yml`) runs exactly the commands above on Python 3.13, with no secrets in
the job.

---

## Known gaps: what this mirror does not have yet

Stated because a README that hides this wastes your afternoon:

- **No multi-format ingestion.** The private pipeline has `pipeline/conversion.py` (DOCX, HTML,
  PPTX, XLSX, CSV, Markdown, TXT, EPUB → the same chunk contract, via `markitdown`). It was never
  replicated here, and publishing it is the owner's call. Today this repo reads **PDF only**; that
  is also why `markitdown` is not in `requirements.txt`.
- **No orchestrator.** There is no single command that runs parse → load → vectorize through
  `pipeline/`. `quickstart.py` does it through the legacy scripts. Use the snippets above, or
  write the ten lines you need.
- **`admin/v1` is read-only.** The five GET routes of the contract are implemented; every write
  (uploads, jobs, classification, review, PATCH, DELETE, re-ingest) and the two optional read
  extras (facets, the unit viewer) are not. The descriptor publishes each one as `false` with a
  note saying what it would take, so a console can show *why* a button is missing instead of a
  screen with something absent.
- **The `/query` router is not profile-driven.** `services/analyzer.py` prompts an LLM with a
  hardcoded medical vocabulary (`Patologia`, `Farmaco`, `CategoriaATC`...). It is the v1.0 router,
  kept because it works, but it does not read `profiles/` the way the descriptor and the parser do.
  On a non-medical profile use `/search/hybrid`, which is domain-neutral.
- **No QA bank, no golden corpus.** The private side tests the load and the retrieval against a
  throwaway Neo4j in Docker and pins the parser output with golden files over a 27-cell
  format × domain matrix. Here the graph-facing tests assert the *statements* against doubles.
- **No structured-logging setup.** `pipeline/eventos.py` emits the events; the private repo has
  the handler that renders them (console `k=v` and JSON for Cloud Logging). Configure `logging`
  however your deployment prefers.
- **No `profiles/derecho.yaml`** (the legal domain profile) and **no `pipeline/pubmed.py`**.

---

## Nomos Graph

MedGraph Engine is the open-source pipeline of **Nomos Graph**, the graph layer of the Nomos
Research Facility ecosystem (a medical instance, a legal one, a clinical one, plus an admin
console). `admin/v1`, the contract that console speaks, is implemented here too: this repo is
its third implementation, and the one that shows the spec is enough — that someone outside
can write one without being inside. The private side adds what is specific to running it as a
service: the orchestrator, the write half of `admin/v1`, the QA bank, per-domain profiles and
the operational tooling.

What lives here is the part that is worth sharing: the pipeline, with its scars documented.

---

## Data and copyright

**This repository contains code only.** No book content, no parsed chunks, no extracted entities,
no database. You provide the documents.

MedGraph is a tool for processing and structuring text. You are solely responsible for holding the
rights to whatever you run through it. The authors do not endorse, facilitate or assume liability
for copyright infringement or unauthorized use of copyrighted material.

---

## Security notes

**Found something exploitable? Do not open an issue.** Use GitHub's private vulnerability
reporting — [`SECURITY.md`](SECURITY.md) has the channel, the scope, and what to expect.

Out of the box this is a local-development setup. Before exposing anything publicly:

- **Auth:** the API uses a single bearer token (`API_KEY`). For production, use real auth (OAuth2 /
  JWT / rotating keys).
- **CORS:** defaults to `localhost`. Add your own origins explicitly; never `allow_origins=["*"]`.
- **Rate limiting:** `slowapi`, 100 req/min by default. Tune it.
- **Credentials:** environment variables or a secret manager. Never in the tree — `.env` is
  gitignored and `tests/test_repo.py` scans for credential-shaped strings.
- **Docker Compose:** the default Neo4j password is `changeme-local-password`. Change it before
  the database is reachable from anywhere else.
- **Input:** queries reach an LLM, not SQL. Sanitize anyway if you expose the API.

Not an exhaustive review. Do your own for your own threat model.

---

## License

**GNU Affero General Public License v3.0** (AGPLv3). You may use, modify and distribute this
software; if you deploy it as a network service, you must make your modified source available to
its users. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — and read
[The pipeline is a mirror](#the-pipeline-is-a-mirror) first: `pipeline/` and `profiles/` are not
edited in this repo.
