"""MedGraph Quickstart -- Get up and running in 5 minutes.

Place a PDF in the examples/ folder, configure .env, and run this script.
It will parse your PDF, generate embeddings, extract medical entities,
and verify the system works with a sample query.

Usage:
  python quickstart.py
  python quickstart.py --skip-extract   # skip entity extraction (faster)
"""

import os
import sys
import glob
import time

def check_env():
    """Verify .env is configured."""
    print("\n[1/7] Checking environment...")

    if not os.path.exists(".env"):
        print("  ERROR: .env file not found.")
        print("  Run: cp .env.example .env")
        print("  Then edit .env with your Neo4j and GCP credentials.")
        return False

    from dotenv import load_dotenv
    load_dotenv()

    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]

    if missing:
        print(f"  ERROR: Missing environment variables: {', '.join(missing)}")
        print("  Edit your .env file with the correct values.")
        return False

    print("  OK - Environment configured")
    return True


def check_dependencies():
    """Verify Python dependencies are installed."""
    print("\n[2/7] Checking dependencies...")

    deps = {
        "neo4j": "neo4j",
        "fitz": "PyMuPDF (pip install pymupdf)",
        "dotenv": "python-dotenv",
    }

    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"  ERROR: Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False

    print("  OK - All dependencies installed")
    return True


def setup_schema():
    """Create Neo4j schema and indexes."""
    print("\n[3/7] Setting up Neo4j schema...")

    try:
        from schema import create_schema
        create_schema()
        print("  OK - Schema and indexes created")
        return True
    except ImportError:
        print("  WARN: schema.py not found, skipping schema creation")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Check your Neo4j credentials in .env")
        return False


def find_pdf():
    """Find a PDF in the examples folder."""
    print("\n[4/7] Looking for PDFs in examples/...")

    pdfs = glob.glob("examples/*.pdf")

    if not pdfs:
        print("  ERROR: No PDF files found in examples/")
        print("  Place a PDF in the examples/ folder and try again.")
        print("  See examples/README.md for suggestions on free medical PDFs.")
        return None

    pdf = pdfs[0]
    print(f"  Found: {pdf}")

    if len(pdfs) > 1:
        print(f"  (Using first one. {len(pdfs)} PDFs found total)")

    return pdf


def run_pipeline(pdf_path, skip_extract=False):
    """Run the full MedGraph pipeline on a PDF."""

    # Derive libro_id from filename
    libro_id = os.path.splitext(os.path.basename(pdf_path))[0]
    libro_id = libro_id.lower().replace(" ", "-").replace("_", "-")

    print(f"\n  Book ID: {libro_id}")

    # Step 1: Parse
    print("\n[5/7] Parsing PDF...")
    try:
        from parser_v2 import parse_libro_v2
        children, parents = parse_libro_v2(libro_id, pdf_path)
        print(f"  OK - {len(children)} chunks, {len(parents)} parent chunks")
    except Exception as e:
        print(f"  ERROR parsing: {e}")
        return False

    if not children:
        print("  ERROR: No chunks generated. The PDF might be scanned (needs OCR).")
        return False

    # Step 2: Upload to Neo4j
    print("\n[6/7] Uploading to Neo4j and generating embeddings...")
    try:
        from migrate_chunks import upload_chunks_for_libro, normalize_chunks
        normalize_chunks(children)
        stats = upload_chunks_for_libro(libro_id, children, parents)
        print(f"  Uploaded: {stats}")
    except ImportError:
        try:
            from upload_chunks import upload_chunks
            upload_chunks(libro_id, children, parents)
            print(f"  Uploaded: {len(children)} chunks")
        except Exception as e:
            print(f"  ERROR uploading: {e}")
            return False
    except Exception as e:
        print(f"  ERROR uploading: {e}")
        return False

    # Step 3: Vectorize
    try:
        from vectorize import vectorize_libro
        vectorize_libro(libro_id)
        print(f"  OK - Embeddings generated")
    except Exception as e:
        print(f"  WARN: Vectorization failed: {e}")
        print("  You may need to configure GCP_API_KEY in .env")

    # Step 4: Extract entities (optional)
    if skip_extract:
        print("\n[7/7] Skipping entity extraction (--skip-extract)")
    else:
        print("\n[7/7] Extracting medical entities (this may take a few minutes)...")
        try:
            from extract_entities import init_model, extract_from_chunk, canonicalize_entities, upload_entities
            import json

            model = init_model()
            all_ext = []

            for i, chunk in enumerate(children):
                ext = extract_from_chunk(model, chunk, libro_id)
                all_ext.append(ext)
                if (i + 1) % 10 == 0:
                    pct = int((i + 1) / len(children) * 100)
                    print(f"  [{pct}%] {i + 1}/{len(children)} chunks processed")
                if (i + 1) % 5 == 0:
                    time.sleep(1)  # Rate limiting

            entities, relations = canonicalize_entities(all_ext)
            print(f"  Extracted: {len(entities)} entities, {len(relations)} relationships")

            # Save extraction
            os.makedirs("extracted", exist_ok=True)
            with open(f"extracted/{libro_id}_entities.json", "w", encoding="utf-8") as f:
                json.dump({"extractions": all_ext}, f, ensure_ascii=False)

            upload_entities(entities, relations, libro_id, dev_mode=False)
            print(f"  OK - Entities uploaded to Neo4j")
        except Exception as e:
            print(f"  WARN: Entity extraction failed: {e}")
            print("  The system still works for bibliography search without entities.")

    return True


def verify():
    """Run a sample query to verify the system works."""
    print("\n--- Verification ---")

    try:
        from db import run_query

        chunks = run_query("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"]
        entities = run_query("MATCH (n) WHERE NOT n:Chunk AND NOT n:ParentChunk RETURN count(n) AS n")[0]["n"]
        rels = run_query("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]

        print(f"\n  Your MedGraph instance:")
        print(f"  Chunks:       {chunks:,}")
        print(f"  Entities:     {entities:,}")
        print(f"  Relationships:{rels:,}")

        if chunks > 0:
            print("\n  SUCCESS! MedGraph is ready.")
            print(f"\n  Next steps:")
            print(f"  - Start the API:  cd api && uvicorn main:app --reload")
            print(f"  - Add more books:  python quickstart.py  (with new PDFs in examples/)")
            print(f"  - Run ontology:    python ontology.py")
            return True
        else:
            print("\n  WARNING: No chunks found. Something went wrong in the pipeline.")
            return False

    except Exception as e:
        print(f"  ERROR verifying: {e}")
        return False


def main():
    print("=" * 50)
    print("  MedGraph Quickstart")
    print("=" * 50)

    skip_extract = "--skip-extract" in sys.argv

    if not check_env():
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    if not setup_schema():
        sys.exit(1)

    pdf = find_pdf()
    if not pdf:
        sys.exit(1)

    if not run_pipeline(pdf, skip_extract=skip_extract):
        sys.exit(1)

    verify()

    print("\n" + "=" * 50)
    print("  Done!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
