# Examples

Place your PDFs here and run `quickstart.py` to process them through the full MedGraph pipeline.

## Getting a test PDF

MedGraph doesn't include copyrighted textbooks. You need to bring your own. For testing, you can use any freely available medical document:

- **WHO Guidelines**: [who.int/publications](https://www.who.int/publications)
- **PubMed Central**: [ncbi.nlm.nih.gov/pmc](https://www.ncbi.nlm.nih.gov/pmc/) (open access papers)

## How it works

1. Put a PDF in this folder
2. Run `python quickstart.py`
3. The pipeline will: parse → chunk → vectorize → extract entities
4. Everything is stored in YOUR Neo4j instance — nothing leaves your infrastructure

## Important

- No copyrighted material is included in this repository
- The PDF content is processed locally
- Embeddings are generated via Google Gemini API (requires GCP credentials)
- Extracted entities are stored in your Neo4j instance
