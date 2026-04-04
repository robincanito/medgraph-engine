# Contributing to MedGraph

Thanks for your interest in contributing! Here's how you can help.

## Ways to Contribute

- **Report bugs** — Open an issue describing the problem, steps to reproduce, and expected behavior
- **Suggest features** — Open an issue with your idea and how it would improve the system
- **Create DAGs** — Clinical reasoning flows for new medical topics (YAML format in `dags/`)
- **Improve the parser** — Better structure detection for different book formats
- **Improve entity extraction** — Better prompts, validation, or canonicalization
- **Add ontology mappings** — Extend ATC/SNOMED coverage
- **Documentation** — Fix typos, improve explanations, add examples

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/medgraph.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and configure your credentials
5. Create a branch: `git checkout -b feature/my-feature`
6. Make your changes
7. Test locally
8. Commit and push
9. Open a Pull Request

## Code Style

- Python code follows standard conventions
- No hardcoded credentials — use environment variables
- Functions should have docstrings explaining what they do
- Keep it simple — avoid over-engineering

## Data and Copyright

- **Do NOT commit** copyrighted book content (PDFs, parsed chunks, extracted entities)
- **Do NOT commit** credentials or API keys
- DAGs, schema definitions, and pipeline code are fine to commit

## Questions?

Open an issue on GitHub.
