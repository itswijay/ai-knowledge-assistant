# AI Knowledge Assistant

A small Phase 1 backend for an embeddable, grounded knowledge assistant. It
ingests text-based PDF documents, stores page-aware chunks and Gemini embeddings
in PostgreSQL with pgvector, retrieves relevant context, and asks Gemini to answer
only from that context.

The application prefers refusal over unsupported answers. When retrieval is
missing or below the configured confidence threshold, it does not call the LLM
and returns:

```text
I couldn't find enough information in the knowledge base to answer that question.
```

## Phase 1 capabilities

- PDF validation, filename sanitization, and configurable upload limits
- Page-preserving text extraction with PyPDF; empty pages are skipped
- Replaceable word chunking with 500-word chunks and 100-word overlap
- Gemini document and query embeddings behind an application port
- PostgreSQL persistence and cosine-similarity search through pgvector
- Retrieval-confidence checks before answer generation
- Grounded Gemini prompts with retrieved context treated as untrusted data
- Document and page citations for supported answers
- FastAPI health, document upload, and chat endpoints
- A reproducible RAG evaluation suite with answer, refusal, source, and similarity
  metrics

This phase intentionally excludes authentication, organizations, billing,
analytics, widgets, conversation history, OCR, and multi-tenancy.

## Architecture

The code follows pragmatic Clean Architecture. Dependencies point inward, while
provider-specific code stays behind domain ports.

```mermaid
flowchart LR
    Client[API client] --> Presentation[FastAPI presentation]
    Presentation --> UseCases[Application use cases]
    UseCases --> Domain[Domain entities and ports]
    Infrastructure[Infrastructure adapters] -. implements .-> Domain
    Infrastructure --> Gemini[Gemini API]
    Infrastructure --> Postgres[(PostgreSQL + pgvector)]
    Infrastructure --> PyPDF[PyPDF]
```

The two primary flows are:

```text
PDF -> validate -> parse pages -> clean -> chunk -> embed -> store

question -> embed -> vector search -> confidence check
         -> grounded generation or refusal -> answer with sources
```

Key directories:

```text
src/app/
├── domain/          # Entities, errors, and external-capability ports
├── application/     # Chunking, prompt construction, and use cases
├── infrastructure/  # Gemini, PostgreSQL/pgvector, and PDF adapters
├── presentation/    # FastAPI routes, schemas, and exception mapping
├── evaluation/      # Evaluation models, scoring, PDF generation, and CLI
├── core/config.py   # Validated environment settings
├── dependencies.py  # Runtime composition root
└── main.py          # FastAPI application factory
```

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- PostgreSQL with the `vector` extension available, such as a Supabase project
- A Gemini API key

Supabase is used only as managed PostgreSQL. The application does not depend on
the Supabase Python SDK.

## Local setup

Clone the repository, then install the locked dependencies:

```bash
uv sync --dev
```

Create the local environment file:

```bash
cp .env.example .env
```

Replace the example database credentials and Gemini key. Never commit `.env`.

### Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `DATABASE_URL` | Yes | — | PostgreSQL URL using the `postgresql+asyncpg` scheme |
| `GEMINI_API_KEY` | Yes | — | Gemini API credential |
| `GEMINI_LLM_MODEL` | No | `gemini-3.7-flash` | Answer-generation model |
| `GEMINI_MAX_OUTPUT_TOKENS` | No | `512` | Maximum generated answer tokens |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-2` | Document and query embedding model |
| `EMBEDDING_DIMENSION` | No | `768` | Expected embedding vector length |
| `RAG_TOP_K` | No | `5` | Maximum chunks retrieved per question |
| `RAG_SIMILARITY_THRESHOLD` | No | `0.70` | Minimum score required before generation |
| `MAX_UPLOAD_SIZE_MB` | No | `10` | Maximum accepted PDF size in MiB |

The embedding dimension used by Gemini, the application, and the pgvector data
must remain consistent. Changing it after documents have been ingested requires
re-embedding those documents.

### Database setup

Use the PostgreSQL connection string from your provider and change its scheme to
`postgresql+asyncpg`. For example:

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@host:5432/postgres
```

Apply the schema:

```bash
uv run alembic upgrade head
```

The initial migration enables the `vector` extension and creates the `documents`
and `document_chunks` tables. The database role must be allowed to create the
extension. If the provider restricts that permission, enable pgvector from its
dashboard or SQL editor before applying the migration.

Useful migration commands:

```bash
uv run alembic current
uv run alembic history
uv run alembic downgrade base
```

Downgrading removes the Phase 1 document tables and their data. It does not remove
the pgvector extension.

## Run the API

Start the development server:

```bash
uv run uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive OpenAPI docs are at
`http://127.0.0.1:8000/docs`.

### Health

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

### Upload a document

Only text-based, unencrypted PDFs are supported. Client-provided MIME type is not
trusted; the filename, PDF signature, structure, and configured size limit are
validated.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -F "file=@/absolute/path/to/warranty-policy.pdf;type=application/pdf"
```

Example response:

```json
{
  "document_id": "6eb56b83-2ea1-4b99-b22b-3e0355c74e10",
  "original_filename": "warranty-policy.pdf",
  "processed_page_count": 4,
  "chunk_count": 6
}
```

Scanned image-only PDFs require OCR and are rejected when they contain no
extractable text.

### Ask a question

Messages are trimmed, must not be blank, and may contain at most 2,000 characters.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How long is the warranty?"}'
```

Supported response:

```json
{
  "answer": "The warranty period is two years.",
  "sources": [{"document": "warranty-policy.pdf", "page": 3}]
}
```

Unsupported response:

```json
{
  "answer": "I couldn't find enough information in the knowledge base to answer that question.",
  "sources": []
}
```

## Tests and code quality

Normal tests use fakes and do not require PostgreSQL, Gemini credentials, or paid
API calls.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

To apply formatting:

```bash
uv run ruff format .
```

The API integration tests exercise FastAPI request validation, dependency
overrides, response serialization, and error mapping. Database and Gemini adapter
tests are isolated unit tests; a live PostgreSQL/pgvector and Gemini end-to-end run
requires configured external services.

## RAG evaluation

The checked-in evaluation dataset defines a generated three-page product guide,
12 answerable questions, and 10 deliberately unanswerable questions. A live run:

1. generates and ingests the sample PDF;
2. executes each question through the real configured RAG stack;
3. records the actual answer, expected behavior, correctness, expected and
   retrieved source, and top similarity score;
4. writes `evaluations/results/latest.json`; and
5. exits unsuccessfully if any Phase 1 target is missed.

Run it only after migrations have been applied and valid Gemini credentials are
configured:

```bash
uv run python -m app.evaluation.run
```

Targets:

- answerable-question accuracy: at least 90%
- unanswerable-question refusal accuracy: 100%
- source attribution accuracy: at least 90%

The command ingests another copy of the sample document on each default run. To
evaluate a copy that is already present, use:

```bash
uv run python -m app.evaluation.run --skip-ingestion
```

Evaluation results are intentionally ignored by Git because they may contain
model output and vary with models and retrieval settings. Tune `RAG_TOP_K` and
`RAG_SIMILARITY_THRESHOLD` from measured results rather than treating the defaults
as universal values.

## Security and operational notes

- Keep database credentials and Gemini API keys only in `.env` or a deployment
  secret manager.
- Do not log uploaded document contents, prompts, embeddings, or secrets.
- Retrieved document text is treated as untrusted reference data, not as prompt
  instructions.
- Unsupported and low-confidence questions bypass answer generation.
- Source references identify retrieved documents and pages; they are not proof
  that every sentence in a generated response is correct. Continue evaluating
  retrieval and grounding before production use.
- `/health` reports process availability, not live database or Gemini readiness.

## Current limitations

- Phase 1 has a single global knowledge base; there is no tenant or assistant
  scope yet.
- PDF ingestion has no OCR, background jobs, or duplicate-document detection.
- Vector search has no approximate-nearest-neighbor index; this is appropriate
  only for a small Phase 1 corpus.
- Chat has no authentication, rate limiting, streaming, or conversation memory.
- The default similarity threshold is a starting value and must be tuned with the
  included evaluation against the deployed models and data.
