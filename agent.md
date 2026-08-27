# Phase 1 Engineering Instructions

## Role and scope

Act as the senior software engineer and technical architect for this product. Build only Phase 1: a small, correct, understandable, and testable RAG backend foundation for an embeddable AI knowledge assistant.

Do not implement Phase 2 SaaS features such as authentication, organizations, users, billing, analytics, widgets, conversation history, roles, or multi-tenancy. Keep future extensibility in mind, but apply YAGNI.

Before changing code:

1. Inspect the repository, `pyproject.toml`, current code, and Git status.
2. Preserve existing work and avoid unrelated changes.
3. Present a concise implementation plan and assumptions.
4. Implement incrementally and verify every meaningful stage.

## Required technology

Use:

- Python 3.12
- FastAPI
- `uv` for project and package management
- PostgreSQL hosted on Supabase free tier
- pgvector for vector similarity search
- SQLAlchemy and Alembic
- Gemini API for the development LLM and embeddings
- PyPDF for PDF parsing
- Pydantic and Pydantic Settings
- pytest and pytest-asyncio
- Ruff

Do not use Django, LangChain, LlamaIndex, Pinecone, Weaviate, Qdrant, Redis, Celery, Kafka, Kubernetes, microservices, CQRS, or event sourcing unless a later explicit requirement justifies one.

Treat Supabase as managed PostgreSQL plus pgvector. Do not couple business logic to the Supabase Python SDK.

Use `pyproject.toml` and `uv.lock`; do not create `requirements.txt`. Use commands such as:

```bash
uv add <package>
uv add --dev <package>
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Architecture

Use pragmatic Clean Architecture with this target layout:

```text
src/
└── app/
    ├── domain/
    │   ├── entities/
    │   └── ports/
    ├── application/
    │   └── use_cases/
    ├── infrastructure/
    │   ├── ai/
    │   ├── database/
    │   └── documents/
    ├── presentation/
    │   └── api/
    │       ├── routes/
    │       └── schemas/
    ├── core/
    │   └── config.py
    ├── dependencies.py
    └── main.py
tests/
├── unit/
└── integration/
```

Dependencies point inward: Presentation → Application → Domain. Infrastructure implements ports owned by inner layers. Domain and Application must not depend directly on FastAPI, Gemini SDK, SQLAlchemy, PostgreSQL, Supabase, or PyPDF.

Create abstractions only when they serve a real boundary. Prefer Python `Protocol` where useful and avoid unnecessary inheritance.

At minimum, consider the useful subset of these domain concepts:

- `Document`
- `DocumentPage`
- `DocumentChunk`
- `RetrievedChunk`
- `SourceReference`
- `Answer`

Define ports for external capabilities:

- `DocumentParser`
- `EmbeddingProvider`
- `VectorRepository`
- `LLMProvider`

## Phase 1 behavior

Implement ingestion:

```text
PDF → page text extraction → cleaning → chunking → embeddings
    → PostgreSQL/pgvector storage with metadata
```

Implement grounded question answering:

```text
question → question embedding → database vector search → sufficiency check
         → grounded LLM answer or refusal → answer with sources
```

The system must answer only from retrieved knowledge-base content. Retrieved context is the sole factual source. Hallucination is worse than refusal, and prompt instructions alone are not an adequate grounding control. Consider retrieval confidence before invoking the LLM.

Use this exact default fallback:

```text
I couldn't find enough information in the knowledge base to answer that question.
```

Unsupported or low-confidence questions return that fallback and no sources. Do not call the LLM when retrieval is insufficient. Supported answers expose document and page sources.

## Application use cases

### `IngestDocument`

Orchestrate parsing, page text extraction, cleaning, chunking, embedding generation, and persistence. Each stored chunk must include:

- document identifier
- original filename
- page number
- chunk index
- content
- embedding
- creation timestamp

Make adding `assistant_id` later straightforward without implementing multi-tenancy now.

### `AskQuestion`

Orchestrate query embedding, retrieval, sufficiency evaluation, grounded answer generation, and source collection. Keep this orchestration out of FastAPI routes.

## API

Implement only:

```text
GET  /health
POST /api/v1/documents
POST /api/v1/chat
```

Chat request:

```json
{"message": "How long is the warranty?"}
```

Supported response shape:

```json
{
  "answer": "The warranty period is two years.",
  "sources": [{"document": "warranty-policy.pdf", "page": 3}]
}
```

Unsupported response shape:

```json
{
  "answer": "I couldn't find enough information in the knowledge base to answer that question.",
  "sources": []
}
```

Routes only validate input, invoke a use case, and serialize output.

## Document ingestion details

For Phase 1, support PDF only and use PyPDF. Preserve page numbers, skip pages without extractable text gracefully, reject invalid or non-PDF content, sanitize filenames, and enforce a configurable upload-size limit. Do not trust client MIME type alone. Do not implement OCR.

Keep chunking isolated and replaceable. Begin with approximately 400–600 words per chunk and 80–120 words of overlap; do not add a framework solely for chunking.

Keep Gemini embeddings behind `EmbeddingProvider`. Configure the model and embedding dimension centrally, ensuring the dimension matches the pgvector column. The use cases must not know which provider is used.

Keep Gemini generation behind `LLMProvider`. Isolate and test prompt construction. Use low temperature and explicitly require using context only, making no assumptions, using no external knowledge, and refusing when context is insufficient.

Perform scoped vector search in PostgreSQL. Do not retrieve globally and filter in application code. Shape repository APIs so an `assistant_id` database filter can be added later.

## Configuration and security

Use Pydantic Settings and environment variables. Eventually create `.env.example` with at least:

```text
DATABASE_URL=
GEMINI_API_KEY=
GEMINI_LLM_MODEL=
GEMINI_EMBEDDING_MODEL=
EMBEDDING_DIMENSION=
RAG_TOP_K=
RAG_SIMILARITY_THRESHOLD=
MAX_UPLOAD_SIZE_MB=
```

Never commit `.env`, hard-code secrets, expose credentials or API keys, or log document contents unnecessarily. Validate question length and uploads, use SQLAlchemy safely, and avoid secrets in errors and logs.

The retrieval threshold must be configuration-driven and tuned through evaluation; do not assume a universal value such as `0.65`.

## Code quality

- Use strong type hints and meaningful names.
- Keep modules and functions focused.
- Avoid giant service classes and circular imports.
- Prefer readability over cleverness.
- Use structured application errors and proper exception handling.
- Avoid broad `except Exception` unless specifically justified.
- Add useful logging without sensitive data.
- Do not leave dead, placeholder, or silently incomplete code.
- Add docstrings only when they add value.

## Testing

Testing is part of implementation. Unit tests must not require PostgreSQL or Gemini. Use fakes such as `FakeDocumentParser`, `FakeEmbeddingProvider`, `FakeVectorRepository`, and `FakeLLMProvider`.

Cover ingestion behavior:

- valid PDF processing
- empty-page handling
- preservation of page metadata
- embedding generation
- chunk persistence

Cover question behavior:

- relevant context produces a grounded answer
- answer sources are returned
- missing context produces the fallback
- low-confidence results produce the fallback
- the LLM is not called when retrieval is insufficient

Add limited integration coverage for SQLAlchemy/PostgreSQL, pgvector similarity search, and FastAPI endpoints. Normal tests must not call paid or external LLM APIs.

## RAG evaluation

Provide a simple evaluation mechanism for one sample document with at least 10 answerable and 10 deliberately unanswerable questions. Record:

- question
- expected behavior
- actual answer
- correctness
- expected source
- retrieved source
- similarity score

Initial targets:

- answerable questions: at least 90% correct
- unanswerable questions: 100% correctly refused
- source attribution: at least 90% correct

## Implementation order

1. Initialize the `uv` project and Python 3.12 configuration.
2. Configure the `src` layout and dependencies.
3. Configure Ruff, pytest, and pytest-asyncio.
4. Add validated application settings.
5. Add the health endpoint.
6. Add domain entities and ports.
7. Configure SQLAlchemy, pgvector, and Alembic.
8. Create the database migration.
9. Implement PDF parsing and text chunking.
10. Implement the Gemini embedding adapter.
11. Implement the PostgreSQL vector repository.
12. Implement `IngestDocument` and its unit tests.
13. Add the document upload endpoint and ingestion integration tests.
14. Implement vector retrieval and the Gemini LLM adapter.
15. Implement `AskQuestion` and its unit tests.
16. Add the chat endpoint and API integration tests.
17. Add the RAG evaluation mechanism and tune retrieval settings.
18. Complete the README and final verification.

After each meaningful stage, run relevant tests and Ruff, fix failures, and inspect the result before continuing. At completion, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Do not claim verification that was not performed. State exactly what remains unverified when credentials or external services are unavailable.

## Git discipline and documentation

Keep changes focused and do not commit secrets. Useful commit boundaries include:

```text
chore: initialize FastAPI project with uv
feat: add document ingestion domain
feat: add pgvector repository
feat: implement document ingestion
feat: implement grounded question answering
test: add RAG use case tests
```

The completed README must explain purpose, architecture, prerequisites, `uv` setup, environment variables, database setup, migrations, local development, tests, and endpoints. Add a Mermaid architecture diagram if useful.

## Phase 1 definition of done

Phase 1 is complete only when:

1. FastAPI starts successfully and `/health` works.
2. A valid PDF can be uploaded, parsed by page, cleaned, and chunked.
3. Embeddings are generated and chunks plus metadata are stored with pgvector.
4. Question embeddings retrieve relevant chunks through database vector search.
5. The LLM answers only from sufficient retrieved context.
6. Unsupported questions are refused with the exact fallback.
7. Answers include correct document/page sources.
8. Core use cases have isolated unit tests.
9. PostgreSQL and vector functionality have integration coverage.
10. Ruff and all tests pass.
11. The README accurately explains setup and operation.

If a requested package or model is deprecated or renamed, verify current official documentation and use the current stable supported equivalent without changing these architectural boundaries.
