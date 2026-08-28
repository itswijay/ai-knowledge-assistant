# AI Knowledge Assistant

A secure multi-tenant FastAPI backend for building grounded AI knowledge
assistants. Organizations create assistants, upload private PDF documents, and
ask questions whose answers are generated only from the selected assistant's
knowledge base.

The application prefers refusal over unsupported answers. When retrieval is
missing or below the configured confidence threshold, it does not call the LLM
and returns:

```text
I couldn't find enough information in the knowledge base to answer that question.
```

## Phase 2 capabilities

- Supabase Auth bearer-token verification through cached public JWKS keys
- Organizations with `owner`, `admin`, and `member` roles
- Assistant creation, customization, listing, updates, and deletion
- Assistant-owned documents and document chunks
- Server-side authorization for every tenant-owned resource
- Assistant filtering inside PostgreSQL vector-search queries
- PDF validation, filename sanitization, and configurable upload limits
- Page-preserving extraction, chunking, Gemini embeddings, and pgvector storage
- Grounded Gemini answers with document and page citations
- Defensive PostgreSQL RLS and grants for Supabase-exposed tables
- Unit, API integration, and opt-in PostgreSQL tenant-isolation tests
- Assistant-scoped RAG evaluation tooling

## Architecture

The code follows pragmatic Clean Architecture. Presentation depends on
application use cases, application depends on domain abstractions, and
infrastructure implements those abstractions.

```mermaid
flowchart LR
    User[User] --> Auth[Supabase Auth]
    Auth -->|Access token| API[FastAPI presentation]
    API --> UseCases[Application use cases]
    UseCases --> Domain[Domain entities and ports]
    UseCases --> Authorization[Authorization services]
    Infrastructure[Infrastructure adapters] -. implements .-> Domain
    Infrastructure --> JWKS[Supabase JWKS]
    Infrastructure --> Gemini[Gemini API]
    Infrastructure --> Postgres[(Supabase PostgreSQL + pgvector)]

    Organization --> Assistant
    Assistant --> Document
    Document --> Chunk[Document chunks]
```

The principal data boundary is:

```text
Authenticated user
  -> organization membership
     -> assistant
        -> documents
           -> document chunks
```

The ingestion and question flows are:

```text
authorize assistant manager
  -> validate PDF -> parse pages -> clean -> chunk -> embed -> store

authorize assistant member
  -> embed question -> assistant-filtered vector search -> confidence check
  -> grounded generation or refusal -> answer with sources
```

Key directories:

```text
src/app/
├── domain/          # Entities, errors, roles, and external-capability ports
├── application/     # Authorization services and business use cases
├── infrastructure/  # Supabase JWT, PostgreSQL/pgvector, Gemini, and PDF adapters
├── presentation/    # FastAPI routes, schemas, authentication, and error mapping
├── evaluation/      # Assistant-scoped evaluation models, scoring, and CLI
├── core/config.py   # Validated environment settings
├── dependencies.py  # Runtime composition root
└── main.py          # FastAPI application factory
```

## Authentication and authorization

Supabase Auth owns users and authentication. This backend does not store
passwords or implement login sessions. Clients send a Supabase access token on
authenticated requests:

```http
Authorization: Bearer <access-token>
```

The backend derives the issuer and JWKS URLs from `SUPABASE_URL`, resolves the
token's asymmetric signing key, and verifies the signature, issuer, audience,
expiry, issued-at time, subject, authenticated role, and non-anonymous status.
JWKS responses are cached; the backend does not call the Supabase user endpoint
for every request. Supported signing algorithms are ES256, RS256, and EdDSA.

The verified token subject is the only source of the user ID. Tenant-owned
resource IDs supplied by clients are always checked against persisted ownership
and membership.

### Role behavior

| Operation | Owner | Admin | Member |
| --- | --- | --- | --- |
| Read organization and list assistants | Yes | Yes | Yes |
| Read an assistant and its documents | Yes | Yes | Yes |
| Ask an assistant a question | Yes | Yes | Yes |
| Create an assistant | Yes | Yes | No |
| Update or delete an assistant | Yes | Yes | No |
| Upload or delete a document | Yes | Yes | No |

Creating an organization automatically creates an `owner` membership for the
authenticated user. Membership-management and invitation endpoints are not part
of Phase 2.

Unknown resources and resources belonging to another tenant return `404` so the
API does not reveal their existence. A known member with an insufficient role
receives `403`.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A Supabase project with Auth and PostgreSQL
- PostgreSQL with the `vector` extension available
- A Gemini API key
- Asymmetric JWT signing enabled for the Supabase project

Supabase is treated as an authentication and database provider. The backend uses
PyJWT for token verification and SQLAlchemy with asyncpg for database access; it
does not require the Supabase Python SDK or a service-role API key.

## Local setup

Install the locked dependencies:

```bash
uv sync --dev
```

Create the local environment file:

```bash
cp .env.example .env
```

Replace the example values. Never commit `.env`, database credentials, Gemini
keys, access tokens, or Supabase service-role keys.

### Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `DATABASE_URL` | Yes | — | Backend PostgreSQL URL using `postgresql+asyncpg` |
| `SUPABASE_URL` | Yes | — | Supabase project origin, with no path or credentials |
| `SUPABASE_JWT_AUDIENCE` | No | `authenticated` | Expected access-token audience |
| `SUPABASE_JWKS_CACHE_TTL_SECONDS` | No | `600` | JWKS cache lifetime, from 60 to 3,600 seconds |
| `GEMINI_API_KEY` | Yes | — | Gemini API credential |
| `GEMINI_LLM_MODEL` | No | `gemini-3.7-flash` | Answer-generation model |
| `GEMINI_MAX_OUTPUT_TOKENS` | No | `512` | Maximum generated answer tokens |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-2` | Document and query embedding model |
| `EMBEDDING_DIMENSION` | No | `768` | Schema-fixed embedding length |
| `RAG_TOP_K` | No | `5` | Maximum chunks retrieved per question |
| `RAG_SIMILARITY_THRESHOLD` | No | `0.65` | Minimum score required before generation |
| `MAX_UPLOAD_SIZE_MB` | No | `10` | Maximum accepted PDF size in MiB |

`RUN_POSTGRES_INTEGRATION_TESTS` and `TEST_DATABASE_URL` are test-only variables
described under [PostgreSQL integration testing](#postgresql-integration-testing).

The embedding dimension is fixed at 768 in the SQLAlchemy model and PostgreSQL
schema. Changing it requires a new migration, regeneration of every stored
embedding, and rebuilding any vector indexes.

## Supabase setup

### Auth

Set `SUPABASE_URL` to the project origin:

```env
SUPABASE_URL=https://your-project-ref.supabase.co
```

The backend derives these endpoints:

```text
issuer: <SUPABASE_URL>/auth/v1
JWKS:   <SUPABASE_URL>/auth/v1/.well-known/jwks.json
```

Use an asymmetric Supabase JWT signing key compatible with ES256, RS256, or
EdDSA. The legacy shared-secret HS256 configuration is intentionally unsupported
because it would require distributing a signing secret to the API.

Create test users through the Supabase Auth dashboard or a trusted client flow.
For local manual testing, a client can exchange a test user's credentials at the
Supabase Auth token endpoint using the project's public publishable/anon key:

```bash
curl -X POST "$SUPABASE_URL/auth/v1/token?grant_type=password" \
  -H "apikey: $SUPABASE_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"email":"developer@example.com","password":"test-password"}'
```

Use the response's `access_token` as `ACCESS_TOKEN` in the examples below. The
public key shown here is needed only by the client login flow; it is not a backend
environment setting. Do not use a service-role key for client authentication.

### Database and migrations

Use the PostgreSQL connection string from Supabase and change its scheme to
`postgresql+asyncpg`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@host:5432/postgres
```

Apply all migrations in order:

```bash
uv run alembic upgrade head
```

The migration history is:

- `0001` enables pgvector and creates `documents` and `document_chunks`.
- `0002` creates organizations, memberships, and assistants, then adds the
  required `documents.assistant_id` relationship and tenant-query indexes.
- `0003` enables RLS and revokes tenant-table privileges from Supabase's `anon`
  and `authenticated` database roles.

Migration `0002` deletes existing Phase 1 documents and chunks before making
`assistant_id` mandatory. Those rows were development data without an owner.
Back up any data that must be retained before applying the migration.

The migration role must be able to create tables, foreign keys into
`auth.users`, the pgvector extension, RLS settings, and grants. The role used by
`DATABASE_URL` at runtime must own the protected tables or have PostgreSQL
`BYPASSRLS`; migration `0003` defines no direct-client RLS policies. This is
intentional: authorization happens in FastAPI, while RLS and revoked grants deny
direct access through Supabase's public Data API roles.

Do not configure the runtime backend with the `anon` or `authenticated` database
role. Confirm the role and RLS behavior in a non-production environment before
deployment.

Useful migration commands:

```bash
uv run alembic current
uv run alembic history
uv run alembic downgrade 0002
```

Downgrading from `0003` to `0002` disables RLS and restores direct table grants
to `anon` and `authenticated`; it intentionally weakens isolation. Downgrading
`0002` removes all Phase 2 tenant tables and relationships.

## Run the API

Start the development server:

```bash
uv run uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive OpenAPI docs are at
`http://127.0.0.1:8000/docs`.

The public health check is:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

All `/api/v1` endpoints require a bearer token. Export one obtained from
Supabase Auth:

```bash
export ACCESS_TOKEN="replace-with-a-test-user-access-token"
```

### API endpoints

| Method | Path | Required access |
| --- | --- | --- |
| `GET` | `/health` | Public |
| `GET` | `/api/v1/me` | Authenticated user |
| `POST` | `/api/v1/organizations` | Authenticated user; creator becomes owner |
| `GET` | `/api/v1/organizations` | Authenticated user; lists memberships only |
| `GET` | `/api/v1/organizations/{organization_id}` | Organization member |
| `POST` | `/api/v1/organizations/{organization_id}/assistants` | Owner or admin |
| `GET` | `/api/v1/organizations/{organization_id}/assistants` | Organization member |
| `GET` | `/api/v1/assistants/{assistant_id}` | Organization member |
| `PATCH` | `/api/v1/assistants/{assistant_id}` | Owner or admin |
| `DELETE` | `/api/v1/assistants/{assistant_id}` | Owner or admin |
| `POST` | `/api/v1/assistants/{assistant_id}/documents` | Owner or admin |
| `GET` | `/api/v1/assistants/{assistant_id}/documents` | Organization member |
| `DELETE` | `/api/v1/documents/{document_id}` | Owner or admin |
| `POST` | `/api/v1/assistants/{assistant_id}/chat` | Organization member |

### Create an organization

```bash
curl -X POST http://127.0.0.1:8000/api/v1/organizations \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Acme Support"}'
```

Save the returned `id` as `ORGANIZATION_ID`.

### Create an assistant

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/organizations/$ORGANIZATION_ID/assistants" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"Support Assistant",
    "description":"Answers product-support questions",
    "welcome_message":"How can I help?",
    "primary_color":"#2563EB"
  }'
```

Assistant responses also contain `system_prompt`, `logo_url`, and timestamps.
Save the returned `id` as `ASSISTANT_ID`.

### Upload a document

Only text-based, unencrypted PDFs are supported. Client-provided MIME type is not
trusted; the filename, PDF signature, structure, and configured size limit are
validated.

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/assistants/$ASSISTANT_ID/documents" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@/absolute/path/to/warranty-policy.pdf;type=application/pdf"
```

Example response:

```json
{
  "document_id": "6eb56b83-2ea1-4b99-b22b-3e0355c74e10",
  "assistant_id": "f831e31c-b0f8-4478-a8e5-4cf44a61ab39",
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
curl -X POST \
  "http://127.0.0.1:8000/api/v1/assistants/$ASSISTANT_ID/chat" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
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

## Tenant-isolation guarantees

Isolation is enforced in several layers:

1. JWT verification derives a trusted user ID.
2. Application authorization verifies organization membership and role.
3. Every document is linked to exactly one assistant.
4. Vector retrieval joins chunks to documents and filters by `assistant_id`
   inside PostgreSQL before ordering or limiting results.
5. Cross-tenant API lookups are concealed as `404`.
6. RLS and revoked public-client grants block direct tenant-table access through
   Supabase's `anon` and `authenticated` database roles.

Filtering results in Python after a global vector search is not used. The normal
backend role is trusted to bypass RLS only so FastAPI can perform its already
authorized operations.

## Tests and code quality

Normal tests use fakes and do not require PostgreSQL, Supabase Auth, Gemini
credentials, or paid API calls:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

API integration tests cover authentication dependencies, validation,
serialization, authorization error mapping, and assistant-scoped routes. Unit
tests cover JWT validation, authorization services, repositories, ingestion,
retrieval, grounding, refusal, and regressions.

### PostgreSQL integration testing

The live pgvector isolation test is disabled by default because it creates and
deletes organizations, assistants, documents, and chunks. Run it only against a
disposable, fully migrated PostgreSQL database:

```bash
RUN_POSTGRES_INTEGRATION_TESTS=1 \
TEST_DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/assistant_test \
uv run pytest tests/integration/database/test_tenant_isolation.py
```

`TEST_DATABASE_URL` must differ from `DATABASE_URL`, must use asyncpg, and must
connect with a table-owning or `BYPASSRLS` role. Never point it at production or a
database containing data that cannot be safely deleted. The test proves that a
query for Assistant A cannot retrieve Assistant B's higher-scoring document, and
vice versa.

## RAG evaluation

The checked-in evaluation dataset defines a generated three-page product guide,
12 answerable questions, and 10 deliberately unanswerable questions. The target
assistant and authenticated owner/admin user must already exist and be related by
an organization membership.

Run the live evaluation after migrations have been applied and Gemini credentials
are configured:

```bash
uv run python -m app.evaluation.run \
  --user-id "$USER_ID" \
  --assistant-id "$ASSISTANT_ID"
```

The evaluation ingests the sample PDF into only that assistant, runs each
question through assistant-scoped retrieval, and writes
`evaluations/results/latest.json`. To reuse an already-ingested sample:

```bash
uv run python -m app.evaluation.run \
  --user-id "$USER_ID" \
  --assistant-id "$ASSISTANT_ID" \
  --skip-ingestion
```

Targets:

- answerable-question accuracy: at least 90%
- unanswerable-question refusal accuracy: 100%
- source attribution accuracy: at least 90%

Evaluation results are ignored by Git because they may contain model output and
vary with models and retrieval settings. Tune `RAG_TOP_K` and
`RAG_SIMILARITY_THRESHOLD` from measured results.

## Security and operational notes

- Keep database credentials, Gemini keys, and access tokens in `.env` or a
  deployment secret manager, never in source control.
- Do not log JWTs, uploaded document contents, prompts, embeddings, or secrets.
- Use separate migration, runtime, and test database credentials when practical.
- Retrieved document text is untrusted reference data, not model instructions.
- Unsupported and low-confidence questions bypass answer generation.
- Cascade deletion means deleting an organization removes its assistants,
  documents, chunks, and memberships; deleting an assistant removes its
  documents and chunks.
- `/health` reports process availability, not database, Supabase Auth, or Gemini
  readiness.
- Source citations identify retrieved documents and pages; continue evaluating
  retrieval and grounding before production use.

## Current limitations

- Chat is authenticated and intended for admin testing; no public widget endpoint
  exists yet.
- There are no membership-management, invitation, billing, analytics, rate-limit,
  streaming, or conversation-history features.
- PDF ingestion has no OCR, background jobs, or duplicate detection.
- Vector search has no approximate-nearest-neighbor index; exact search is
  appropriate only for a modest corpus.
- The trusted backend database role can bypass RLS, so correct FastAPI
  authorization and assistant-scoped repository queries remain security-critical.
- The default similarity threshold must be tuned against deployed models and
  representative tenant data.
