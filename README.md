# Lexory


# System Overview

Lexory is a prototype system that generates grammar lessons from users’ real-life texts.

It is designed for advanced language learners and native speakers.

For those who use the language every day in regular communication.

For people who do not have much time or energy for formal language classes and want learning to be close to their real-life usage.

Also for those who do not have enough patience to read entire textbooks.

Lexory is created to analyze and assess grammar gaps based on a user’s real-life language use.

It is designed to function like a personal tutor–copilot that can identify weaker areas by observing how you speak or by analyzing texts you write for any purpose, without requiring formal tests. It then attempts to teach you using different pedagogical approaches until it finds what works best for you.

Qdrant collections behave as user repositories (TODO: add pseudonymization after LanguageTool, before vectorization).

### Pipeline

```
user text
→ grammar detection
→ embedding generation
→ vector retrieval
→ lesson generation
```

### Stack

- Backend: FastAPI
- Vector database: Qdrant
- Grammar detection: LanguageTool
- LLM generation: Ollama
- Storage: PostgreSQL (relational data) + Qdrant (vectors)
- Schema: SQLAlchemy + [Alembic](https://alembic.sqlalchemy.org/) (migrations)
- Infrastructure: Docker

### Current State

Semi-functional prototype.

Some retrieval logic is intentionally simplified while the system pipeline is being stabilized.

---

# Engineering Notes

This project explores a **RAG-based architecture for grammar learning systems**.

Several design decisions were made during development.

---

## Semantic deduplication of examples

Examples are stored in Qdrant using **two named vectors**:

- **mistake_logic** – 64-dimensional vector used for mistake category grouping
- **semantic context** – 384-dimensional embedding used for contextual similarity

To prevent storing nearly identical examples:

- embeddings are compared using cosine similarity
- new examples are stored only if similarity < **0.9**
- otherwise only an additional **occurrence** is recorded

This keeps the dataset compact while preserving usage frequency.

---

## Multi-service architecture

The system integrates several external components:

- grammar analysis
- vector storage
- LLM lesson generation

All services are orchestrated with Docker.

---

## Stability before retrieval quality

Part of the semantic retrieval pipeline was temporarily simplified while stabilizing the end-to-end system flow.

Current pipeline:

```
User text
→ LanguageTool grammar detection
→ embedding generation
→ semantic deduplication
→ vector retrieval (simplified)
→ lesson generation
```

Once the system pipeline is stable, retrieval quality improvements are planned.

---

# Known Issues / Tradeoffs

### Retrieval bug

`recently_used_explanations` currently returns an empty list due to an issue in `_retrieve_lesson_artifacts`.

Planned fix:

- switch to Named Vectors with 64-dim `mistake_logic` vector + 384-dim artifact_vector or single `mistake_logic` vector + lesson artifacts in payload 
- retrieve by `mistake_logic` vector first
- then filter by most recent examples

---

### LearningSummaryBatch (Not Integrated Yet)

LearningSummaryBatch is currently unused. It is planned to be part of the context assembly and the lesson generation logic once the taxonomy is stabilized.

---

### Local vector query limitation

The relational store holds occurrence metadata, lesson artifacts, exercise attempts, and **incremental user scoring** rows used for time-based and aggregate queries.

A typical fallback workflow (when session candidates are empty) uses aggregate scores, then Qdrant:

```
PostgreSQL (user_scoring_events, per mistake_type)
→ top mistake_type
Qdrant (mistake_examples, filter by user_id + mistake_type)
```



## Project Status

Lexory is an experimental prototype.

Current focus:

- refining mistake taxonomy
- improving retrieval quality

Future improvements include stronger fine-tuned LLM models and expanded semantic retrieval.


## Architecture

### Text ingestion flow
How user text is processed and stored.

```mermaid
flowchart TB
    A[User text /submit]
    B[ingest_user_text]
    C[LanguageTool Pipeline]
    D[Embedding generation]
    E[Create mistake event]

    F{Category exists?}
    G[Create example + occurrence]
    H[Semantic similarity check]

    I{Similarity > 0.9?}
    J[Store occurrence only]
    K[Store example + occurrence]

    L[(Qdrant mistake_examples)]
    M[(Qdrant mistake_occurrences)]
    P[(PostgreSQL: occurrences, lesson artifacts, exercise attempts, user_scoring_events)]

    A --> B
    B --> C
    C --> D
    D --> E

    E --> F
    F -- No --> G
    F -- Yes --> H

    H --> I
    I -- Yes --> J
    I -- No --> K

    G --> L
    G --> M
    G --> P

    J --> M
    J --> P

    K --> L
    K --> M
    K --> P
```

*Category exists* = user already has examples for this `mistake_type`. *Similarity > 0.9* = near-duplicate; store occurrence only. `exercise_attempt` and Qdrant-excluded `mistake_type` values (see `rag/mistake_exclusion.py`) skip new `mistake_examples` points; rows are still written to PostgreSQL and scored.

### Lesson generation flow
How stored data is used to generate lessons.

```mermaid
flowchart TB
    A[submit_and_lesson]
    B{Text provided?}
    C[ingest_user_text]
    D[Get query embedding]
    E{Embedding available?}
    F[No-context lesson]

    G[_retrieve_staged_context]
    H[_construct_lesson]
    I[_persist_lesson_artifact]

    J[Session candidates]
    K[Fallback: PostgreSQL scores then Qdrant by mistake_type]
    L[(mistake_examples)]
    M[(lesson_artifact_embeddings)]
    N[(learning_summary_embeddings)]
    O[(lesson_artifact_embeddings — persist new lesson)]

    A --> B
    B -- Yes --> C
    B -- No --> D
    C --> J
    J --> D
    K --> D
    L --> D

    D --> E
    E -- No --> F
    E -- Yes --> G

    G --> H
    M --> G
    N --> G

    H --> I
    I --> O
```

*Query embedding*: from session candidates (from ingest) or **fallback** when there are no candidates: top `mistake_type` by clamped user score in PostgreSQL (`user_scoring_events`, tie-break: latest `occurred_at`), then the **latest** `mistake_examples` point for that type (Qdrant scroll ordered by payload `detected_at` desc). *Context*: primary mistake + similar lesson artifacts + learning summaries. *Lesson*: approach handler (LLM or stub) builds explanation and exercises.

## Running with Docker

The fastest way to run Lexory for development and demonstration:

```bash
docker compose up --build
```

This starts:

- **Lexory (FastAPI)** – API at http://localhost:8000  
  - Swagger UI: http://localhost:8000/docs
- **Qdrant** – Vector database at http://localhost:6333  
  - Data persisted in `./qdrant_storage`
- **PostgreSQL** - Database at postgresql+asyncpg://lexory:lexory@localhost:5432/lexory
- **Ollama** – LLM service for lesson generation (default mode)
- **LanguageTool** – Grammar checking at http://localhost:8010 (no rate limits; allow ~30s for Java to start)

**Optional / maintainer-only files:**
- `assets/languagetool_rule_ids_en_all.json` – full list of rule ids found in upstream English rule XMLs for the LT version that last regenerated the mapping (not read at runtime).
- `assets/languagetool_rule_inventory_manifest.json` – which `en` rules tree and `--lt-ref` were used.
To **bump** the LanguageTool server version: pin the Docker image tag to the matching [LanguageTool release](https://github.com/languagetool-org/languagetool/tags), sparse-clone that tag’s `languagetool-language-modules/.../rules/en` under `vendor/lt-en-rules` (gitignored; see `.gitignore`), then run `scripts/extract_languagetool_rule_ids.py` with the same `--lt-ref` and `--bulk-fill-mapping` as needed, and commit the updated JSON assets. 

After the first start, pull a model:

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b-instruct
```

Set `GENERATOR_MODE=stub` to use deterministic stub lessons instead of the LLM.

**Troubleshooting:** If the LLM fails with connection errors, pull the model and check the **ollama** logs. If Lexory cannot reach Qdrant on startup, use the bundled `docker-compose.yml` as-is (service URLs are fixed there). For **uvicorn on the host** with backends in Docker, use `.env` / `.env.example` with `http://localhost:…` on the published ports.

### Environment variables

With **`docker compose`**, the **lexory** service receives **`QDRANT_URL`**, **`OLLAMA_URL`**, and **`LANGUAGETOOL_URL`** from `docker-compose.yml` (Docker network hostnames). **`GENERATOR_MODE`**, **`OLLAMA_MODEL`**, and **`HF_TOKEN`** still come from your environment (e.g. `.env` in the project directory). When you run the app **locally** (not in Compose), unset `QDRANT_URL` for embedded Qdrant or set URLs yourself; see `.env.example` for localhost examples.

| Variable | Lexory-in-Docker | Description |
|----------|------------------|-------------|
| `GENERATOR_MODE` | From `.env` / default `llm` | `llm` or `stub` |
| `OLLAMA_MODEL` | From `.env` / default `qwen2.5:1.5b-instruct` | Ollama model name |
| `OLLAMA_URL` | Fixed in `docker-compose.yml` | `http://ollama:11434/api/generate` (Lexory calls **`/api/chat`** for lessons; the host/port are taken from this URL) |
| `OLLAMA_STRUCTURED_OUTPUT` | From `.env` / default `1` | Set `0` if your Ollama build rejects JSON-schema **`format`** on chat |
| `OLLAMA_TIMEOUT` | From `.env` / default `120` | Chat request timeout in seconds |
| `QDRANT_URL` | Fixed in `docker-compose.yml` | `http://qdrant:6333` |
| `LANGUAGETOOL_URL` | Fixed in `docker-compose.yml` | `http://languagetool:8010` |
| `DATABASE_URL` | Set in `docker-compose.yml` for Lexory-in-Docker (`postgres:5432`) | Async SQLAlchemy URL, e.g. `postgresql+asyncpg://user:pass@host:port/db` |

### Database schema and Alembic

Relational data lives in **PostgreSQL** (see `storage/models.py`). **SQLAlchemy 2** (async) with **asyncpg** is used at runtime. **Alembic** owns schema versions: revision scripts live under `alembic/versions/`, and `alembic.ini` points at the `alembic` env in this repo.

On application startup, the service runs `alembic upgrade head` (using a **sync** driver URL derived from `DATABASE_URL` for the migration process). You can also run migrations manually from the project root, with `DATABASE_URL` set:

```bash
alembic upgrade head
```

Current tables (evolve with migrations) include, among others: **`mistake_occurrences`**, **`lesson_artifacts`**, **`exercise_attempts`**, **`user_scoring_events`**.

---

## Using the API via Swagger UI

### 1. Open Swagger UI
<img src="docs/screenshots/swagger_ui/0.png" width="600"/>

Click “Try it out”.


### 2. Send a request
<img src="docs/screenshots/swagger_ui/1.png" width="600"/>

Fill out fields `text` and `user_id`, then click “Execute”.


### 3. Get the response
<img src="docs/screenshots/swagger_ui/2.png" width="600"/>

The LLM response is returned in the properties `topic`, `explanation`, and `exercise`.

---

## Third-Party Software

This application uses the following third-party components:

### Core components

- Qdrant Server (Apache 2.0)
- Ollama
- Qwen2.5 1.5B Instruct (Ollama: `qwen2.5:1.5b-instruct`)
- PostgreSQL (PostgreSQL License)
- Docker (Apache 2.0)

### Python libraries

- FastAPI (MIT)
- Pydantic (MIT)
- language-tool-python (LGPL 2.1+)
- SQLAlchemy (MIT)
- Alembic (MIT) — database migrations
- asyncpg (Apache 2.0) — async PostgreSQL driver
- psycopg (GNU LGPL) — sync driver for Alembic
- sentence-transformers (Apache 2.0)
- PyTorch (BSD-style)
- Uvicorn (BSD)
- NumPy (BSD)
- requests (Apache)
- python-dotenv (BSD)
- Polars (Apache 2.0)
- Qdrant Client (Apache 2.0)

See THIRD_PARTY_LICENSES.txt for full license information.

---

## License

This project is licensed under the GPL-3.0 License — see the LICENSE file for details.

This repository currently serves as a personal research and portfolio project.
If you are interested in commercial use or collaboration, feel free to contact the author.

---

## Author

© 2026 Aliona Sîrf