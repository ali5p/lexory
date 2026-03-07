# Lexory

Minimal RAG-based language learning backend. Ingests user text, detects grammar mistakes via LanguageTool, stores examples in a vector database, and generates personalized lessons.

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
- **Ollama** – LLM service for lesson generation (default mode)

After the first start, pull a model:

```bash
docker compose exec ollama ollama pull qwen2:1.5b
```

Set `GENERATOR_MODE=stub` to use deterministic stub lessons instead of the LLM.

### Environment variables

| Variable        | Default                          | Description                    |
|----------------|----------------------------------|--------------------------------|
| `GENERATOR_MODE` | `llm`                           | `llm` (default) or `stub`      |
| `OLLAMA_MODEL` | `qwen2:1.5b`                     | Ollama model name              |
| `OLLAMA_URL` | `http://ollama:11434/api/generate` | Ollama API URL                 |
| `QDRANT_URL` | `http://qdrant:6333`             | Qdrant URL (used when set)     |
| `HF_TOKEN`   | *(optional)*                     | Hugging Face token for higher rate limits |
