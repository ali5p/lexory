FROM python:3.11-slim

WORKDIR /app

# install uv
RUN pip install --no-cache-dir uv

# copy dependency metadata
COPY pyproject.toml uv.lock ./

# Lock file uses CPU-only torch (pyproject.toml tool.uv.sources).
ENV UV_HTTP_TIMEOUT=300
RUN uv sync --no-dev

# copy project source
COPY . .

EXPOSE 8000

# No --reload: bind mount .:/app includes qdrant_storage/; StatReload races with Qdrant temp files (FileNotFoundError).
# For local dev with reload: run uvicorn on the host or override CMD.
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]