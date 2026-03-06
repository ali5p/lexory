FROM python:3.11-slim

WORKDIR /app

# install uv
RUN pip install --no-cache-dir uv

# copy dependency metadata first for layer caching
COPY pyproject.toml uv.lock ./

# install dependencies
RUN uv sync --no-dev

# copy project source
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]