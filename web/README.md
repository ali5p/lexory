# Lexory web (React + Vite)

Web UI for the Lexory API.

## Dev setup

1. Start the API (Docker or local uvicorn on port 8000).
2. Install and run the dev server:

```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173. Requests go to `/api/*`, proxied to `http://localhost:8000/*`.

## Production build

```bash
npm run build
```

Output: `web/dist/` (deploy this folder on Cloudflare Pages).

Set `VITE_API_BASE_URL` to your public API origin, for example:

```bash
VITE_API_BASE_URL=https://api.example.com npm run build
```

Ensure the API allows your Pages origin via `LEXORY_CORS_ORIGINS`.

## API contract

- `POST /submit` — text + user_id → lessons and exercises
- `POST /exercises/{id}/answer` — grade MCQ or fill-blank (no client-side answer keys)

Types live in `src/api/types.ts` and should stay aligned with FastAPI OpenAPI.
