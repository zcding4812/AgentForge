# AgentForge Backend

## Quick Start

1. Install dependencies:

```bash
uv sync
```

2. Start development server:

```bash
uv run fastapi dev --host 0.0.0.0 --port 8000
```

3. Verify service:

- Health check: `GET /health`
- OpenAPI: `GET /openapi.json`
- Swagger: `GET /docs`

## Database Migration (Alembic)

1. Generate migration (auto-diff models):

```bash
uv run alembic revision --autogenerate -m "init schema"
```

2. Apply migrations:

```bash
uv run alembic upgrade head
```

3. Roll back one version (optional):

```bash
uv run alembic downgrade -1
```

## Project Structure

- `app/main.py`: FastAPI application entrypoint
- `app/config.py`: runtime settings
- `app/core/`: middleware, exception handlers, lifespan, deps
- `app/api/`: route modules
- `app/schemas/`: request/response schemas
- `tests/`: test cases
