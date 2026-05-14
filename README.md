# AI Scientist Copilot v2

Industrial-grade AI-first scientific assistant for data analysis, modeling, simulation, optimization, and knowledge retrieval.

## Stack

- Frontend: Next.js, TypeScript, TailwindCSS
- Backend: FastAPI, Python
- Database: PostgreSQL
- Infra: Docker Compose
- Local AI: Ollama with `llama3.2:3b` planned for later phases

## Project Structure

```text
apps/
  api/
    app/
      db/
      routes/
      schemas/
      services/
    requirements.txt
  web/
    app/
docker-compose.yml
docs/
```

## Setup

### 1. Start PostgreSQL

```bash
docker compose up -d
```

### 2. Configure the API

```bash
cd apps/api
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The API health check will be available at:

```text
http://localhost:8000/health
```

Database tables are created automatically when the API starts. Make sure PostgreSQL is running before starting `uvicorn`.

If the API fails on startup with a PostgreSQL password or missing-role error after changing Compose credentials, recreate the local database volume:

```powershell
docker compose down -v
docker compose up -d
```

This deletes local development database data for this project.

Workspace persistence endpoints:

```text
POST http://localhost:8000/projects
GET  http://localhost:8000/projects
GET  http://localhost:8000/projects/{project_id}
```

Example project creation:

```bash
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name":"Demo Project","description":"Initial workspace"}'
```

### 3. Run the frontend

```bash
cd apps/web
npm install
npm run dev
```

The web app will be available at:

```text
http://localhost:3000
```

## Current Phase

Phase 2 workspace persistence:

- FastAPI app scaffold
- Next.js app scaffold
- PostgreSQL Docker Compose service
- Clean backend module layout
- Health endpoint
- SQLAlchemy `Project` and `Dataset` models
- Automatic database initialization on API startup
- Project create/list/detail API routes

No AI chat, dataset upload, ML workflows, or simulation features are implemented yet.
