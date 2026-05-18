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
GET  http://localhost:8000/projects/{project_id}/memory
DELETE http://localhost:8000/projects/{project_id}/memory/{key}
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

Phase 7C memory transparency and user-controlled memory editing:

- Project-scoped `ProjectMemory` records store structured JSON facts by key.
- Dataset uploads update `latest_dataset_id`.
- Dataset uploads also update `latest_dataset_filename` and `dataset_count`.
- Document uploads update `latest_document_id`, `latest_document_filename`, and
  `document_count`.
- Persistent model training updates `latest_model_run_id`,
  `selected_target_column`, and `latest_task_type`.
- Target-column confirmation updates `selected_target_column`.
- The chat agent includes bounded project memory context in LLM prompts.
- Dataset tools use remembered `latest_dataset_id` when no explicit dataset is
  requested.
- Baseline model training uses remembered `selected_target_column` when the user
  says "train a model" without naming a target.
- Model explanation requests use remembered `latest_model_run_id` when
  available.
- Explicit user input always overrides remembered memory.
- Memory can be listed or deleted through project-scoped API routes.
- Users can ask what the copilot remembers about the current project.
- Users can add project-local notes with messages like `remember that ...`.
- Users can delete remembered items with messages like `forget the target column`.
- Users can set the remembered target column with messages like
  `use yield_pct as the target column from now on`; the backend validates the
  column against the active/latest dataset before saving it.
- The backend maintains a concise `project_summary` memory item for prompt
  context, refreshed after document/dataset/model/memory changes.
- The copilot can explain the latest saved model from workspace memory,
  including metrics, top features, limitations, and next steps without
  retraining or claiming causality.
- Deleting projects or data resets empty-table ID sequences for easier local
  backend debugging; IDs are not reset while rows remain.

No vector memory, semantic long-term memory, LangChain, simulation, or
optimization is included in this phase.
