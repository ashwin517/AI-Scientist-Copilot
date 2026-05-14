# Architecture

## Phase 5A Tool-Calling Agent

The copilot chat path now routes through a small backend agent layer in
`apps/api/app/agent`.

Request flow:

1. The frontend sends `/chat` a user message plus the active `project_id`.
2. `app.routes.chat` calls `AgentService` and stays thin.
3. `IntentParser` applies deterministic rules to decide whether an approved tool
   is needed.
4. `ToolExecutor` validates the tool name and required `project_id`.
5. Approved tools call existing backend services and models.
6. `AgentService` asks Ollama to summarize tool output when available, with a
   deterministic fallback when Ollama is unavailable.
7. The response keeps the existing `{ "reply": "..." }` shape.

Approved tools:

- `list_datasets`
- `get_dataset_summary`
- `show_missing_values`
- `train_baseline_model`
- `list_model_runs`

Safety boundaries:

- The LLM does not choose arbitrary code or database queries.
- Tools are allow-listed in `ToolRegistry`.
- Tool execution requires a valid project scope.
- Baseline model training reuses the existing `model_training` service.
- Dataset tools read the existing persisted dataset JSON.

Dataset upload flow:

- Project CSV uploads use `POST /projects/{project_id}/datasets/upload`.
- The backend parses the CSV, computes preview/profile metadata, and immediately
  creates the project `Dataset` record.
- Uploaded datasets are immediately available to the approved dataset and model
  agent tools.
- The frontend exposes uploaded project datasets through a `Files` workbench tab
  instead of a separate manual save step.
- The older `/datasets/upload-preview` route remains available for preview-only
  flows, but the primary workspace flow persists on upload.

Model run flow:

- Baseline training logic remains in the reusable `model_training` service.
- Persistent project training uses the same training service and writes a
  `model_runs` database record with dataset, target, task type, metrics, and
  feature importance.
- The REST training endpoint and the agent `train_baseline_model` tool both use
  the shared persistent training path when a project dataset is supplied.
- The workspace Model tab lists saved model runs from
  `GET /models/projects/{project_id}/model-runs`.

Pending action flow:

- The agent checks for a project-scoped pending action before normal intent
  parsing.
- Phase 5B supports one active pending action per project.
- If the user asks to train a model without a target column, the agent resolves
  the latest saved dataset, stores a pending `train_baseline_model` action, and
  asks which target column to use.
- A follow-up message fills `target_column` by exact column match first, then
  case-insensitive match.
- Invalid target-column replies keep the pending action active and list available
  columns again.
- `cancel`, `stop`, `never mind`, or `nevermind` clears the pending action.
- Successful execution clears the pending action and persists the model run.

Current limitations:

- Intent parsing is rule-based and intentionally narrow.
- Pending actions only support one active action per project and currently cover
  target-column clarification for baseline model training.
- The agent does not implement RAG, simulation, optimization, authentication, or
  paid API integrations.
