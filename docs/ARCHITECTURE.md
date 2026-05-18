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
7. The response keeps the existing `{ "reply": "..." }` shape and also includes
   a compatible `message` alias.
8. When a tool runs, the response includes optional structured metadata:
   `tool_used`, `tool_name`, `tool_result`, and `pending_action`.
9. Phase 11A adds an optional deterministic planner before single-intent
   parsing for supported project-review requests. When a plan is produced, the
   agent executes 3-8 approved registry tools sequentially and synthesizes one
   response from their summaries.

Approved tools:

- `list_datasets`
- `get_dataset_summary`
- `show_missing_values`
- `train_baseline_model`
- `list_model_runs`
- `answer_document_question`

Safety boundaries:

- The LLM does not choose arbitrary code or database queries.
- Tools are allow-listed in `ToolRegistry`.
- Tool execution requires a valid project scope.
- Baseline model training reuses the existing `model_training` service.
- Dataset tools read the existing persisted dataset JSON.

Multi-step planning:

- `apps/api/app/agent/planner.py` defines `PlanStep` with `step_id`,
  `tool_name`, `arguments_json`, `purpose`, `status`, and `result_summary`.
- `create_execution_plan(user_query, project_context)` is deterministic and
  rule-based. There is no LLM planner, LangChain planner, or LangGraph.
- Initial supported plan intents cover project review and next-step requests:
  `analyze my project`, `suggest next experiments`, `review project status`,
  and `what should I do next`.
- Plans use only tools present in `ToolRegistry`; `ToolExecutor` still performs
  the final allow-list check.
- The initial review plan inspects memory, latest dataset, uploaded document
  context, latest model, latest simulation, latest optimization, and optionally
  existing optimization-based experiment recommendations.
- The planner does not train models, run simulations, or run optimization. It
  only calls existing inspection/recommendation tools.
- Chat responses for planned executions include optional `plan_executed`,
  `steps_summary`, and `tools_used` metadata. The frontend can display this as a
  compact trace card.

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

Document upload flow:

- Project document uploads use `POST /projects/{project_id}/documents/upload`.
- The backend validates PDF, TXT, and Markdown files, saves the original file
  under project-scoped local storage, and creates a `Document` record.
- TXT and Markdown text is extracted into a sidecar text file. PDF extraction is
  best-effort when a lightweight PDF reader is available; failures do not block
  upload persistence.
- Uploaded documents are listed through `GET /projects/{project_id}/documents`
  and can be fetched by project-scoped document id.
- Documents are visible in the workspace Files tab, but the agent does not use
  document text for answering yet.

Document chunking and embedding flow:

- Phase 6B adds project-scoped `DocumentChunk` records linked to both
  `document_id` and `project_id`.
- The upload route processes extracted text automatically after creating the
  `Document` record. A manual reprocessing route is also available at
  `POST /projects/{project_id}/documents/{document_id}/process`.
- Extracted text is split into roughly 1,000-character chunks with 150
  characters of overlap.
- Embeddings are generated through local Ollama using the configurable
  `OLLAMA_EMBEDDING_MODEL` and `OLLAMA_EMBEDDING_URL` settings.
- pgvector is not wired into this project yet, so chunk embeddings are stored as
  JSON in `document_chunks.embedding_json` as a temporary foundation.
- If Ollama is unavailable or returns an invalid embedding, the chunk is still
  persisted with a null embedding and processing reports the embedding error
  count.
- The agent and chat path still do not retrieve from, cite, or answer using
  document chunks.

Document retrieval flow:

- Phase 6C adds a backend-only retrieval service in
  `apps/api/app/ai/retrieval_service.py`.
- Project document search uses
  `POST /projects/{project_id}/documents/search` with a query and `top_k`.
- The service embeds the query with the same configurable local Ollama embedding
  client used during ingestion.
- Because embeddings are still stored as JSON, retrieval computes cosine
  similarity in Python and returns the highest-scoring project chunks.
- Search responses include document id, filename, chunk id, chunk index, score,
  and a bounded content preview. Full documents and embedding vectors are not
  returned.
- Retrieval is project-scoped and ignores chunks from other projects.
- This is not connected to chat or the agent yet.

Document Q&A flow:

- Phase 6D adds `apps/api/app/ai/rag_service.py` for standalone document Q&A.
- Project document questions use
  `POST /projects/{project_id}/documents/ask` with a question and `top_k`.
- The RAG service retrieves relevant chunks through the Phase 6C retrieval
  service, builds a grounded prompt, and calls the configurable local Ollama
  chat model.
- Prompt rules require the model to answer only from provided chunks, admit when
  the chunks are insufficient, avoid invented citations, and cite sources as
  `[doc:filename chunk:X]`.
- Responses include the generated answer and lightweight source metadata using
  the same bounded content previews as retrieval.
- Phase 6E also exposes this capability to the main project chat agent through
  the approved `answer_document_question` tool.
- The agent detects document-style questions, calls the existing RAG service,
  returns structured tool metadata with answer and sources, and saves the chat
  exchange through the existing chat history path.

Model run flow:

- Baseline training logic remains in the reusable `model_training` service.
- Persistent project training uses the same training service and writes a
  `model_runs` database record with dataset, target, task type, metrics, and
  feature importance.
- The REST training endpoint and the agent `train_baseline_model` tool both use
  the shared persistent training path when a project dataset is supplied.
- The workspace Model tab lists saved model runs from
  `GET /models/projects/{project_id}/model-runs`.
- Phase 7E adds the approved read-only `explain_latest_model` agent tool.
- Model explanation intents such as "explain latest model", "important
  features", "feature importance", "why is the model performing this way?", and
  "what should I try next?" load the latest saved model run from
  `latest_model_run_id` memory or the latest persisted model run fallback.
- The explanation returns saved metrics, target column, dataset metadata, top
  feature importances, limitations, and suggested next steps. It does not
  retrain models and explicitly warns that feature importance indicates
  predictive association, not necessarily causation.

Simulation and optimization flow:

- The batch reactor simulator models a simple educational A -> B -> C batch
  reactor benchmark. It is not calibrated chemistry and is not a plant-valid
  operating model.
- Simulation runs are persisted as `simulation_runs` and can be triggered by
  the REST endpoint or the approved `run_batch_reactor_simulation` agent tool.
- Optimization runs are persisted as `optimization_runs` with objective,
  constraints, search space, result JSON, and timestamps.
- `POST /projects/{project_id}/optimization/batch-reactor` and the
  `optimize_batch_reactor` tool run a transparent grid search over temperature,
  batch time, initial concentration, and catalyst factor.
- The objective is `final_yield - penalty_weight * final_impurity`, with a
  default impurity constraint of `final_impurity <= 0.15`.
- Phase 9B adds read-only optimization workflow tools:
  `explain_latest_optimization`, `list_optimization_runs`, and
  `recommend_next_experiment`.
- Optimization explanation loads the latest remembered optimization run when
  available, summarizes the objective and constraint, reports best inputs and
  predicted yield/impurity/conversion, and explains the yield/impurity tradeoff.
- Recommendation uses the top candidates from the latest optimization result,
  recommends one to three simulated follow-up experiments, records
  `latest_recommended_experiment` and `recommended_experiment_count` in project
  memory, and explicitly labels them as simulated recommendations rather than
  validated plant instructions.
- The "compare optimization with latest simulation" intent explains the latest
  optimization and includes a simple comparison against the latest saved
  simulation when one exists.

Project analysis workflow:

- Phase 10A adds a first synchronous workflow foundation in
  `apps/api/app/workflows`.
- Workflow runs are persisted in `workflow_runs` with project id, workflow type,
  status, step JSON, result JSON, creation time, and completion time.
- `POST /projects/{project_id}/workflows/project-analysis` runs the project
  analysis workflow immediately. There are no background jobs and no LangGraph.
- The approved `run_project_analysis_workflow` agent tool inspects existing
  project memory, the latest dataset, documents, latest model run, latest
  simulation, and latest optimization. It does not launch training, simulation,
  or optimization.
- The result returns concise project status, current asset availability, gaps,
  and three recommended next actions. It records `latest_workflow_run_id` and
  `latest_project_analysis_summary` in structured project memory.
- Phase 10B adds read-only workflow history tools:
  `list_workflow_runs`, `explain_latest_workflow`, and
  `compare_workflow_runs`.
- Workflow history requests list recent saved runs. Latest-workflow explanation
  uses `latest_workflow_run_id` memory when available and falls back to the most
  recent persisted run. Workflow comparison compares the latest two runs by
  status, asset availability changes, identified gaps, and recommendations.

Report generation:

- Phase 10C adds a deterministic report module in `apps/api/app/reports`.
- Reports are persisted in `reports` with project id, report type, title,
  markdown content, source summary JSON, and creation time.
- `POST /projects/{project_id}/reports/generate` creates a markdown project
  report from saved workspace state. `GET /projects/{project_id}/reports` lists
  generated reports, and `GET /projects/{project_id}/reports/{report_id}` loads
  a specific report.
- The approved `generate_project_report` agent tool creates the same persisted
  markdown report from project memory, datasets, documents, latest model run,
  latest simulation, latest optimization, and latest workflow run.
- Report sections are fixed and inspectable: Project Overview, Available Data,
  Uploaded Documents, Model Results, Simulation Results, Optimization Results,
  Workflow Recommendations, Limitations, and Recommended Next Steps.
- Report generation records `latest_report_id` and `latest_report_title` in
  structured project memory.
- There is no PDF export, DOCX export, email/share flow, background job, or LLM
  report prose in this phase.
- Phase 10D adds read-only report review tools:
  `list_reports`, `explain_latest_report`, and `review_latest_report`.
- Latest-report tools load `latest_report_id` from project memory when possible,
  then fall back to the newest saved report. Review inspects saved markdown
  sections, reports strengths, missing sections, weak sections, suggested edits,
  and review limitations. It does not overwrite or edit report content.

Project memory flow:

- Phase 7A adds structured project-scoped memory in the `project_memory` table.
- Memory records are keyed per project and store `memory_type`, `key`,
  JSON-serialized value, source, and created/updated timestamps.
- The service in `apps/api/app/services/memory_service.py` exposes
  `upsert_memory`, `get_memory`, `list_memory`, and `delete_memory`.
- The backend automatically records `latest_dataset_id` on dataset creation,
  `latest_dataset_filename`, and `dataset_count` on dataset creation;
  `latest_document_id`, `latest_document_filename`, and `document_count` on
  document upload; `latest_model_run_id`, `selected_target_column`, and
  `latest_task_type` after persistent model training; and
  `selected_target_column` when the user confirms a pending training target.
- Optimization tools also record `latest_optimization_run_id`,
  `latest_optimization_type`, `latest_recommended_experiment`, and
  `recommended_experiment_count` when those workflows run.
- The agent loads recent project memory before prompt construction and includes
  a concise bounded memory JSON block. Large values and long memory lists are
  truncated before they reach the prompt.
- Project memory is available through
  `GET /projects/{project_id}/memory` and removable by key through
  `DELETE /projects/{project_id}/memory/{key}`.
- This is structured memory only: no vector memory, semantic long-term memory,
  LangChain, simulation, or optimization is introduced in this phase.
- Phase 7B makes approved tools memory-aware while keeping explicit user input
  authoritative.
- `AgentService` loads project memory once per request, passes decoded memory to
  the intent parser and tool executor, and passes bounded memory context to LLM
  prompt builders.
- Dataset tools resolve datasets in this order: explicit `dataset_id`,
  remembered `latest_dataset_id`, then latest uploaded dataset.
- Baseline training resolves the target column in this order: explicit
  `target_column`, remembered `selected_target_column`, then the existing
  clarification flow.
- Model explanation requests such as "explain the model" use the remembered
  `latest_model_run_id` when available.
- Tool responses include concise notes when memory assisted execution, for
  example "I used your remembered target column: yield_pct." or "I used your
  latest uploaded dataset: batch_process_yield_test.csv."
- Phase 7C adds user-controlled memory editing through approved agent tools:
  `list_project_memory`, `upsert_project_memory`, and
  `delete_project_memory`.
- Memory transparency requests such as "what do you remember about this
  project?" list project-local database memory in a readable project summary,
  including latest document, document count, latest dataset, dataset count,
  latest model run, task type, and selected target when available.
- Before listing memory, the backend syncs memory from persisted project
  records so documents, datasets, and model runs created in earlier sessions
  are reflected even if they predate the automatic memory hooks.
- Phase 7D adds the `project_summary` memory key. The summary is regenerated
  from project records and important project facts after document upload or
  processing, dataset upload, target-column selection, model training, and
  manual memory edits.
- `project_summary` is concise and includes project name/description when
  useful, uploaded document count and latest document, dataset count and latest
  dataset, selected target column, latest model run, and user-added project
  facts. It does not include document chunks, dataset previews, vector memory,
  or semantic memory.
- Agent prompt memory context exposes `project_summary` directly so the agent
  can use a compact project-level summary without loading every memory item as
  separate prompt context.
- User instructions such as "remember that this project is about batch reactor
  yield optimization" write project notes, while "forget the target column"
  deletes `selected_target_column`.
- Target-setting instructions such as "use yield_pct as the target column from
  now on" validate the column against the active/latest project dataset first,
  then search saved project datasets. If exactly one saved dataset contains the
  target column, the backend accepts it and remembers that dataset as the
  modeling dataset.
- Ambiguous forget requests ask for clarification instead of deleting broad or
  global memory.

Development ID reset behavior:

- Delete paths reset primary-key sequences only for tables that are empty after
  the delete finishes.
- This keeps local backend testing predictable after clearing projects or data,
  while avoiding ID collisions when any rows still exist.
- PostgreSQL sequences are reset with `setval(..., 1, false)`. SQLite
  `sqlite_sequence` rows are removed when that metadata table exists.

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

Structured chat results:

- Phase 5C keeps assistant text as the primary chat artifact while returning
  backend-generated metadata for approved tool calls.
- Dataset summary responses include dataset id, filename, row count, column
  names, numeric columns, categorical columns, missing values, and preview rows.
- Missing-value responses include the dataset metadata, per-column missing
  counts, columns with missing values, and total missing values.
- Model training responses include model run id, dataset id, target column, task
  type, metrics, and top features.
- Simulation and optimization responses include run ids, operating inputs,
  objective/result metrics, top candidates, and recommendation candidates when
  relevant.
- Project analysis responses include workflow run id, inspected steps, current
  asset availability, gaps, and recommended next actions.
- Workflow history responses include recent workflow summaries, latest workflow
  recommendations, or latest-vs-previous analysis comparisons.
- Report-generation responses include report id, title, markdown content, source
  summary, and recommended next steps.
- Report-review responses include strengths, missing sections, weak sections,
  suggested edits, and limitations grounded in the saved report markdown.
- The frontend renders small chat cards for dataset summaries, missing values,
  model training results, simulation results, optimization results,
  optimization recommendations, project analysis workflow results, and workflow
  history/explanation/comparison results, generated reports, and report review
  results. The Reports tab lists generated reports and renders their saved
  markdown. Workspace Files, Model, and Reports tabs remain the source of truth
  for saved objects.

Current limitations:

- Intent parsing is rule-based and intentionally narrow.
- Pending actions only support one active action per project and currently cover
  target-column clarification for baseline model training.
- Structured tool metadata is returned on the live `/chat` response; historical
  chat messages still store plain assistant text.
- The agent does not implement authentication or paid API integrations.
- Phase 6B persists document chunks and best-effort embeddings for future RAG.
  Phase 6C adds backend retrieval over those JSON embeddings. Phase 6D adds a
  standalone document Q&A endpoint. Phase 6E integrates document Q&A into the
  main project chat agent. Phase 7A adds structured project memory, Phase 7B
  makes tools memory-aware, and Phase 7C adds user-visible memory editing.
  There is still no LangChain, pgvector retrieval, vector memory, semantic
  long-term memory, Bayesian optimization, historian connector, LangGraph,
  report editing workflow, PDF/DOCX export, background workflow execution, or
  autonomous experiment execution.
