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
- The frontend renders small chat cards for dataset summaries, missing values,
  and model training results. Workspace Files and Model tabs remain the source
  of truth for saved objects.

Current limitations:

- Intent parsing is rule-based and intentionally narrow.
- Pending actions only support one active action per project and currently cover
  target-column clarification for baseline model training.
- Structured tool metadata is returned on the live `/chat` response; historical
  chat messages still store plain assistant text.
- The agent does not implement RAG, simulation, optimization, authentication, or
  paid API integrations.
- Phase 6B persists document chunks and best-effort embeddings for future RAG.
  Phase 6C adds backend retrieval over those JSON embeddings. Phase 6D adds a
  standalone document Q&A endpoint. Phase 6E integrates document Q&A into the
  main project chat agent. There is still no LangChain or pgvector retrieval.
