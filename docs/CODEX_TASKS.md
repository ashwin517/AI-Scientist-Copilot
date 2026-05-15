# Codex Tasks

## Phase 6E Main Chat Document RAG

Implemented:

- Added `answer_document_question` to the approved agent tool registry.
- Added document-style intent detection for uploaded papers, PDFs, documents,
  and SOP questions.
- Added an agent tool wrapper that reuses the existing RAG service.
- Returned structured chat metadata with `tool_name`,
  `answer_document_question`, the grounded answer, and sources.
- Added simple frontend source cards below RAG chat answers.
- Added clean no-document and no-relevant-information failure messages.
- Added tests for RAG intent detection, no-document handling, source-bearing
  tool results, and registry coverage.

Manual checks:

- Upload and process a document with embeddings.
- Ask the main chat: `What does the uploaded paper say about yield?`
- Confirm the assistant answers from retrieved chunks and shows sources.
- Confirm dataset/model prompts such as `summarize my dataset` and
  `train a model using yield` still route to their existing tools.

Known follow-ups:

- Add broader document-intent examples and regression tests.
- Improve source display once the document Q&A UI becomes more prominent.

## Phase 6D RAG Document Q&A Endpoint

Implemented:

- Added `apps/api/app/ai/rag_service.py`.
- Added `answer_document_question(db, project_id, question, top_k=5)`.
- Reused the Phase 6C retrieval service to gather project-scoped sources.
- Built a grounded prompt that includes retrieved chunks and citation rules.
- Called the configurable local Ollama chat model for answer generation.
- Added `POST /projects/{project_id}/documents/ask`.
- Returned a lightweight answer plus source metadata without full documents or
  embeddings.
- Added tests for empty retrieval, returned sources, and prompt construction.
- Kept the endpoint separate from the main `/chat` route and the agent.

Manual checks:

- Ensure a document has processed chunks with embeddings.
- POST to `/projects/{project_id}/documents/ask` with a question and `top_k`.
- Confirm answers cite chunks using `[doc:filename chunk:X]` when supported by
  retrieved chunks.
- Confirm main chat behavior is unchanged.

Known follow-ups:

- Add a UI surface for document Q&A.
- Decide whether and how to merge document Q&A into the main chat/agent flow.

## Phase 6C RAG Retrieval Service

Implemented:

- Added `apps/api/app/ai/retrieval_service.py`.
- Added `retrieve_relevant_chunks(db, project_id, query, top_k=5)` for
  project-scoped retrieval.
- Embedded search queries through the local Ollama embedding client.
- Implemented simple Python cosine similarity over JSON-stored embeddings.
- Added `POST /projects/{project_id}/documents/search`.
- Returned lightweight search results with document id, filename, chunk id,
  chunk index, score, and bounded content preview.
- Kept full documents and embedding vectors out of the response.
- Added tests for cosine similarity, ranking, project scoping, and empty
  project behavior.

Manual checks:

- Ensure a document has processed chunks with embeddings.
- POST to `/projects/{project_id}/documents/search` with a query and `top_k`.
- Confirm results only include chunks from the requested project.
- Confirm chat behavior is unchanged and does not use retrieved chunks yet.

Known follow-ups:

- Integrate retrieval into chat with explicit source handling in a later phase.
- Replace JSON embedding comparison with pgvector-backed search when available.

## Phase 6B RAG Foundation

Implemented:

- Added a project-scoped `DocumentChunk` SQLAlchemy model linked to
  `document_id` and `project_id`.
- Added `apps/api/app/ai/embedding_client.py` for local Ollama embeddings with
  configurable URL and model settings.
- Added document chunking and processing services with 1,000-character chunks
  and 150-character overlap.
- Extended document upload so extracted text is chunked and embedded
  automatically after the `Document` record is created.
- Added `POST /projects/{project_id}/documents/{document_id}/process` for
  manual reprocessing.
- Stored embeddings as JSON in `document_chunks.embedding_json` because pgvector
  is not yet wired into the project.
- Preserved chunks when Ollama embedding generation fails; failed chunks keep a
  null embedding and processing reports error counts.
- Added tests for chunk creation/linkage and graceful embedding failure.

Manual checks:

- Upload a TXT or Markdown document and confirm chunks are created for the
  document.
- Stop Ollama, upload a text document, and confirm chunks persist with null
  embeddings rather than failing the upload.
- Confirm chat still does not answer from uploaded documents.

Known follow-ups:

- Add pgvector-backed embedding storage and similarity search.
- Add retrieval and citation design before enabling document Q&A.

## Phase 6A RAG Foundation

Implemented:

- Added a project-scoped `Document` SQLAlchemy model.
- Added document schemas, service, and API routes for upload/list/detail.
- Saved uploaded PDF, TXT, and Markdown files to project-scoped local storage.
- Added best-effort text extraction to sidecar text files for plain text,
  Markdown, and PDFs when a PDF reader is available.
- Added a simple document upload form and document list to the existing
  workspace UI.
- Kept documents out of agent answering, embeddings, vector search, pgvector,
  and LangChain.
- Added backend route tests for document upload/list/detail.

Manual checks:

- Upload a TXT or Markdown document from a project workspace and confirm it
  appears in Files.
- Upload a PDF and confirm the document record is created even if text
  extraction is unavailable.
- Confirm chat responses do not cite or answer from uploaded documents.

Known follow-ups:

- Add document deletion and file cleanup.
- Choose the future RAG extraction/chunking/embedding pipeline in a later phase.

## Phase 5A Agent Implementation

Implemented:

- Added `apps/api/app/agent` with service, parser, registry, executor, schemas,
  prompts, and approved tool modules.
- Updated `/chat` to delegate to `AgentService`.
- Added `project_id` to chat requests so tools can operate within the active
  project.
- Updated project CSV upload so datasets are persisted immediately on upload.
- Reorganized dataset UX into workbench tabs: Upload, Files, and Model.
- Added persistent model runs shared by REST model training and agent training.
- Reused existing dataset persistence and baseline ML training logic.
- Added unit tests for parser mappings, target extraction, registry contents, and
  executor rejection of unknown tools.
- Added a route test that verifies project CSV upload creates a dataset record.

Manual checks:

- Ask: `list datasets in this project`.
- Upload a CSV and confirm it appears in the Files tab without an extra save
  click.
- Ask: `summarize my dataset`.
- Ask: `show missing values`.
- Ask: `train a model using yield`.
- Ask: `show previous model runs`.
- Confirm the Model tab lists the saved run after REST or agent-triggered
  training.
- Confirm dataset summary, missing value, and model training chat replies show a
  compact tool card when returned by `/chat`.

Known follow-ups:

- Add integration tests with a temporary database.
- Broaden parser coverage without allowing arbitrary tool names.
- Decide whether `/chat` should own chat message persistence end to end, replacing
  the current frontend-mediated persistence flow.

## Phase 5B Pending Actions

Implemented:

- Added `AgentPendingAction` persistence for one active pending action per
  project.
- Added pending action helpers and target-column slot filling.
- Updated `AgentService` to check pending actions before normal intent parsing.
- `train a model` now asks for a target column when one is missing and stores
  the pending training action.
- Follow-up replies like `yield` fill the missing target column, execute the
  training tool, persist the model run, and clear the pending action.
- Invalid target-column replies keep the pending action active.
- `cancel`, `stop`, `never mind`, and `nevermind` clear the pending action.

Manual checks:

- Ask: `train a model`.
- Reply with a valid target column, such as `yield`.
- Confirm a model run is saved.
- Repeat `train a model`, reply with an invalid column, and confirm the agent
  asks again.
- Reply `cancel` and confirm the pending action is cleared.

## Phase 5C Structured Tool Results

Implemented:

- Extended `/chat` responses with optional `message`, `tool_used`, `tool_name`,
  `tool_result`, and `pending_action` fields while preserving `reply`.
- Added backend-generated structured results for dataset summaries, missing
  values, model training, and generic approved tools.
- Added pending action metadata to clarification responses.
- Updated the frontend chat types to accept structured tool metadata.
- Added compact chat cards for dataset summaries, missing values, and model
  training results.
- Added backend tests for structured dataset summary, model training, and
  pending-action response metadata.

Manual checks:

- Ask: `summarize my dataset` and confirm a dataset card appears.
- Ask: `show missing values` and confirm a missing-values card appears.
- Ask: `train a model using yield` and confirm a model-run card appears.
- Ask: `train a model` and confirm the response includes a pending target-column
  clarification.
