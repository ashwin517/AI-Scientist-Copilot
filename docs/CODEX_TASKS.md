# Codex Tasks

## Phase 11A Multi-Step Agent Planning Foundation

Implemented:

- Added `apps/api/app/agent/planner.py` with an inspectable `PlanStep` schema:
  `step_id`, `tool_name`, `arguments_json`, `purpose`, `status`, and
  `result_summary`.
- Added deterministic `create_execution_plan(user_query, project_context)` for
  supported project-review intents.
- Supported initial multi-step phrases such as `analyze my project`,
  `suggest next experiments`, `review project status`, and
  `what should I do next`.
- Plans execute sequentially through the existing `ToolRegistry` and
  `ToolExecutor`; unregistered tools are filtered out and executor validation
  still applies.
- The initial plan inspects project memory, latest dataset, uploaded document
  context, latest model, latest simulation, latest optimization, and optional
  next-experiment recommendations.
- Kept the workflow bounded to 3-8 steps and read/recommend oriented: no model
  training, no new simulation run, and no new optimization run is launched by
  the planner.
- `/chat` responses may now include `plan_executed`, `steps_summary`, and
  `tools_used`, plus a compact `execution_plan` structured tool result for the
  frontend trace card.
- Added a small frontend plan trace card without changing the workspace layout.

Manual checks:

- Ask: `Analyze my project and suggest next experiments`.
- Ask: `review project status`.
- Confirm the response includes a plan trace, uses only registered tools, and
  does not create model, simulation, or optimization runs.

## Phase 10D Report Review And Improvement Agent

Implemented:

- Added read-only report tools:
  `review_latest_report`, `list_reports`, and `explain_latest_report`.
- Registered the new tools in `ToolRegistry`.
- Routed report review, latest report, missing-section, improvement, report
  explanation, and limitations-check phrases through the rule-based intent
  parser.
- `review_latest_report` loads `latest_report_id` from project memory when
  available, with newest-report fallback.
- Report review inspects saved markdown sections, identifies missing/weak
  sections, names strengths, suggests edits, and reports review limitations.
- `list_reports` returns recent generated reports.
- `explain_latest_report` summarizes what the latest saved report contains.
- Added concise deterministic chat responses and a small optional frontend
  review card.
- Kept this phase review-only: no report content is modified or overwritten.

Manual checks:

- Ask: `review the latest report`.
- Ask: `what is missing from the report?`.
- Ask: `how can I improve this report?`.
- Ask: `does the report clearly explain the model?`.
- Ask: `check the report limitations section`.
- Confirm the response is grounded in saved markdown and does not edit the
  report.

## Phase 10C Report Generation Foundation

Implemented:

- Added `apps/api/app/reports` with schemas and deterministic report service.
- Added persisted `Report` records with report type, title, markdown content,
  source summary JSON, and creation time.
- Implemented project report generation from project memory, saved datasets,
  uploaded documents, latest model run, latest simulation, latest optimization,
  and latest workflow run.
- Added fixed markdown sections: Project Overview, Available Data, Uploaded
  Documents, Model Results, Simulation Results, Optimization Results, Workflow
  Recommendations, Limitations, and Recommended Next Steps.
- Added report REST endpoints:
  `POST /projects/{project_id}/reports/generate`,
  `GET /projects/{project_id}/reports`, and
  `GET /projects/{project_id}/reports/{report_id}`.
- Added and registered the approved `generate_project_report` agent tool.
- Routed report-generation phrases such as `generate report`, `create project
  report`, `analysis report`, and `technical summary`.
- Saved `latest_report_id` and `latest_report_title` in project memory.
- Added a simple Reports tab with report list and markdown viewer, plus a small
  report chat result card.

Manual checks:

- Ask: `generate a project report`.
- Ask: `summarize this project as a report`.
- Use the Reports tab to generate a report, list saved reports, and view the
  markdown.
- Confirm no PDF, DOCX, email/share, background job, or autonomous mutation is
  introduced.

## Phase 10B Workflow History And Explanation

Implemented:

- Added read-only workflow tools:
  `list_workflow_runs`, `explain_latest_workflow`, and
  `compare_workflow_runs`.
- Registered the new tools in `ToolRegistry`.
- Routed workflow history, latest workflow/project-analysis, last
  recommendation, and compare-analysis phrases through the rule-based intent
  parser.
- `list_workflow_runs` returns recent persisted `WorkflowRun` summaries.
- `explain_latest_workflow` loads `latest_workflow_run_id` from project memory
  when available, with a latest-created fallback.
- `compare_workflow_runs` compares the latest two workflow runs by status, asset
  availability changes, gaps, and recommendations.
- Added concise deterministic chat responses, including clear no-history and
  single-run messages.
- Added a compact frontend workflow history/explanation/comparison card without
  redesigning the workspace.

Manual checks:

- Ask: `show workflow history`.
- Ask: `explain the latest project analysis`.
- Ask: `what did the last workflow recommend?`.
- Ask: `compare the last two analyses`.
- Confirm no workflow history creates a clear message and does not run a new
  workflow unless the user asks to analyze the project.

## Phase 10A Autonomous Analysis Workflow Foundation

Implemented:

- Added a synchronous workflow module under `apps/api/app/workflows`.
- Added persisted `WorkflowRun` records with step and result JSON.
- Implemented `project_analysis_workflow` to inspect project memory, latest
  dataset, documents, latest model run, latest simulation, and latest
  optimization without launching training, simulation, or optimization.
- Added `POST /projects/{project_id}/workflows/project-analysis`.
- Added and registered the approved `run_project_analysis_workflow` agent tool.
- Added rule-based project-analysis intents for project status, project
  summaries, and next-step recommendations.
- Saved `latest_workflow_run_id` and `latest_project_analysis_summary` in
  project memory.
- Added concise deterministic chat output and a small frontend workflow result
  card.

## Phase 9B Optimization Explanation And Recommendations

Implemented:

- Added approved optimization workflow tools:
  `explain_latest_optimization`, `list_optimization_runs`, and
  `recommend_next_experiment`.
- Registered the new tools in `ToolRegistry`.
- Routed optimization explanation, latest optimization, top candidate, next
  experiment, and optimization-vs-simulation comparison phrases through the
  rule-based intent parser.
- `explain_latest_optimization` loads the latest remembered optimization run
  when possible, reports objective, constraints, best inputs, predicted
  yield/impurity/conversion, and explains the yield/impurity tradeoff.
- `list_optimization_runs` returns recent saved optimization runs for the
  project.
- `recommend_next_experiment` selects one to three top simulated candidates from
  the latest optimization result, includes a reason for each, and saves
  `latest_recommended_experiment` plus `recommended_experiment_count` to project
  memory.
- Added deterministic chat summaries that clearly state optimization
  recommendations are simulated outputs from a simplified benchmark, not
  validated plant instructions.
- Added a compact frontend recommendation card without redesigning the
  workspace.

Manual checks:

- Ask: `explain the latest optimization`.
- Ask: `why were these conditions selected?`.
- Ask: `show top candidate experiments`.
- Ask: `what experiment should I run next?`.
- Ask: `compare optimization with latest simulation`.
- Confirm recommendations are not presented as autonomous execution or real
  plant-valid instructions.

Known follow-ups:

- Add focused regression tests for the new intent mappings and recommendation
  memory writes.
- Add richer optimization result browsing only if the workflow graduates beyond
  chat cards.
- Keep Bayesian optimization, historian connectors, and autonomous execution out
  of scope until explicitly planned.

## Phase 7E Memory-Aware Model Result Explanations

Implemented:

- Added the approved read-only `explain_latest_model` agent tool.
- Registered the tool in `ToolRegistry`.
- Routed model explanation, feature-importance, performance, and next-step
  intents to `explain_latest_model`.
- Loaded the latest model run from `latest_model_run_id` memory, with a fallback
  to the latest persisted model run.
- Returned structured explanation results with model run id, dataset id,
  dataset metadata, target column, task type, metrics, top features,
  limitations, and suggested next steps.
- Added deterministic chat output that warns feature importance indicates
  predictive association, not necessarily causation.
- Preserved regular `list_model_runs` behavior for model listing requests.
- Added tests for memory-based latest model explanation, no-model fallback,
  structured metrics/features, and regular model listing.

Manual checks:

- Train a model.
- Ask: `explain the latest model`.
- Ask: `what are the important features?`
- Ask: `why is the model performing this way?`
- Ask: `what should I try next?`
- Confirm the answer uses saved model results and does not retrain or claim
  causality.

## Phase 7D Automatic Project Summary Memory

Implemented:

- Added the structured memory key `project_summary`.
- Added `update_project_summary(db, project_id)` in the memory service.
- Regenerated the summary after document upload/process, dataset upload,
  target-column selection, model training, and manual memory update/delete.
- Built the summary from project metadata, uploaded document count and latest
  document, dataset count and latest dataset, selected target column, latest
  model run, and user-added project facts.
- Added `project_summary` directly to agent prompt memory context.
- Preserved selected modeling dataset memory during summary sync instead of
  overwriting it with the newest uploaded CSV.
- Returned `No project summary has been created yet.` for empty projects with no
  meaningful project context.
- Added tests for summary updates from document upload, dataset upload, model
  training, manual memory update, empty summary behavior, and prompt inclusion.

Manual checks:

- Upload/process a document and confirm `project_summary` names the latest
  document.
- Upload a dataset and confirm `project_summary` names the latest dataset.
- Set a target column and confirm the summary includes it.
- Train a model and confirm the summary includes the latest model run and task
  type.

## Phase 7C Memory Transparency And Editing

Implemented:

- Added memory-management intents to the rule-based agent parser:
  `list_project_memory`, `remember_project_fact`, `forget_project_memory`, and
  target-column setting through `upsert_project_memory`.
- Added approved memory tools:
  `list_project_memory`, `upsert_project_memory`, and `delete_project_memory`.
- Supported chat commands such as `what do you remember about this project?`,
  `remember that this project is about batch reactor yield optimization`,
  `forget the target column`, and
  `use yield_pct as the target column from now on`.
- Kept memory project-local by writing only to the existing project database
  memory table. No ChatGPT/bio memory is used.
- Added target-column validation against the active/latest project dataset
  before updating `selected_target_column`.
- Updated target-column selection to search all saved project datasets when the
  active/latest dataset does not contain the requested target; if exactly one
  dataset matches, it is remembered as the modeling dataset.
- Added clarification behavior for ambiguous forget requests.
- Added deterministic assistant summaries for memory list, update, validation
  failure, and delete operations.
- Added tests for memory intents, remembering facts, forgetting target memory,
  valid target updates, and invalid target feedback with available columns.

Manual checks:

- Ask: `what do you remember about this project?`
- Ask: `remember that this project is about batch reactor yield optimization`.
- Ask: `forget the target column`.
- Ask: `use yield_pct as the target column from now on` after uploading a
  dataset with that column.
- Try an invalid target column and confirm the assistant lists available
  columns instead of updating memory.

Known follow-ups:

- Add a lightweight frontend Project Memory panel if chat-only transparency is
  not enough.
- Broaden memory phrase coverage carefully without introducing global memory or
  semantic/vector memory.

## Phase 7 Memory Completeness Fix

Implemented:

- Expanded automatic dataset memory to include `latest_dataset_filename` and
  `dataset_count` in addition to `latest_dataset_id`.
- Expanded automatic document memory to include `latest_document_filename` and
  `document_count` in addition to `latest_document_id`.
- Expanded automatic model-training memory to include
  `selected_target_column` and `latest_task_type`.
- Updated the "what do you remember about this project?" response to summarize
  non-model context clearly, including no-dataset and no-model states.
- Added memory sync from existing persisted documents, datasets, and model runs
  before memory is listed, so previous-session uploads are visible without
  re-uploading.
- Kept memory project-scoped and structured; no semantic/vector memory was
  introduced.

Manual checks:

- Upload a PDF and ask: `what do you remember about this project?`
- Confirm the assistant names the latest document and document count.
- Upload a CSV and confirm the latest dataset filename and dataset count appear.
- Train a model and confirm the latest model run, task type, and selected target
  appear.

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
