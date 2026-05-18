# Roadmap

## Completed Through Phase 10D

- Next.js frontend and FastAPI backend.
- PostgreSQL and SQLAlchemy project persistence.
- Project CSV uploads that persist datasets immediately.
- Baseline ML training endpoint and frontend training UI.
- Ollama local chat integration.
- Persistent project chat history.
- Dataset/model-aware prompting.
- Phase 5A first-version tool-calling agent.
- Persistent workspace model runs for REST and agent-triggered baseline
  training.
- Phase 5B pending action state for multi-turn model-training clarification.
- Phase 5C structured agent tool results for chat responses.
- Phase 6A project-scoped document upload and metadata persistence foundation.
- Phase 6B document chunking with local Ollama embeddings stored as temporary
  JSON.
- Phase 6C backend-only retrieval service over stored document chunk embeddings.
- Phase 6D standalone document Q&A endpoint with grounded citations.
- Phase 6E main project chat agent integration for document RAG questions.
- Phase 7A structured project-scoped memory persistence.
- Phase 7B memory-aware agent tool behavior for datasets, targets, and latest
  model runs.
- Phase 7C memory transparency and user-controlled project memory editing.
- Phase 7D automatic concise `project_summary` memory for compact agent context.
- Phase 7E memory-aware latest model explanations from saved metrics and
  feature importance.
- Phase 8/9 simulation and optimization foundation for the educational batch
  reactor benchmark.
- Phase 9A simple transparent grid-search optimization over batch reactor
  operating conditions.
- Phase 9B optimization explanation, top-candidate listing, and next simulated
  experiment recommendations.
- Phase 10A synchronous project analysis workflow foundation.
- Phase 10B workflow history, latest workflow explanation, and latest-vs-previous
  workflow comparison.
- Phase 10C deterministic markdown project report generation and report viewer
  foundation.
- Phase 10D read-only report review, report listing, and latest report
  explanation.

## Phase 5A Scope

Phase 5A converts the copilot from plain chat into a bounded tool-calling agent.
It understands project-scoped requests such as listing datasets, summarizing the
latest dataset, showing missing values, training a baseline model with a target
column, and asking about previous model runs.

## Phase 5C Scope

Phase 5C returns structured backend tool metadata alongside normal assistant
text. The frontend can render lightweight tool cards in chat without parsing LLM
text, while existing chat compatibility is preserved through the text response.

## Next Phase

Recommended next work:

- Add focused regression tests for the Phase 10 workflow intent mappings and
  history/comparison outputs.
- Add focused regression tests for report generation source summaries and
  markdown sections.
- Add focused regression tests for report review section detection and
  latest-report memory fallback.
- Add a small frontend Project Memory debug panel if memory inspection becomes
  useful outside chat.
- Harden memory intent detection with more examples and evaluation cases.
- Replace JSON embedding storage with pgvector when database migrations and
  extension setup are introduced.
- Add richer argument extraction for dataset selection and target columns.
- Expand pending actions beyond model target-column clarification when new tools
  need multi-turn slot filling.
- Decide whether structured tool metadata should be persisted with chat history
  for replay after refresh.
- Add a richer optimization result view if optimization workflows become a core
  workspace surface.
- Keep the project analysis workflow transparent and synchronous until there is
  a clear need for background execution or graph orchestration.

Out of scope until later phases:

- Vector or semantic long-term memory.
- pgvector retrieval and LangChain.
- LangGraph workflow orchestration.
- Background workflow jobs.
- PDF/DOCX report export.
- Report editing and revision workflow.
- Email/share workflows.
- Bayesian optimization.
- Historian connector.
- Autonomous experiment execution.
- Authentication.
- Paid model APIs.
