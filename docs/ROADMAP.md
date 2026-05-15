# Roadmap

## Completed Through Phase 5A

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

- Decide how the standalone document Q&A flow should be surfaced in the UI.
- Harden document RAG intent detection with more examples and evaluation cases.
- Replace JSON embedding storage with pgvector when database migrations and
  extension setup are introduced.
- Add document deletion and storage cleanup if document management becomes
  user-facing beyond upload/list/detail.
- Add tests around dataset and training tools with a test database.
- Add richer argument extraction for dataset selection and target columns.
- Expand pending actions beyond model target-column clarification when new tools
  need multi-turn slot filling.
- Decide whether structured tool metadata should be persisted with chat history
  for replay after refresh.
- Design retrieval and source attribution before enabling document-grounded chat.

Out of scope until later phases:

- pgvector retrieval and LangChain.
- Simulation.
- Optimization.
- Authentication.
- Paid model APIs.
