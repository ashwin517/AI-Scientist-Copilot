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

## Phase 5A Scope

Phase 5A converts the copilot from plain chat into a bounded tool-calling agent.
It understands project-scoped requests such as listing datasets, summarizing the
latest dataset, showing missing values, training a baseline model with a target
column, and asking about previous model runs.

## Next Phase

Recommended Phase 5B work:

- Add tests around dataset and training tools with a test database.
- Add richer argument extraction for dataset selection and target columns.
- Expand pending actions beyond model target-column clarification when new tools
  need multi-turn slot filling.
- Consider moving chat persistence fully behind the backend agent route.

Out of scope until later phases:

- RAG.
- Simulation.
- Optimization.
- Authentication.
- Paid model APIs.
