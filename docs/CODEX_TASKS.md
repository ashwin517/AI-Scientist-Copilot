# Codex Tasks

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
