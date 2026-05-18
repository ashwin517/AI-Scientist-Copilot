import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import SimulationRun
from app.services.memory_service import update_project_summary, upsert_memory
from app.simulation.batch_reactor import simulate_batch_reactor
from app.simulation.schemas import (
    BatchReactorSimulationInput,
    BatchReactorSimulationOutput,
)


BATCH_REACTOR_SIMULATION_TYPE = "batch_reactor"


def run_and_persist_batch_reactor_simulation(
    db: Session,
    project_id: int,
    simulation_input: BatchReactorSimulationInput,
) -> tuple[SimulationRun, BatchReactorSimulationOutput]:
    result = simulate_batch_reactor(simulation_input)
    simulation_run = SimulationRun(
        project_id=project_id,
        simulation_type=BATCH_REACTOR_SIMULATION_TYPE,
        input_json=simulation_input.model_dump_json(),
        result_json=result.model_dump_json(),
    )
    db.add(simulation_run)
    db.commit()
    db.refresh(simulation_run)

    upsert_memory(
        db,
        project_id,
        "latest_simulation_run_id",
        simulation_run.id,
        memory_type="simulation",
        source="batch_reactor_simulation",
    )
    upsert_memory(
        db,
        project_id,
        "latest_simulation_type",
        BATCH_REACTOR_SIMULATION_TYPE,
        memory_type="simulation",
        source="batch_reactor_simulation",
    )
    update_project_summary(db, project_id)

    return simulation_run, result


def list_project_simulation_runs(
    db: Session,
    project_id: int,
) -> list[SimulationRun]:
    query = (
        select(SimulationRun)
        .where(SimulationRun.project_id == project_id)
        .order_by(SimulationRun.created_at.desc(), SimulationRun.id.desc())
    )
    return list(db.execute(query).scalars().all())


def simulation_run_payload(
    simulation_run: SimulationRun,
    result: BatchReactorSimulationOutput | None = None,
) -> dict[str, object]:
    result_payload = (
        result.model_dump()
        if result is not None
        else json.loads(simulation_run.result_json)
    )
    input_payload = json.loads(simulation_run.input_json)
    return {
        "simulation_run_id": simulation_run.id,
        "project_id": simulation_run.project_id,
        "simulation_type": simulation_run.simulation_type,
        "input": input_payload,
        "result": result_payload,
        **result_payload,
    }
