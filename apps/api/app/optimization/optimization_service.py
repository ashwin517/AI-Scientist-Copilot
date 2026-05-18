import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import OptimizationRun
from app.optimization.batch_reactor_optimizer import (
    BATCH_REACTOR_OPTIMIZATION_TYPE,
    OBJECTIVE_DESCRIPTION,
    optimize_batch_reactor_grid,
)
from app.optimization.schemas import (
    BatchReactorOptimizationInput,
    BatchReactorOptimizationResult,
)
from app.services.memory_service import update_project_summary, upsert_memory


def run_and_persist_batch_reactor_optimization(
    db: Session,
    project_id: int,
    optimization_input: BatchReactorOptimizationInput,
) -> tuple[OptimizationRun, BatchReactorOptimizationResult]:
    result = optimize_batch_reactor_grid(optimization_input)
    optimization_run = OptimizationRun(
        project_id=project_id,
        optimization_type=BATCH_REACTOR_OPTIMIZATION_TYPE,
        objective=OBJECTIVE_DESCRIPTION,
        constraints_json=json.dumps(result.constraints),
        search_space_json=optimization_input.search_space.model_dump_json(),
        result_json=result.model_dump_json(),
    )
    db.add(optimization_run)
    db.commit()
    db.refresh(optimization_run)

    upsert_memory(
        db,
        project_id,
        "latest_optimization_run_id",
        optimization_run.id,
        memory_type="optimization",
        source="batch_reactor_optimization",
    )
    upsert_memory(
        db,
        project_id,
        "latest_optimization_type",
        BATCH_REACTOR_OPTIMIZATION_TYPE,
        memory_type="optimization",
        source="batch_reactor_optimization",
    )
    update_project_summary(db, project_id)

    return optimization_run, result


def list_project_optimization_runs(
    db: Session,
    project_id: int,
) -> list[OptimizationRun]:
    query = (
        select(OptimizationRun)
        .where(OptimizationRun.project_id == project_id)
        .order_by(OptimizationRun.created_at.desc(), OptimizationRun.id.desc())
    )
    return list(db.execute(query).scalars().all())


def optimization_run_payload(
    optimization_run: OptimizationRun,
    result: BatchReactorOptimizationResult | None = None,
) -> dict[str, object]:
    result_payload = (
        result.model_dump()
        if result is not None
        else json.loads(optimization_run.result_json)
    )
    return {
        "optimization_run_id": optimization_run.id,
        "project_id": optimization_run.project_id,
        "optimization_type": optimization_run.optimization_type,
        "objective": optimization_run.objective,
        "constraints": json.loads(optimization_run.constraints_json),
        "search_space": json.loads(optimization_run.search_space_json),
        "result": result_payload,
        **result_payload,
    }

