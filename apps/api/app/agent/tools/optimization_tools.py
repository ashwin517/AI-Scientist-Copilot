import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import OptimizationRun, SimulationRun

from app.optimization.optimization_service import (
    list_project_optimization_runs,
    optimization_run_payload,
    run_and_persist_batch_reactor_optimization,
)
from app.optimization.schemas import (
    BatchReactorOptimizationInput,
    BatchReactorSearchSpace,
    NumericSearchRange,
)
from app.services.memory_service import get_memory, update_project_summary, upsert_memory


NO_OPTIMIZATION_AVAILABLE_MESSAGE = (
    "No optimization runs are saved in this project yet. Run a batch reactor "
    "optimization first."
)
NO_SIMULATION_AVAILABLE_MESSAGE = (
    "No simulation runs are saved in this project yet, so I can explain the "
    "optimization but cannot compare it with a simulation."
)


def optimize_batch_reactor(
    db: Session,
    project_id: int,
    penalty_weight: float = 1.0,
    max_final_impurity: float | None = 0.15,
) -> dict[str, object]:
    optimization_input = BatchReactorOptimizationInput(
        penalty_weight=penalty_weight,
        max_final_impurity=max_final_impurity,
        search_space=BatchReactorSearchSpace(
            temperature_c=NumericSearchRange(min=70.0, max=110.0, steps=9),
            batch_time_min=NumericSearchRange(min=30.0, max=240.0, steps=8),
            initial_concentration=NumericSearchRange(min=0.5, max=2.0, steps=4),
            catalyst_factor=NumericSearchRange(min=0.5, max=2.0, steps=4),
        ),
    )
    optimization_run, result = run_and_persist_batch_reactor_optimization(
        db,
        project_id,
        optimization_input,
    )
    return optimization_run_payload(optimization_run, result)


def list_optimization_runs(
    db: Session,
    project_id: int,
    limit: int = 10,
) -> dict[str, object]:
    optimization_runs = list_project_optimization_runs(db, project_id)[:limit]
    return {
        "optimization_runs": [
            _optimization_summary_payload(optimization_run)
            for optimization_run in optimization_runs
        ],
        "count": len(optimization_runs),
    }


def explain_latest_optimization(
    db: Session,
    project_id: int,
    include_latest_simulation_comparison: bool = False,
) -> dict[str, object]:
    optimization_run = _latest_optimization_run(db, project_id)
    if optimization_run is None:
        return {
            "optimization_available": False,
            "message": NO_OPTIMIZATION_AVAILABLE_MESSAGE,
        }

    payload = optimization_run_payload(optimization_run)
    result = _dict_value(payload.get("result"))
    best_inputs = _dict_value(payload.get("best_inputs"))
    constraints = _dict_value(payload.get("constraints"))
    explanation = _optimization_explanation(
        objective=str(payload.get("objective") or ""),
        constraints=constraints,
        best_inputs=best_inputs,
        result=result,
    )
    response: dict[str, object] = {
        "optimization_available": True,
        "optimization_run_id": payload["optimization_run_id"],
        "project_id": payload["project_id"],
        "optimization_type": payload["optimization_type"],
        "created_at": optimization_run.created_at.isoformat(),
        "objective": payload.get("objective"),
        "constraints": constraints,
        "search_space": payload.get("search_space", {}),
        "best_inputs": best_inputs,
        "best_final_yield": payload.get("best_final_yield"),
        "best_final_impurity": payload.get("best_final_impurity"),
        "best_conversion": payload.get("best_conversion"),
        "objective_value": payload.get("objective_value"),
        "top_candidates": payload.get("top_candidates", []),
        "evaluated_candidates": payload.get("evaluated_candidates"),
        "feasible_candidates": payload.get("feasible_candidates"),
        "explanation": explanation,
        "model_note": payload.get("note"),
    }

    if include_latest_simulation_comparison:
        response["simulation_comparison"] = _compare_with_latest_simulation(
            db,
            project_id,
            payload,
        )

    return response


def recommend_next_experiment(
    db: Session,
    project_id: int,
    count: int = 3,
) -> dict[str, object]:
    optimization_run = _latest_optimization_run(db, project_id)
    if optimization_run is None:
        return {
            "recommendation_available": False,
            "message": NO_OPTIMIZATION_AVAILABLE_MESSAGE,
        }

    count = max(1, min(count, 3))
    payload = optimization_run_payload(optimization_run)
    candidates = [
        candidate
        for candidate in _list_value(payload.get("top_candidates"))
        if isinstance(candidate, dict)
    ][:count]
    recommendations = [
        {
            "rank": index + 1,
            "inputs": _dict_value(candidate.get("inputs")),
            "final_yield": candidate.get("final_yield"),
            "final_impurity": candidate.get("final_impurity"),
            "conversion": candidate.get("conversion"),
            "objective_value": candidate.get("objective_value"),
            "constraint_satisfied": candidate.get("constraint_satisfied"),
            "reason": _candidate_reason(index, candidate),
        }
        for index, candidate in enumerate(candidates)
    ]

    upsert_memory(
        db,
        project_id,
        "latest_recommended_experiment",
        recommendations[0] if recommendations else None,
        memory_type="optimization",
        source="optimization_recommendation",
    )
    upsert_memory(
        db,
        project_id,
        "recommended_experiment_count",
        len(recommendations),
        memory_type="optimization",
        source="optimization_recommendation",
    )
    update_project_summary(db, project_id)

    return {
        "recommendation_available": True,
        "optimization_run_id": payload.get("optimization_run_id"),
        "optimization_type": payload.get("optimization_type"),
        "recommendations": recommendations,
        "recommendation_count": len(recommendations),
        "note": (
            "These are simulated recommendations from the simplified batch "
            "reactor benchmark, not validated plant instructions."
        ),
    }


def _latest_optimization_run(db: Session, project_id: int) -> OptimizationRun | None:
    memory = get_memory(db, project_id, "latest_optimization_run_id")
    remembered_id = _as_int(_decode_memory(memory.value_json) if memory else None)
    if remembered_id:
        remembered_run = db.get(OptimizationRun, remembered_id)
        if remembered_run is not None and remembered_run.project_id == project_id:
            return remembered_run

    optimization_runs = list_project_optimization_runs(db, project_id)
    return optimization_runs[0] if optimization_runs else None


def _latest_simulation_run(db: Session, project_id: int) -> SimulationRun | None:
    query = (
        select(SimulationRun)
        .where(SimulationRun.project_id == project_id)
        .order_by(SimulationRun.created_at.desc(), SimulationRun.id.desc())
    )
    return db.execute(query).scalars().first()


def _optimization_summary_payload(
    optimization_run: OptimizationRun,
) -> dict[str, object]:
    payload = optimization_run_payload(optimization_run)
    return {
        "optimization_run_id": payload["optimization_run_id"],
        "project_id": payload["project_id"],
        "optimization_type": payload["optimization_type"],
        "created_at": optimization_run.created_at.isoformat(),
        "objective": payload.get("objective"),
        "constraints": payload.get("constraints", {}),
        "best_inputs": payload.get("best_inputs", {}),
        "best_final_yield": payload.get("best_final_yield"),
        "best_final_impurity": payload.get("best_final_impurity"),
        "best_conversion": payload.get("best_conversion"),
        "objective_value": payload.get("objective_value"),
        "feasible_candidates": payload.get("feasible_candidates"),
        "evaluated_candidates": payload.get("evaluated_candidates"),
    }


def _optimization_explanation(
    *,
    objective: str,
    constraints: dict[str, object],
    best_inputs: dict[str, object],
    result: dict[str, object],
) -> str:
    max_impurity = constraints.get("max_final_impurity")
    return (
        f"The optimizer selected {best_inputs.get('temperature_c')} C, "
        f"{best_inputs.get('batch_time_min')} minutes, initial concentration "
        f"{best_inputs.get('initial_concentration')}, and catalyst factor "
        f"{best_inputs.get('catalyst_factor')} because that candidate gave the "
        f"best value of the objective ({objective}) among the searched grid. "
        f"The predicted yield is {result.get('best_final_yield')}, impurity is "
        f"{result.get('best_final_impurity')}, and conversion is "
        f"{result.get('best_conversion')}. The tradeoff is that longer or more "
        "aggressive conditions can improve conversion and desired intermediate "
        "formation, but can also push material onward into impurity. "
        f"The impurity constraint used here was {max_impurity}."
    )


def _compare_with_latest_simulation(
    db: Session,
    project_id: int,
    optimization_payload: dict[str, object],
) -> dict[str, object]:
    simulation_run = _latest_simulation_run(db, project_id)
    if simulation_run is None:
        return {
            "comparison_available": False,
            "message": NO_SIMULATION_AVAILABLE_MESSAGE,
        }

    simulation_input = _loads_dict(simulation_run.input_json)
    simulation_result = _loads_dict(simulation_run.result_json)
    return {
        "comparison_available": True,
        "simulation_run_id": simulation_run.id,
        "simulation_input": simulation_input,
        "simulation_result": {
            "final_yield": simulation_result.get("final_yield"),
            "final_impurity": simulation_result.get("final_impurity"),
            "conversion": simulation_result.get("conversion"),
        },
        "optimization_best_inputs": optimization_payload.get("best_inputs", {}),
        "result_differences": {
            "final_yield": _difference(
                optimization_payload.get("best_final_yield"),
                simulation_result.get("final_yield"),
            ),
            "final_impurity": _difference(
                optimization_payload.get("best_final_impurity"),
                simulation_result.get("final_impurity"),
            ),
            "conversion": _difference(
                optimization_payload.get("best_conversion"),
                simulation_result.get("conversion"),
            ),
        },
        "interpretation": (
            "Positive yield and conversion differences mean the optimized "
            "candidate is predicted higher than the latest simulation. A "
            "positive impurity difference means it is also predicted to make "
            "more impurity, so compare that against the impurity constraint."
        ),
    }


def _candidate_reason(index: int, candidate: dict[str, Any]) -> str:
    if index == 0:
        return (
            "Highest ranked grid-search candidate by the yield-minus-impurity "
            "objective while respecting the configured impurity rule when possible."
        )
    if candidate.get("constraint_satisfied"):
        return (
            "Close alternative with a feasible impurity prediction; useful as a "
            "neighboring confirmation experiment."
        )
    return (
        "Included as a lower-ranked simulated tradeoff point; review impurity "
        "before considering it."
    )


def _loads_dict(value: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _decode_memory(value_json: str) -> object:
    try:
        return json.loads(value_json)
    except json.JSONDecodeError:
        return value_json


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _difference(candidate_value: object, baseline_value: object) -> float | None:
    if not isinstance(candidate_value, int | float):
        return None
    if not isinstance(baseline_value, int | float):
        return None
    return round(float(candidate_value) - float(baseline_value), 8)


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []
