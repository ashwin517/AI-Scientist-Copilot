from sqlalchemy.orm import Session

from app.db.models import SimulationRun
from app.simulation.schemas import BatchReactorSimulationInput
from app.simulation.simulation_service import (
    list_project_simulation_runs,
    run_and_persist_batch_reactor_simulation,
    simulation_run_payload,
)


NO_SIMULATION_AVAILABLE_MESSAGE = (
    "No simulation runs are saved in this project yet. Run a batch reactor "
    "simulation first."
)
NOT_ENOUGH_SIMULATIONS_MESSAGE = (
    "At least two simulation runs are needed to compare scenarios."
)


def run_batch_reactor_simulation(
    db: Session,
    project_id: int,
    temperature: float = 85.0,
    batch_time: float = 120.0,
    initial_concentration: float = 1.0,
    catalyst_factor: float = 1.0,
) -> dict[str, object]:
    simulation_input = BatchReactorSimulationInput(
        temperature=temperature,
        batch_time=batch_time,
        initial_concentration=initial_concentration,
        catalyst_factor=catalyst_factor,
    )
    simulation_run, result = run_and_persist_batch_reactor_simulation(
        db,
        project_id,
        simulation_input,
    )
    return simulation_run_payload(simulation_run, result)


def list_simulation_runs(
    db: Session,
    project_id: int,
    limit: int = 10,
) -> dict[str, object]:
    simulation_runs = list_project_simulation_runs(db, project_id)[:limit]
    return {
        "simulation_runs": [
            _summary_payload(simulation_run)
            for simulation_run in simulation_runs
        ],
        "count": len(simulation_runs),
    }


def explain_latest_simulation(
    db: Session,
    project_id: int,
) -> dict[str, object]:
    simulation_run = _latest_simulation_run(db, project_id)
    if simulation_run is None:
        return {
            "simulation_available": False,
            "message": NO_SIMULATION_AVAILABLE_MESSAGE,
        }

    payload = _summary_payload(simulation_run)
    result = payload["result"]
    input_values = payload["input"]
    return {
        "simulation_available": True,
        **payload,
        "interpretation": _interpret_simulation(input_values, result),
        "model_note": _model_note(result),
    }


def compare_simulation_runs(
    db: Session,
    project_id: int,
    first_simulation_run_id: int | None = None,
    second_simulation_run_id: int | None = None,
) -> dict[str, object]:
    simulation_runs = list_project_simulation_runs(db, project_id)
    if len(simulation_runs) == 0:
        return {
            "comparison_available": False,
            "message": NO_SIMULATION_AVAILABLE_MESSAGE,
        }

    selected_runs = _select_comparison_runs(
        simulation_runs,
        first_simulation_run_id,
        second_simulation_run_id,
    )
    if selected_runs is None:
        return {
            "comparison_available": False,
            "message": NOT_ENOUGH_SIMULATIONS_MESSAGE,
        }

    newer, older = selected_runs
    newer_payload = _summary_payload(newer)
    older_payload = _summary_payload(older)
    result_differences = _result_differences(older_payload["result"], newer_payload["result"])
    input_differences = _input_differences(older_payload["input"], newer_payload["input"])

    return {
        "comparison_available": True,
        "baseline": older_payload,
        "candidate": newer_payload,
        "input_differences": input_differences,
        "result_differences": result_differences,
        "interpretation": _interpret_comparison(input_differences, result_differences),
        "model_note": _model_note(newer_payload["result"]),
    }


def _latest_simulation_run(db: Session, project_id: int) -> SimulationRun | None:
    simulation_runs = list_project_simulation_runs(db, project_id)
    return simulation_runs[0] if simulation_runs else None


def _select_comparison_runs(
    simulation_runs: list[SimulationRun],
    first_simulation_run_id: int | None,
    second_simulation_run_id: int | None,
) -> tuple[SimulationRun, SimulationRun] | None:
    if first_simulation_run_id is None and second_simulation_run_id is None:
        if len(simulation_runs) < 2:
            return None
        return simulation_runs[0], simulation_runs[1]

    by_id = {simulation_run.id: simulation_run for simulation_run in simulation_runs}
    if first_simulation_run_id is None or second_simulation_run_id is None:
        return None
    first = by_id.get(first_simulation_run_id)
    second = by_id.get(second_simulation_run_id)
    if first is None or second is None or first.id == second.id:
        return None
    return first, second


def _summary_payload(simulation_run: SimulationRun) -> dict[str, object]:
    payload = simulation_run_payload(simulation_run)
    return {
        "simulation_run_id": payload["simulation_run_id"],
        "project_id": payload["project_id"],
        "simulation_type": payload["simulation_type"],
        "created_at": simulation_run.created_at.isoformat(),
        "input": payload["input"],
        "result": {
            "final_yield": payload["final_yield"],
            "final_impurity": payload["final_impurity"],
            "conversion": payload["conversion"],
            "rate_constants": payload.get("rate_constants", {}),
            "note": payload.get("note"),
        },
    }


def _input_differences(
    baseline_input: object,
    candidate_input: object,
) -> dict[str, float]:
    if not isinstance(baseline_input, dict) or not isinstance(candidate_input, dict):
        return {}

    differences: dict[str, float] = {}
    for key in (
        "temperature",
        "batch_time",
        "initial_concentration",
        "catalyst_factor",
    ):
        baseline_value = baseline_input.get(key)
        candidate_value = candidate_input.get(key)
        if isinstance(baseline_value, int | float) and isinstance(candidate_value, int | float):
            difference = float(candidate_value) - float(baseline_value)
            if difference != 0:
                differences[key] = round(difference, 8)
    return differences


def _result_differences(
    baseline_result: object,
    candidate_result: object,
) -> dict[str, float]:
    if not isinstance(baseline_result, dict) or not isinstance(candidate_result, dict):
        return {}

    differences: dict[str, float] = {}
    for key in ("final_yield", "final_impurity", "conversion"):
        baseline_value = baseline_result.get(key)
        candidate_value = candidate_result.get(key)
        if isinstance(baseline_value, int | float) and isinstance(candidate_value, int | float):
            differences[key] = round(float(candidate_value) - float(baseline_value), 8)
    return differences


def _interpret_simulation(input_values: object, result: object) -> str:
    if not isinstance(input_values, dict) or not isinstance(result, dict):
        return "The saved simulation could not be interpreted."

    final_yield = float(result.get("final_yield", 0.0))
    final_impurity = float(result.get("final_impurity", 0.0))
    conversion = float(result.get("conversion", 0.0))
    temperature = input_values.get("temperature")
    batch_time = input_values.get("batch_time")
    return (
        f"At {temperature} C for {batch_time} minutes, the simplified model "
        f"ends with yield {final_yield:.4g}, impurity {final_impurity:.4g}, "
        f"and conversion {conversion:.4g}. B is the desired intermediate, so "
        "higher B is favorable, while higher C indicates more impurity formation."
    )


def _interpret_comparison(
    input_differences: dict[str, float],
    result_differences: dict[str, float],
) -> str:
    changed_inputs = ", ".join(
        f"{key} {value:+.4g}" for key, value in input_differences.items()
    )
    if not changed_inputs:
        changed_inputs = "no input changes"

    impurity_delta = result_differences.get("final_impurity", 0.0)
    yield_delta = result_differences.get("final_yield", 0.0)
    conversion_delta = result_differences.get("conversion", 0.0)
    impurity_text = "increased" if impurity_delta > 0 else "decreased or stayed flat"
    return (
        f"Compared with the older run, the newer run changed {changed_inputs}. "
        f"Yield changed by {yield_delta:+.4g}, impurity changed by "
        f"{impurity_delta:+.4g}, and conversion changed by {conversion_delta:+.4g}. "
        f"Impurity {impurity_text}; in this A -> B -> C benchmark, that means "
        "more material progressed from desired B into impurity C."
    )


def _model_note(result: object) -> str:
    if isinstance(result, dict) and isinstance(result.get("note"), str):
        return str(result["note"])
    return "Simple educational benchmark for A -> B -> C kinetics; not calibrated for real chemistry."
