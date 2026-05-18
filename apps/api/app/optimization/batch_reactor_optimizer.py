from itertools import product

from app.optimization.schemas import (
    BatchReactorOptimizationCandidate,
    BatchReactorOptimizationInput,
    BatchReactorOptimizationResult,
    NumericSearchRange,
)
from app.simulation.batch_reactor import SIMULATION_NOTE, simulate_batch_reactor
from app.simulation.schemas import BatchReactorSimulationInput


BATCH_REACTOR_OPTIMIZATION_TYPE = "batch_reactor"
OBJECTIVE_DESCRIPTION = "maximize final_yield - penalty_weight * final_impurity"
OPTIMIZATION_NOTE = (
    f"{SIMULATION_NOTE} Optimization uses a transparent grid search over the "
    "configured operating ranges."
)


def optimize_batch_reactor_grid(
    optimization_input: BatchReactorOptimizationInput,
) -> BatchReactorOptimizationResult:
    search_space = optimization_input.search_space
    candidates: list[BatchReactorOptimizationCandidate] = []

    for temperature_c, batch_time_min, initial_concentration, catalyst_factor in product(
        _linspace(search_space.temperature_c),
        _linspace(search_space.batch_time_min),
        _linspace(search_space.initial_concentration),
        _linspace(search_space.catalyst_factor),
    ):
        simulation_result = simulate_batch_reactor(
            BatchReactorSimulationInput(
                temperature=temperature_c,
                batch_time=batch_time_min,
                initial_concentration=initial_concentration,
                catalyst_factor=catalyst_factor,
                time_points=61,
            )
        )
        objective_value = (
            simulation_result.final_yield
            - optimization_input.penalty_weight * simulation_result.final_impurity
        )
        constraint_satisfied = (
            optimization_input.max_final_impurity is None
            or simulation_result.final_impurity
            <= optimization_input.max_final_impurity
        )
        candidates.append(
            BatchReactorOptimizationCandidate(
                inputs={
                    "temperature_c": round(temperature_c, 8),
                    "batch_time_min": round(batch_time_min, 8),
                    "initial_concentration": round(initial_concentration, 8),
                    "catalyst_factor": round(catalyst_factor, 8),
                },
                final_yield=simulation_result.final_yield,
                final_impurity=simulation_result.final_impurity,
                conversion=simulation_result.conversion,
                objective_value=round(objective_value, 8),
                constraint_satisfied=constraint_satisfied,
            )
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.constraint_satisfied,
            candidate.objective_value,
            candidate.final_yield,
        ),
        reverse=True,
    )
    best = ranked_candidates[0]

    return BatchReactorOptimizationResult(
        best_inputs=best.inputs,
        best_final_yield=best.final_yield,
        best_final_impurity=best.final_impurity,
        best_conversion=best.conversion,
        objective_value=best.objective_value,
        top_candidates=ranked_candidates[: optimization_input.top_k],
        evaluated_candidates=len(candidates),
        feasible_candidates=sum(
            1 for candidate in candidates if candidate.constraint_satisfied
        ),
        objective=OBJECTIVE_DESCRIPTION,
        constraints={"max_final_impurity": optimization_input.max_final_impurity},
        search_space=optimization_input.search_space.model_dump(),
        note=OPTIMIZATION_NOTE,
    )


def _linspace(search_range: NumericSearchRange) -> list[float]:
    if search_range.steps == 1:
        return [search_range.min]
    step_size = (search_range.max - search_range.min) / (search_range.steps - 1)
    return [
        search_range.min + index * step_size
        for index in range(search_range.steps)
    ]

