from math import exp

import numpy as np
from scipy.integrate import solve_ivp

from app.simulation.schemas import (
    BatchReactorSimulationInput,
    BatchReactorSimulationOutput,
)


R_GAS_CONSTANT = 8.314
K10_PER_MINUTE = 2.0e3
K20_PER_MINUTE = 6.0e2
E1_J_PER_MOL = 28_000.0
E2_J_PER_MOL = 33_000.0
SIMULATION_NOTE = (
    "Simple educational benchmark for A -> B -> C batch kinetics; "
    "not calibrated for real chemistry."
)


def simulate_batch_reactor(
    simulation_input: BatchReactorSimulationInput,
) -> BatchReactorSimulationOutput:
    temperature_kelvin = simulation_input.temperature + 273.15
    k1 = (
        K10_PER_MINUTE
        * exp(-E1_J_PER_MOL / (R_GAS_CONSTANT * temperature_kelvin))
        * simulation_input.catalyst_factor
    )
    k2 = (
        K20_PER_MINUTE
        * exp(-E2_J_PER_MOL / (R_GAS_CONSTANT * temperature_kelvin))
        * simulation_input.catalyst_factor
    )

    time_grid = np.linspace(
        0.0,
        simulation_input.batch_time,
        simulation_input.time_points,
    )
    initial_state = [simulation_input.initial_concentration, 0.0, 0.0]

    solution = solve_ivp(
        fun=lambda _time, state: _batch_reactor_ode(state, k1, k2),
        t_span=(0.0, simulation_input.batch_time),
        y0=initial_state,
        t_eval=time_grid,
        method="RK45",
    )
    if not solution.success:
        raise ValueError("Batch reactor ODE solver failed.")

    concentrations = np.maximum(solution.y, 0.0)
    ca_profile = concentrations[0].tolist()
    cb_profile = concentrations[1].tolist()
    cc_profile = concentrations[2].tolist()
    initial_ca = simulation_input.initial_concentration
    final_ca = ca_profile[-1]
    final_cb = cb_profile[-1]
    final_cc = cc_profile[-1]

    return BatchReactorSimulationOutput(
        time_grid=_round_list(time_grid.tolist()),
        CA_profile=_round_list(ca_profile),
        CB_profile=_round_list(cb_profile),
        CC_profile=_round_list(cc_profile),
        final_yield=round(final_cb / initial_ca, 8),
        final_impurity=round(final_cc / initial_ca, 8),
        conversion=round((initial_ca - final_ca) / initial_ca, 8),
        rate_constants={"k1": round(k1, 10), "k2": round(k2, 10)},
        note=SIMULATION_NOTE,
    )


def _batch_reactor_ode(state: list[float], k1: float, k2: float) -> list[float]:
    ca, cb, _cc = state
    return [
        -k1 * ca,
        k1 * ca - k2 * cb,
        k2 * cb,
    ]


def _round_list(values: list[float]) -> list[float]:
    return [round(float(value), 8) for value in values]
