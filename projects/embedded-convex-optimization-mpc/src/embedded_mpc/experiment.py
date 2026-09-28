"""Repeated-solve MPC benchmark across open-source convex solvers."""

from __future__ import annotations

import json

import numpy as np

from .model import ParametricMPC, double_integrator_config


def run(steps: int = 12) -> dict[str, object]:
    config = double_integrator_config()
    payload: dict[str, object] = {}

    for solver in ("OSQP", "CLARABEL", "SCS"):
        controller = ParametricMPC(config)
        state = np.array([-1.0, 0.3], dtype=float)
        records: list[dict[str, object]] = []

        for step in range(steps):
            result = controller.solve(state, solver=solver, warm_start=step > 0)
            records.append(
                {
                    "step": step,
                    "objective": result.objective,
                    "first_control": result.first_control.tolist(),
                    "dynamics_residual": result.max_dynamics_residual,
                    "bound_violation": result.max_bound_violation,
                    "solver_seconds": result.solver_seconds,
                    "wall_seconds": result.wall_seconds,
                }
            )
            state = config.a @ state + config.b @ result.first_control

        payload[solver] = {
            "final_state": state.tolist(),
            "records": records,
        }

    return {
        "benchmark": "parametric_double_integrator_mpc",
        "warm_start_timing_claimed": False,
        "solvers": payload,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
