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


def compare_warm_cold(solver: str = "OSQP", repeats: int = 6) -> dict[str, object]:
    """Compare repeated solves with and without warm starts without asserting a speed winner."""
    config=double_integrator_config()
    initial=np.array([-0.8,0.25],dtype=float)
    controller=ParametricMPC(config)
    warm=[]
    cold=[]
    warm_obj=[]
    cold_obj=[]
    for k in range(repeats):
        state=initial + np.array([0.03*k,0.0])
        w=controller.solve(state,solver=solver,warm_start=True)
        cold_controller=ParametricMPC(config)
        d=cold_controller.solve(state,solver=solver,warm_start=False)
        warm.append(w.wall_seconds)
        cold.append(d.wall_seconds)
        warm_obj.append(w.objective)
        cold_obj.append(d.objective)
    return {
        "solver":solver,
        "warm_wall_seconds":warm,
        "cold_wall_seconds":cold,
        "objective_max_abs_difference":float(np.max(np.abs(np.asarray(warm_obj)-np.asarray(cold_obj)))),
        "speedup_claimed":False,
    }
