"""Matched CPU first-order benchmark with an optional CuPy parity run."""

from __future__ import annotations

import json

from .solver import SolveResult, highs_reference, make_demo_problem, solve_first_order


def _payload(result: SolveResult, reference: float) -> dict[str, object]:
    return {
        "mode": result.mode,
        "backend": result.backend,
        "objective": result.objective,
        "relative_objective_error": abs(result.objective - reference) / max(1.0, abs(reference)),
        "primal_residual": result.primal_residual,
        "projected_stationarity": result.projected_stationarity,
        "iterations": result.iterations,
        "converged": result.converged,
        "wall_seconds": result.wall_seconds,
        "primal_weight": result.primal_weight,
    }


def run(seed: int = 12) -> dict[str, object]:
    problem = make_demo_problem(seed=seed, rows=12, columns=40)
    reference, _ = highs_reference(problem)

    results = [
        solve_first_order(problem, mode="pdhg", max_iter=20_000),
        solve_first_order(problem, mode="halpern", max_iter=20_000),
        solve_first_order(problem, mode="restarted_halpern", max_iter=20_000),
        solve_first_order(problem, mode="pid_adaptive_halpern", max_iter=20_000),
    ]

    gpu_payload: dict[str, object] | None
    try:
        gpu = solve_first_order(
            problem,
            mode="restarted_halpern",
            backend="cupy",
            max_iter=20_000,
        )
        gpu_payload = _payload(gpu, reference)
    except RuntimeError:
        gpu_payload = None

    return {
        "highs_objective": reference,
        "cpu_results": [_payload(result, reference) for result in results],
        "gpu_result": gpu_payload,
        "gpu_speedup_claimed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
