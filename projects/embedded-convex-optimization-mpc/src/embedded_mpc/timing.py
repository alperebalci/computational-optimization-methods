"""Repeated-solve benchmark for parametric MPC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .model import ParametricMPC, SolverName


@dataclass(frozen=True)
class RepeatedSolveBenchmark:
    solver: str
    warm_wall_median: float
    cold_wall_median: float
    warm_solver_median: float | None
    cold_solver_median: float | None
    max_objective_difference: float


def _median_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return None if not finite else float(np.median(finite))


def benchmark_repeated_solves(
    model: ParametricMPC,
    initial_states: ArrayLike,
    *,
    solver: SolverName = "OSQP",
) -> RepeatedSolveBenchmark:
    """Compare repeated warm-start and cold-start solves on identical states."""

    states = np.asarray(initial_states, dtype=float)
    if states.ndim != 2 or states.shape[1] != model.config.nx or states.shape[0] < 2:
        raise ValueError("initial_states must contain at least two valid state rows")

    cold = [model.solve(state, solver=solver, warm_start=False) for state in states]
    warm = [model.solve(state, solver=solver, warm_start=True) for state in states]

    differences = [
        abs(cold_result.objective - warm_result.objective)
        for cold_result, warm_result in zip(cold, warm, strict=True)
    ]
    return RepeatedSolveBenchmark(
        solver=solver,
        warm_wall_median=float(np.median([result.wall_seconds for result in warm])),
        cold_wall_median=float(np.median([result.wall_seconds for result in cold])),
        warm_solver_median=_median_optional([result.solver_seconds for result in warm]),
        cold_solver_median=_median_optional([result.solver_seconds for result in cold]),
        max_objective_difference=float(max(differences)),
    )
