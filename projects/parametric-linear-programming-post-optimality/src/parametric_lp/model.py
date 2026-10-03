"""A transparent one-parameter LP with solver-derived dual sensitivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class ParametricSolution:
    demand: float
    x: np.ndarray
    objective: float
    demand_shadow_price: float
    active_set: tuple[str, ...]


class ParametricProductionLP:
    """Meet demand using cheap capped capacity and expensive flexible capacity.

    min x_cheap + 3 x_flex
    s.t. x_cheap + x_flex >= demand
         x_cheap <= cheap_capacity
         x >= 0

    The value function is piecewise linear, and the dual multiplier on demand
    changes when cheap capacity becomes binding.
    """

    def __init__(self, cheap_capacity: float = 4.0):
        if cheap_capacity <= 0:
            raise ValueError("cheap_capacity must be positive")
        self.cheap_capacity = float(cheap_capacity)

    def solve(self, demand: float, tol: float = 1e-8) -> ParametricSolution:
        if demand < 0:
            raise ValueError("demand must be nonnegative")
        result = linprog(
            c=np.array([1.0, 3.0]),
            A_ub=np.array([
                [-1.0, -1.0],  # meet demand
                [1.0, 0.0],    # cheap capacity
            ]),
            b_ub=np.array([-demand, self.cheap_capacity]),
            bounds=[(0.0, None), (0.0, None)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(result.message)

        x = np.asarray(result.x)
        slacks = np.asarray(result.ineqlin.residual)
        active = []
        if abs(slacks[0]) <= tol:
            active.append("demand")
        if abs(slacks[1]) <= tol:
            active.append("cheap_capacity")
        if x[0] <= tol:
            active.append("cheap_nonnegativity")
        if x[1] <= tol:
            active.append("flex_nonnegativity")

        # HiGHS reports dV/db for A_ub x <= b. Here b_demand = -demand,
        # so dV/ddemand = - marginal.
        shadow = -float(result.ineqlin.marginals[0])
        return ParametricSolution(
            demand=float(demand),
            x=x,
            objective=float(result.fun),
            demand_shadow_price=shadow,
            active_set=tuple(active),
        )

    def sweep(self, demands: np.ndarray) -> list[ParametricSolution]:
        values = np.asarray(demands, dtype=float)
        if values.ndim != 1:
            raise ValueError("demands must be one-dimensional")
        return [self.solve(float(d)) for d in values]

    @staticmethod
    def regime_changes(solutions: list[ParametricSolution]) -> list[tuple[float, tuple[str, ...]]]:
        """Return the first sampled demand at each new active-set regime."""
        out: list[tuple[float, tuple[str, ...]]] = []
        previous: tuple[str, ...] | None = None
        for solution in solutions:
            if solution.active_set != previous:
                out.append((solution.demand, solution.active_set))
                previous = solution.active_set
        return out
