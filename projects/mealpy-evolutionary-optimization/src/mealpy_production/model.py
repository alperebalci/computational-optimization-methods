from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mealpy import FloatVar, GA


def objective(solution) -> float:
    x = np.asarray(solution, dtype=float)
    contribution = 14.0 * x[0] + 11.0 * x[1] + 9.0 * x[2]
    labor_violation = max(0.0, 2.0 * x[0] + x[1] + 1.5 * x[2] - 100.0)
    machine_violation = max(0.0, x[0] + 2.0 * x[1] + x[2] - 90.0)
    return float(-contribution + 100.0 * (labor_violation**2 + machine_violation**2))


@dataclass(frozen=True)
class ProductionResult:
    quantities: np.ndarray
    objective: float


def solve(seed: int = 7, epochs: int = 40, population_size: int = 30) -> ProductionResult:
    if epochs <= 0 or population_size < 5:
        raise ValueError("epochs must be positive and population_size must be at least 5")
    problem = {
        "obj_func": objective,
        "bounds": FloatVar(lb=(0.0, 0.0, 0.0), ub=(60.0, 60.0, 60.0), name="production"),
        "minmax": "min",
        "name": "production-mix",
        "log_to": None,
    }
    model = GA.OriginalGA(epoch=epochs, pop_size=population_size, pc=0.9, pm=0.05)
    best = model.solve(problem=problem, seed=seed)
    return ProductionResult(np.asarray(best.solution, dtype=float), float(best.target.fitness))


if __name__ == "__main__":
    result = solve()
    print("quantities:", np.round(result.quantities, 3))
    print("objective:", result.objective)
