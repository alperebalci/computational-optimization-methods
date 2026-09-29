from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


class FactoryDesignProblem(ElementwiseProblem):
    def __init__(self) -> None:
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.array([0.0, 0.5]),
            xu=np.array([1.0, 1.5]),
        )

    @staticmethod
    def metrics(x: np.ndarray) -> tuple[float, float, float]:
        automation, speed = map(float, x)
        annualized_cost = 100.0 + 80.0 * automation + 40.0 * speed**2
        emissions = 200.0 - 90.0 * automation + 60.0 * (speed - 0.8) ** 2
        throughput = 0.70 * speed + 0.50 * automation
        return annualized_cost, emissions, throughput

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        cost, emissions, throughput = self.metrics(np.asarray(x, dtype=float))
        out["F"] = [cost, emissions]
        out["G"] = [1.0 - throughput]


@dataclass(frozen=True)
class ParetoApproximation:
    decisions: np.ndarray
    objectives: np.ndarray


def solve(seed: int = 7, generations: int = 40, population_size: int = 60) -> ParetoApproximation:
    if generations <= 0 or population_size < 4:
        raise ValueError("generations must be positive and population_size must be at least 4")
    result = minimize(
        FactoryDesignProblem(),
        NSGA2(pop_size=population_size),
        ("n_gen", generations),
        seed=seed,
        verbose=False,
    )
    return ParetoApproximation(np.asarray(result.X), np.asarray(result.F))


if __name__ == "__main__":
    result = solve()
    print(f"nondominated solutions: {len(result.objectives)}")
    for x, f in zip(result.decisions[:5], result.objectives[:5]):
        print("x=", np.round(x, 4), "objectives=", np.round(f, 4))
