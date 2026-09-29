from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from niapy.algorithms.basic import DifferentialEvolution
from niapy.task import Task


@dataclass(frozen=True)
class Result:
    solution: np.ndarray
    fitness: float


def run(seed: int = 7, dimension: int = 10, max_evaluations: int = 2000, population_size: int = 40) -> Result:
    if dimension <= 0 or max_evaluations < population_size:
        raise ValueError("invalid dimension or evaluation budget")
    task = Task(problem="griewank", dimension=dimension, max_evals=max_evaluations)
    algorithm = DifferentialEvolution(population_size=population_size, seed=seed)
    best_x, best_fitness = algorithm.run(task)
    return Result(np.asarray(best_x, dtype=float), float(best_fitness))


if __name__ == "__main__":
    result = run()
    print("fitness:", result.fitness)
    print("solution:", np.round(result.solution, 5))
