from __future__ import annotations

from dataclasses import dataclass
from random import Random

import inspyred


@dataclass(frozen=True)
class Result:
    candidate: tuple[float, ...]
    fitness: float


def run(seed: int = 7, dimensions: int = 5, max_evaluations: int = 1000, population_size: int = 50) -> Result:
    if dimensions <= 0 or max_evaluations < population_size:
        raise ValueError("invalid dimensions or evaluation budget")
    rng = Random(seed)
    problem = inspyred.benchmarks.Rastrigin(dimensions)
    algorithm = inspyred.ec.ES(rng)
    algorithm.terminator = inspyred.ec.terminators.evaluation_termination
    final_population = algorithm.evolve(
        generator=problem.generator,
        evaluator=problem.evaluator,
        pop_size=population_size,
        maximize=problem.maximize,
        bounder=problem.bounder,
        max_evaluations=max_evaluations,
        mutation_rate=0.25,
    )
    best = max(final_population)
    return Result(tuple(float(v) for v in best.candidate), float(best.fitness))


if __name__ == "__main__":
    print(run())
