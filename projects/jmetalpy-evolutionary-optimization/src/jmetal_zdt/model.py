from __future__ import annotations

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT1
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations


def run_nsga2(max_evaluations: int = 2000, population_size: int = 80) -> list[tuple[float, float]]:
    if max_evaluations < population_size:
        raise ValueError("max_evaluations must be at least population_size")
    problem = ZDT1()
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables(),
            distribution_index=20,
        ),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    algorithm.run()
    front = get_non_dominated_solutions(algorithm.result())
    return [(float(s.objectives[0]), float(s.objectives[1])) for s in front]


if __name__ == "__main__":
    front = run_nsga2()
    print(f"nondominated solutions: {len(front)}")
    for row in front[:10]:
        print(row)
