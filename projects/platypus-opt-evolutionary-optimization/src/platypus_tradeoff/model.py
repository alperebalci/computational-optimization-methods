from __future__ import annotations

from platypus import NSGAII, Problem, Real, nondominated


def objectives(x):
    a, b = x
    structural_cost = a * a + b * b
    performance_loss = (a - 2.0) ** 2 + (b - 1.0) ** 2
    return [structural_cost, performance_loss]


def run(nfe: int = 1000, population_size: int = 60) -> list[tuple[float, float]]:
    if nfe <= 0 or population_size < 2:
        raise ValueError("nfe must be positive and population_size must be at least 2")
    problem = Problem(2, 2)
    problem.types[:] = [Real(-3.0, 3.0), Real(-3.0, 3.0)]
    problem.function = objectives
    algorithm = NSGAII(problem, population_size=population_size)
    algorithm.run(nfe)
    return [tuple(map(float, solution.objectives)) for solution in nondominated(algorithm.result)]


if __name__ == "__main__":
    front = run()
    print(f"nondominated solutions: {len(front)}")
    for point in front[:10]:
        print(point)
