from __future__ import annotations

from dataclasses import dataclass

import torch
from evotorch.algorithms.functional import pgpe, pgpe_ask, pgpe_tell


def production_fitness(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    profit = 8.0 * a + 6.0 * b + 5.0 * c
    violations = torch.stack(
        [
            torch.relu(2.0 * a + 3.0 * b - 45.0),
            torch.relu(5.0 * a + 2.0 * c - 75.0),
            torch.relu(3.0 * b + c - 50.0),
            torch.relu(-a),
            torch.relu(-b),
            torch.relu(-c),
        ],
        dim=1,
    )
    return profit - 20.0 * torch.sum(violations.square(), dim=1)


@dataclass(frozen=True)
class SearchResult:
    solution: torch.Tensor
    fitness: float


def optimize(seed: int = 7, generations: int = 20, popsize: int = 80) -> SearchResult:
    if generations <= 0 or popsize < 4:
        raise ValueError("generations must be positive and popsize must be at least 4")
    torch.manual_seed(seed)
    state = pgpe(
        objective_sense="max",
        center_init=torch.zeros(3),
        stdev_init=10.0,
        center_learning_rate=0.10,
        stdev_learning_rate=0.10,
        stdev_max_change=0.20,
        stdev_min=0.01,
        ranking_method="centered",
        optimizer="clipup",
        optimizer_config={"max_speed": 1.0},
    )
    best_x = None
    best_f = float("-inf")
    for _ in range(generations):
        population = pgpe_ask(state, popsize=popsize)
        fitnesses = production_fitness(population)
        state = pgpe_tell(state, population, fitnesses)
        index = int(torch.argmax(fitnesses))
        value = float(fitnesses[index])
        if value > best_f:
            best_f = value
            best_x = population[index].detach().clone()
    assert best_x is not None
    return SearchResult(best_x, best_f)


if __name__ == "__main__":
    result = optimize()
    print("solution:", result.solution.tolist())
    print("penalized profit:", result.fitness)
