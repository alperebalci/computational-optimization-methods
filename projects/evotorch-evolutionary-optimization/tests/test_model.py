import torch

from evotorch_production.model import optimize, production_fitness


def test_vectorized_fitness_shape_and_feasible_profit():
    points = torch.tensor([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
    values = production_fitness(points)
    assert values.shape == (2,)
    assert values[0] > values[1]


def test_pgpe_runs_end_to_end():
    result = optimize(seed=2, generations=2, popsize=20)
    assert result.solution.shape == (3,)
    assert torch.isfinite(torch.tensor(result.fitness))
