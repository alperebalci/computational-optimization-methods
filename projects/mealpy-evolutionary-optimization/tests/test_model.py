import numpy as np

from mealpy_production.model import objective, solve


def test_objective_rewards_feasible_profitable_plan():
    assert objective([10, 10, 10]) < objective([0, 0, 0])
    assert np.isfinite(objective([100, 100, 100]))


def test_mealpy_ga_runs():
    result = solve(seed=3, epochs=2, population_size=10)
    assert result.quantities.shape == (3,)
    assert np.isfinite(result.objective)
