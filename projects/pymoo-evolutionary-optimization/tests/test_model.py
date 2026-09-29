import numpy as np

from pymoo_factory.model import FactoryDesignProblem, solve


def test_factory_metrics_are_interpretable():
    cost, emissions, throughput = FactoryDesignProblem.metrics(np.array([0.6, 1.0]))
    assert cost > 0
    assert emissions > 0
    assert throughput == 1.0


def test_nsga2_returns_two_objective_front():
    result = solve(seed=3, generations=4, population_size=12)
    assert result.decisions.ndim == 2
    assert result.objectives.ndim == 2
    assert result.objectives.shape[1] == 2
    assert len(result.objectives) > 0
