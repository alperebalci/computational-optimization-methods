import numpy as np

from niapy_griewank.model import run


def test_differential_evolution_runs():
    result = run(seed=4, dimension=4, max_evaluations=80, population_size=20)
    assert result.solution.shape == (4,)
    assert np.isfinite(result.fitness)
    assert result.fitness >= 0
