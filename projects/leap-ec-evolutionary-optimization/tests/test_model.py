import numpy as np

from leap_maxones.model import run_maxones


def test_leap_pipeline_runs_and_returns_binary_genome():
    genome, fitness = run_maxones(length=12, generations=2, population_size=8)
    assert genome.shape == (12,)
    assert set(np.unique(genome)).issubset({0, 1})
    assert 0 <= fitness <= 12
