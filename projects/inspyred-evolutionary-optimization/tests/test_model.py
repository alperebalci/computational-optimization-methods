import math

from inspyred_rastrigin.model import run


def test_inspyred_es_runs_reproducibly():
    a = run(seed=5, dimensions=3, max_evaluations=60, population_size=20)
    b = run(seed=5, dimensions=3, max_evaluations=60, population_size=20)
    assert len(a.candidate) == 3
    assert math.isfinite(a.fitness)
    assert a == b
