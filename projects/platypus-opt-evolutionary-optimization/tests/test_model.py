from platypus_tradeoff.model import objectives, run


def test_objectives_express_tradeoff():
    assert objectives([0.0, 0.0]) == [0.0, 5.0]
    assert objectives([2.0, 1.0]) == [5.0, 0.0]


def test_platypus_nsga2_runs():
    front = run(nfe=80, population_size=20)
    assert front
    assert all(len(point) == 2 for point in front)
