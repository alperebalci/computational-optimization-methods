from jmetal_zdt.model import run_nsga2


def test_nsga2_returns_nondominated_objective_pairs():
    front = run_nsga2(max_evaluations=80, population_size=20)
    assert front
    assert all(len(point) == 2 for point in front)
    assert all(point[0] >= 0 and point[1] >= 0 for point in front)
