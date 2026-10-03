import numpy as np
from parametric_lp import ParametricProductionLP


def test_piecewise_linear_value_function():
    model = ParametricProductionLP(cheap_capacity=4.0)
    low = model.solve(3.0)
    high = model.solve(6.0)
    assert np.isclose(low.objective, 3.0)
    assert np.allclose(low.x, [3.0, 0.0])
    assert np.isclose(high.objective, 10.0)
    assert np.allclose(high.x, [4.0, 2.0])


def test_shadow_price_matches_local_value_derivative_away_from_breakpoint():
    model = ParametricProductionLP(cheap_capacity=4.0)
    for demand, expected in [(2.0, 1.0), (6.0, 3.0)]:
        solution = model.solve(demand)
        h = 1e-5
        derivative = (
            model.solve(demand + h).objective - model.solve(demand - h).objective
        ) / (2.0 * h)
        assert np.isclose(solution.demand_shadow_price, expected, atol=1e-8)
        assert np.isclose(solution.demand_shadow_price, derivative, atol=1e-5)


def test_active_set_changes_when_cheap_capacity_binds():
    model = ParametricProductionLP(cheap_capacity=4.0)
    solutions = model.sweep(np.array([1.0, 3.0, 4.5, 6.0]))
    assert "cheap_capacity" not in solutions[1].active_set
    assert "cheap_capacity" in solutions[2].active_set
    assert solutions[1].demand_shadow_price < solutions[2].demand_shadow_price
