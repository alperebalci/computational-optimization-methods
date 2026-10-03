import math

from spatial_bb import solve_bilinear_demo


def test_spatial_branch_and_bound_certifies_global_optimum():
    result = solve_bilinear_demo(tolerance=1e-6)
    assert result.x + result.y <= 1.5 + 1e-9
    assert 0.0 <= result.x <= 1.0
    assert 0.0 <= result.y <= 1.0
    assert math.isclose(result.objective, result.x * result.y, rel_tol=0, abs_tol=1e-10)
    assert math.isclose(result.objective, 0.5625, rel_tol=0, abs_tol=2e-5)
    assert result.upper_bound >= result.objective - 1e-12
    assert result.absolute_gap <= 1e-6
    assert result.explored_nodes > 0
