import numpy as np
import pytest

from halpern_pdhg import LPProblem, highs_reference, make_demo_problem, solve_first_order, spectral_norm


def tiny_lp() -> LPProblem:
    return LPProblem.from_arrays(
        a=[[1.0, 1.0]],
        b=[1.0],
        c=[1.0, 2.0],
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
    )


def test_spectral_norm_matches_numpy() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, -1.0]])
    assert np.isclose(spectral_norm(matrix), np.linalg.norm(matrix, 2), rtol=1e-7)


@pytest.mark.parametrize("mode", ["pdhg", "halpern", "restarted_halpern"])
def test_first_order_modes_approach_tiny_lp_optimum(mode: str) -> None:
    problem = tiny_lp()
    reference, _ = highs_reference(problem)
    result = solve_first_order(
        problem,
        mode=mode,
        max_iter=20_000,
        tolerance=2e-4,
        restart_period=500,
    )

    assert result.primal_residual <= 2.5e-4
    assert result.projected_stationarity <= 2.5e-4
    assert abs(result.objective - reference) <= 5e-4
    assert np.all(result.x >= problem.lower - 1e-10)
    assert np.all(result.x <= problem.upper + 1e-10)


def test_generated_problem_is_feasible_and_reference_solves() -> None:
    problem = make_demo_problem(seed=3, rows=5, columns=16)
    objective, x = highs_reference(problem)

    assert np.isfinite(objective)
    assert np.linalg.norm(problem.a @ x - problem.b) <= 1e-7


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        solve_first_order(tiny_lp(), backend="invalid")
