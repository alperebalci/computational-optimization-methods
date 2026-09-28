import cvxpy as cp
import numpy as np
import pytest

from embedded_mpc import ParametricMPC, double_integrator_config


def test_problem_is_dcp_and_dpp() -> None:
    controller = ParametricMPC(double_integrator_config(horizon=8))
    assert controller.problem.is_dcp()
    assert controller.problem.is_dpp()


def test_osqp_solution_respects_dynamics_and_bounds() -> None:
    controller = ParametricMPC(double_integrator_config(horizon=10))
    result = controller.solve([-1.0, 0.3], solver="OSQP", warm_start=False)

    assert result.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    assert result.max_dynamics_residual <= 1e-6
    assert result.max_bound_violation <= 1e-7


@pytest.mark.parametrize("solver", ["CLARABEL", "SCS"])
def test_open_source_solvers_agree_with_osqp(solver: str) -> None:
    config = double_integrator_config(horizon=8)
    reference = ParametricMPC(config).solve([-0.8, 0.2], solver="OSQP", warm_start=False)
    candidate = ParametricMPC(config).solve([-0.8, 0.2], solver=solver, warm_start=False)

    assert abs(candidate.objective - reference.objective) <= 2e-3
    assert np.allclose(candidate.first_control, reference.first_control, atol=3e-3)
    assert candidate.max_dynamics_residual <= 2e-4
    assert candidate.max_bound_violation <= 2e-4


def test_repeated_parametric_solves_update_initial_state() -> None:
    config = double_integrator_config(horizon=8)
    controller = ParametricMPC(config)

    first = controller.solve([-1.0, 0.0], solver="OSQP", warm_start=False)
    next_state = config.a @ np.array([-1.0, 0.0]) + config.b @ first.first_control
    second = controller.solve(next_state, solver="OSQP", warm_start=True)

    assert np.allclose(second.states[:, 0], next_state, atol=1e-7)
    assert second.max_dynamics_residual <= 1e-6


def test_invalid_initial_state_is_rejected() -> None:
    controller = ParametricMPC(double_integrator_config())
    with pytest.raises(ValueError, match="state bounds"):
        controller.solve([8.0, 0.0])
