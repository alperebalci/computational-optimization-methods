"""Reusable parametric quadratic-program MPC model."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

SolverName = Literal["OSQP", "CLARABEL", "SCS"]


@dataclass(frozen=True)
class MPCConfig:
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    q: NDArray[np.float64]
    r: NDArray[np.float64]
    q_terminal: NDArray[np.float64]
    target: NDArray[np.float64]
    state_lower: NDArray[np.float64]
    state_upper: NDArray[np.float64]
    input_lower: NDArray[np.float64]
    input_upper: NDArray[np.float64]
    horizon: int

    @classmethod
    def from_arrays(
        cls,
        a: ArrayLike,
        b: ArrayLike,
        q: ArrayLike,
        r: ArrayLike,
        q_terminal: ArrayLike,
        target: ArrayLike,
        state_lower: ArrayLike,
        state_upper: ArrayLike,
        input_lower: ArrayLike,
        input_upper: ArrayLike,
        horizon: int,
    ) -> MPCConfig:
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1]:
            raise ValueError("a must be square")
        nx = a_arr.shape[0]
        if b_arr.ndim != 2 or b_arr.shape[0] != nx:
            raise ValueError("b must have one row per state")
        nu = b_arr.shape[1]

        q_arr = np.asarray(q, dtype=float)
        r_arr = np.asarray(r, dtype=float)
        qf_arr = np.asarray(q_terminal, dtype=float)
        target_arr = np.asarray(target, dtype=float)
        xl = np.asarray(state_lower, dtype=float)
        xu = np.asarray(state_upper, dtype=float)
        ul = np.asarray(input_lower, dtype=float)
        uu = np.asarray(input_upper, dtype=float)

        if q_arr.shape != (nx, nx) or qf_arr.shape != (nx, nx):
            raise ValueError("state cost matrices have invalid shape")
        if r_arr.shape != (nu, nu):
            raise ValueError("input cost matrix has invalid shape")
        if target_arr.shape != (nx,) or xl.shape != (nx,) or xu.shape != (nx,):
            raise ValueError("state vectors have invalid shape")
        if ul.shape != (nu,) or uu.shape != (nu,):
            raise ValueError("input bounds have invalid shape")
        if np.any(xl > xu) or np.any(ul > uu):
            raise ValueError("lower bounds must not exceed upper bounds")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        for name, arr in (
            ("a", a_arr),
            ("b", b_arr),
            ("q", q_arr),
            ("r", r_arr),
            ("q_terminal", qf_arr),
            ("target", target_arr),
            ("state_lower", xl),
            ("state_upper", xu),
            ("input_lower", ul),
            ("input_upper", uu),
        ):
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")

        if np.min(np.linalg.eigvalsh(0.5 * (q_arr + q_arr.T))) < -1e-10:
            raise ValueError("q must be positive semidefinite")
        if np.min(np.linalg.eigvalsh(0.5 * (qf_arr + qf_arr.T))) < -1e-10:
            raise ValueError("q_terminal must be positive semidefinite")
        if np.min(np.linalg.eigvalsh(0.5 * (r_arr + r_arr.T))) <= 0.0:
            raise ValueError("r must be positive definite")

        return cls(
            a_arr,
            b_arr,
            q_arr,
            r_arr,
            qf_arr,
            target_arr,
            xl,
            xu,
            ul,
            uu,
            int(horizon),
        )

    @property
    def nx(self) -> int:
        return int(self.a.shape[0])

    @property
    def nu(self) -> int:
        return int(self.b.shape[1])


@dataclass(frozen=True)
class MPCResult:
    solver: str
    status: str
    objective: float
    first_control: NDArray[np.float64]
    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    max_dynamics_residual: float
    max_bound_violation: float
    solver_seconds: float | None
    wall_seconds: float


class ParametricMPC:
    """Compile one convex MPC QP and repeatedly update only the initial state."""

    def __init__(self, config: MPCConfig) -> None:
        self.config = config
        n = config.horizon
        self.x0 = cp.Parameter(config.nx, name="x0")
        self.x = cp.Variable((config.nx, n + 1), name="x")
        self.u = cp.Variable((config.nu, n), name="u")

        objective = 0
        constraints: list[cp.Constraint] = [self.x[:, 0] == self.x0]
        for t in range(n):
            deviation = self.x[:, t] - config.target
            objective += cp.quad_form(deviation, config.q)
            objective += cp.quad_form(self.u[:, t], config.r)
            constraints.extend(
                [
                    self.x[:, t + 1] == config.a @ self.x[:, t] + config.b @ self.u[:, t],
                    config.state_lower <= self.x[:, t],
                    self.x[:, t] <= config.state_upper,
                    config.input_lower <= self.u[:, t],
                    self.u[:, t] <= config.input_upper,
                ]
            )
        terminal_deviation = self.x[:, n] - config.target
        objective += cp.quad_form(terminal_deviation, config.q_terminal)
        constraints.extend(
            [
                config.state_lower <= self.x[:, n],
                self.x[:, n] <= config.state_upper,
            ]
        )
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        if not self.problem.is_dcp():
            raise ValueError("constructed MPC problem is not DCP")

    def solve(
        self,
        initial_state: ArrayLike,
        *,
        solver: SolverName = "OSQP",
        warm_start: bool = True,
    ) -> MPCResult:
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape != (self.config.nx,) or np.any(~np.isfinite(initial)):
            raise ValueError("initial_state has invalid shape or values")
        if np.any(initial < self.config.state_lower - 1e-12) or np.any(
            initial > self.config.state_upper + 1e-12
        ):
            raise ValueError("initial_state violates declared state bounds")
        if solver not in {"OSQP", "CLARABEL", "SCS"}:
            raise ValueError("unsupported solver")

        self.x0.value = initial
        options: dict[str, object] = {"warm_start": warm_start, "verbose": False}
        if solver == "OSQP":
            options.update({"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 100_000})
        elif solver == "SCS":
            options.update({"eps": 1e-5, "max_iters": 50_000})
        else:
            options.update({"tol_gap_abs": 1e-8, "tol_feas": 1e-8, "max_iter": 200})

        start = perf_counter()
        value = self.problem.solve(solver=solver, **options)
        wall = perf_counter() - start
        if self.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"{solver} failed with status {self.problem.status}")
        if self.x.value is None or self.u.value is None or value is None:
            raise RuntimeError(f"{solver} returned no primal solution")

        states = np.asarray(self.x.value, dtype=float)
        controls = np.asarray(self.u.value, dtype=float)
        predicted = (
            self.config.a @ states[:, :-1]
            + self.config.b @ controls
        )
        dynamics = float(np.max(np.abs(states[:, 1:] - predicted)))
        bound_violation = max(
            float(np.max(self.config.state_lower[:, None] - states)),
            float(np.max(states - self.config.state_upper[:, None])),
            float(np.max(self.config.input_lower[:, None] - controls)),
            float(np.max(controls - self.config.input_upper[:, None])),
            0.0,
        )
        stats_time = self.problem.solver_stats.solve_time
        return MPCResult(
            solver=solver,
            status=str(self.problem.status),
            objective=float(value),
            first_control=controls[:, 0].copy(),
            states=states,
            controls=controls,
            max_dynamics_residual=dynamics,
            max_bound_violation=bound_violation,
            solver_seconds=None if stats_time is None else float(stats_time),
            wall_seconds=float(wall),
        )


def double_integrator_config(horizon: int = 12, dt: float = 0.2) -> MPCConfig:
    """Return a stable benchmark configuration for a constrained double integrator."""

    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.5 * dt**2], [dt]])
    return MPCConfig.from_arrays(
        a=a,
        b=b,
        q=np.diag([5.0, 1.0]),
        r=np.array([[0.2]]),
        q_terminal=np.diag([12.0, 2.0]),
        target=[1.5, 0.0],
        state_lower=[-5.0, -3.0],
        state_upper=[5.0, 3.0],
        input_lower=[-1.5],
        input_upper=[1.5],
        horizon=horizon,
    )
