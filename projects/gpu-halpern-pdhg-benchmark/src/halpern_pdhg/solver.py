"""Auditable PDHG / Halpern-style fixed-point solvers for box-constrained equality LPs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

Mode = Literal["pdhg", "halpern", "restarted_halpern", "adaptive_restarted_halpern"]


@dataclass(frozen=True)
class LPProblem:
    """Linear program min c^T x subject to A x = b and lower <= x <= upper."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    @classmethod
    def from_arrays(
        cls,
        a: ArrayLike,
        b: ArrayLike,
        c: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> LPProblem:
        matrix = np.asarray(a, dtype=float)
        rhs = np.asarray(b, dtype=float)
        cost = np.asarray(c, dtype=float)
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)

        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError("a must be a non-empty matrix")
        m, n = matrix.shape
        for name, arr, shape in (
            ("b", rhs, (m,)),
            ("c", cost, (n,)),
            ("lower", lo, (n,)),
            ("upper", hi, (n,)),
        ):
            if arr.shape != shape or np.any(~np.isfinite(arr)):
                raise ValueError(f"{name} has invalid shape or non-finite values")
        if np.any(lo > hi):
            raise ValueError("lower must be <= upper")
        return cls(matrix, rhs, cost, lo, hi)


@dataclass(frozen=True)
class SolveResult:
    mode: str
    backend: str
    x: NDArray[np.float64]
    objective: float
    primal_residual: float
    projected_stationarity: float
    iterations: int
    converged: bool
    wall_seconds: float
    restarts: int


def make_demo_problem(seed: int = 12, rows: int = 12, columns: int = 40) -> LPProblem:
    """Generate a well-scaled feasible LP for correctness and algorithm comparisons."""

    if rows < 1 or columns <= rows:
        raise ValueError("require columns > rows >= 1")
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(rows, columns)) / np.sqrt(columns)
    feasible = rng.uniform(0.15, 0.85, size=columns)
    b = a @ feasible
    c = rng.normal(size=columns)
    return LPProblem.from_arrays(a, b, c, np.zeros(columns), np.ones(columns))


def spectral_norm(a: ArrayLike, iterations: int = 80) -> float:
    """Estimate ||A||_2 by deterministic power iteration on A^T A."""

    matrix = np.asarray(a, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("a must be two-dimensional")
    vector = np.ones(matrix.shape[1], dtype=float)
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        candidate = matrix.T @ (matrix @ vector)
        norm = float(np.linalg.norm(candidate))
        if norm == 0.0:
            return 0.0
        vector = candidate / norm
    return float(np.linalg.norm(matrix @ vector))


def highs_reference(problem: LPProblem) -> tuple[float, NDArray[np.float64]]:
    """Solve the same LP with SciPy/HiGHS as an independent reference."""

    result = linprog(
        c=problem.c,
        A_eq=problem.a,
        b_eq=problem.b,
        bounds=list(zip(problem.lower, problem.upper, strict=True)),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"HiGHS reference failed: {result.message}")
    return float(result.fun), np.asarray(result.x, dtype=float)


def _load_backend(name: str) -> tuple[Any, Any]:
    if name == "numpy":
        return np, lambda value: np.asarray(value, dtype=float)
    if name != "cupy":
        raise ValueError("backend must be 'numpy' or 'cupy'")

    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CuPy backend requested but CuPy is not installed") from exc
    return cp, cp.asarray


def _to_numpy(xp: Any, value: Any) -> NDArray[np.float64]:
    if xp is np:
        return np.asarray(value, dtype=float)
    return np.asarray(xp.asnumpy(value), dtype=float)  # pragma: no cover


def _diagnostics(
    problem: LPProblem,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[float, float, float]:
    primal = float(np.linalg.norm(problem.a @ x - problem.b))
    gradient = problem.c + problem.a.T @ y
    projected = x - np.clip(x - gradient, problem.lower, problem.upper)
    stationarity = float(np.linalg.norm(projected))
    objective = float(problem.c @ x)
    return objective, primal, stationarity


def solve_first_order(
    problem: LPProblem,
    *,
    mode: Mode = "pdhg",
    backend: str = "numpy",
    max_iter: int = 10_000,
    tolerance: float = 1e-5,
    restart_period: int = 500,
    check_every: int = 25,
    restart_ratio: float = 0.9,
    min_restart_iterations: int = 100,
) -> SolveResult:
    """Run PDHG, Halpern PDHG, or a periodic-restart Halpern research baseline.

    The base PDHG fixed-point operator is

        x+ = proj_[l,u](x - tau(c + A^T y))
        y+ = y + sigma(A(2x+ - x) - b)

    with tau*sigma*||A||^2 < 1. Halpern modes apply
    z_{k+1} = alpha_k * anchor + (1-alpha_k) * T(z_k).

    The periodic restart rule is intentionally simple and is not cuPDLPx's restart criterion.
    """

    if mode not in {"pdhg", "halpern", "restarted_halpern", "adaptive_restarted_halpern"}:
        raise ValueError("unknown mode")
    if max_iter < 1 or check_every < 1:
        raise ValueError("iteration counts must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if restart_period < 2:
        raise ValueError("restart_period must be at least two")
    if not 0.0 < restart_ratio < 1.0:
        raise ValueError("restart_ratio must lie in (0, 1)")
    if min_restart_iterations < 1:
        raise ValueError("min_restart_iterations must be positive")

    norm_a = spectral_norm(problem.a)
    if norm_a <= 0.0:
        raise ValueError("constraint matrix must have nonzero spectral norm")
    tau = 0.9 / norm_a
    sigma = 0.9 / norm_a

    xp, asarray = _load_backend(backend)
    a = asarray(problem.a)
    b = asarray(problem.b)
    c = asarray(problem.c)
    lower = asarray(problem.lower)
    upper = asarray(problem.upper)

    x = 0.5 * (lower + upper)
    y = xp.zeros(a.shape[0], dtype=a.dtype)
    anchor_x = x.copy()
    anchor_y = y.copy()
    local_iteration = 0
    converged = False
    restarts = 0
    best_merit = float("inf")

    start = perf_counter()
    final_iteration = max_iter
    for iteration in range(1, max_iter + 1):
        x_operator = xp.clip(x - tau * (c + a.T @ y), lower, upper)
        x_bar = 2.0 * x_operator - x
        y_operator = y + sigma * (a @ x_bar - b)

        if mode == "pdhg":
            x_next = x_operator
            y_next = y_operator
        else:
            if mode == "restarted_halpern" and local_iteration >= restart_period:
                anchor_x = x.copy()
                anchor_y = y.copy()
                local_iteration = 0
                restarts += 1
            alpha = 1.0 / (local_iteration + 2.0)
            x_next = alpha * anchor_x + (1.0 - alpha) * x_operator
            y_next = alpha * anchor_y + (1.0 - alpha) * y_operator
            local_iteration += 1

        x, y = x_next, y_next

        if iteration % check_every == 0 or iteration == max_iter:
            x_cpu = _to_numpy(xp, x)
            y_cpu = _to_numpy(xp, y)
            _, primal, stationarity = _diagnostics(problem, x_cpu, y_cpu)
            merit = max(primal, stationarity)
            if mode == "adaptive_restarted_halpern":
                if merit <= restart_ratio * best_merit:
                    best_merit = merit
                elif local_iteration >= min_restart_iterations:
                    anchor_x = x.copy()
                    anchor_y = y.copy()
                    local_iteration = 0
                    best_merit = merit
                    restarts += 1
            if merit <= tolerance:
                converged = True
                final_iteration = iteration
                break

    wall = perf_counter() - start
    x_cpu = _to_numpy(xp, x)
    y_cpu = _to_numpy(xp, y)
    objective, primal, stationarity = _diagnostics(problem, x_cpu, y_cpu)

    return SolveResult(
        mode=mode,
        backend=backend,
        x=x_cpu,
        objective=objective,
        primal_residual=primal,
        projected_stationarity=stationarity,
        iterations=final_iteration,
        converged=converged,
        wall_seconds=float(wall),
        restarts=restarts,
    )
