"""Transparent infeasibility diagnostics for small linear models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class LinearInequalityModel:
    """Model A x <= b with ordinary variable bounds."""

    a_ub: np.ndarray
    b_ub: np.ndarray
    bounds: tuple[tuple[float | None, float | None], ...]

    def __post_init__(self) -> None:
        a = np.asarray(self.a_ub, dtype=float)
        b = np.asarray(self.b_ub, dtype=float)
        if a.ndim != 2 or b.shape != (a.shape[0],):
            raise ValueError("A and b dimensions are inconsistent")
        if len(self.bounds) != a.shape[1]:
            raise ValueError("bounds must match variable count")
        object.__setattr__(self, "a_ub", a)
        object.__setattr__(self, "b_ub", b)

    @property
    def n_constraints(self) -> int:
        return self.a_ub.shape[0]

    def feasible(self, indices: tuple[int, ...] | list[int] | None = None) -> bool:
        if indices is None:
            indices = list(range(self.n_constraints))
        idx = np.asarray(indices, dtype=int)
        a = self.a_ub[idx] if idx.size else None
        b = self.b_ub[idx] if idx.size else None
        result = linprog(
            np.zeros(self.a_ub.shape[1]),
            A_ub=a,
            b_ub=b,
            bounds=self.bounds,
            method="highs",
        )
        return bool(result.success)


@dataclass(frozen=True)
class RelaxationResult:
    x: np.ndarray
    slacks: np.ndarray
    weighted_violation: float


def minimal_infeasible_subset(model: LinearInequalityModel) -> tuple[int, ...]:
    """Deletion filter returning an inclusion-minimal infeasible subsystem."""
    if model.feasible():
        return ()
    active = list(range(model.n_constraints))
    changed = True
    while changed:
        changed = False
        for i in active.copy():
            candidate = [j for j in active if j != i]
            if not model.feasible(candidate):
                active = candidate
                changed = True
    return tuple(active)


def minimum_constraint_deletion(model: LinearInequalityModel) -> tuple[int, ...]:
    """Exact minimum-cardinality row deletion for small diagnostic models."""
    if model.feasible():
        return ()
    indices = range(model.n_constraints)
    for size in range(1, model.n_constraints + 1):
        for deleted in combinations(indices, size):
            kept = [i for i in indices if i not in deleted]
            if model.feasible(kept):
                return tuple(deleted)
    raise RuntimeError("No feasible repair found even after deleting all rows")


def l1_feasibility_relaxation(
    model: LinearInequalityModel,
    weights: np.ndarray | None = None,
) -> RelaxationResult:
    """Minimize weighted positive row violations in A x <= b."""
    m, n = model.a_ub.shape
    if weights is None:
        w = np.ones(m)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (m,) or np.any(w < 0):
            raise ValueError("weights must be a nonnegative vector with one entry per row")

    # Variables [x, s], s >= 0 and A x - s <= b.
    c = np.concatenate([np.zeros(n), w])
    a = np.hstack([model.a_ub, -np.eye(m)])
    bounds = tuple(model.bounds) + tuple((0.0, None) for _ in range(m))
    result = linprog(c, A_ub=a, b_ub=model.b_ub, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Feasibility relaxation failed: {result.message}")
    return RelaxationResult(
        x=np.asarray(result.x[:n]),
        slacks=np.asarray(result.x[n:]),
        weighted_violation=float(result.fun),
    )
