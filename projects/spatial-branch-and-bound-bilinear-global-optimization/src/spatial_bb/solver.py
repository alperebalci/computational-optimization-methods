"""Spatial branch-and-bound with McCormick envelopes.

The benchmark solves

    maximize x*y
    s.t. x + y <= 1.5
         0 <= x,y <= 1

The global optimum is 0.5625 at x=y=0.75, but the solver does not use that
closed form. Each node solves an LP McCormick relaxation to obtain a valid
upper bound and branches on the widest variable interval.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class GlobalResult:
    x: float
    y: float
    objective: float
    upper_bound: float
    absolute_gap: float
    explored_nodes: int


@dataclass(frozen=True)
class _Box:
    lx: float
    ux: float
    ly: float
    uy: float


def _mccormick_upper_bound(box: _Box) -> tuple[float, np.ndarray] | None:
    """Return a valid max(x*y) upper bound on one box."""
    lx, ux, ly, uy = box.lx, box.ux, box.ly, box.uy

    # Variables z = [x, y, w]. linprog minimizes -w.
    a_ub = [
        [1.0, 1.0, 0.0],  # x + y <= 1.5
        [ly, lx, -1.0],    # w >= lx*y + ly*x - lx*ly
        [uy, ux, -1.0],    # w >= ux*y + uy*x - ux*uy
        [-ly, -ux, 1.0],   # w <= ux*y + ly*x - ux*ly
        [-uy, -lx, 1.0],   # w <= lx*y + uy*x - lx*uy
    ]
    b_ub = [
        1.5,
        lx * ly,
        ux * uy,
        -ux * ly,
        -lx * uy,
    ]
    result = linprog(
        c=np.array([0.0, 0.0, -1.0]),
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=[(lx, ux), (ly, uy), (0.0, 1.0)],
        method="highs",
    )
    if not result.success:
        return None
    return float(result.x[2]), np.asarray(result.x)


def _local_incumbent() -> tuple[float, float, float]:
    """Find a feasible incumbent; validity does not depend on local optimality."""
    result = minimize(
        lambda z: -float(z[0] * z[1]),
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints=[{"type": "ineq", "fun": lambda z: 1.5 - z[0] - z[1]}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Local incumbent search failed: {result.message}")
    x, y = map(float, result.x)
    return x, y, x * y


def _split(box: _Box) -> tuple[_Box, _Box]:
    if box.ux - box.lx >= box.uy - box.ly:
        mid = 0.5 * (box.lx + box.ux)
        return (
            _Box(box.lx, mid, box.ly, box.uy),
            _Box(mid, box.ux, box.ly, box.uy),
        )
    mid = 0.5 * (box.ly + box.uy)
    return (
        _Box(box.lx, box.ux, box.ly, mid),
        _Box(box.lx, box.ux, mid, box.uy),
    )


def solve_bilinear_demo(tolerance: float = 1e-6, max_nodes: int = 50_000) -> GlobalResult:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    incumbent_x, incumbent_y, incumbent = _local_incumbent()
    root = _Box(0.0, 1.0, 0.0, 1.0)
    root_relax = _mccormick_upper_bound(root)
    if root_relax is None:
        raise RuntimeError("Root relaxation is infeasible")

    counter = 0
    queue: list[tuple[float, int, _Box]] = []
    heapq.heappush(queue, (-root_relax[0], counter, root))
    explored = 0

    while queue and explored < max_nodes:
        best_upper = -queue[0][0]
        if best_upper - incumbent <= tolerance:
            break

        neg_upper, _, box = heapq.heappop(queue)
        upper = -neg_upper
        if upper <= incumbent + tolerance:
            continue

        explored += 1
        for child in _split(box):
            relaxation = _mccormick_upper_bound(child)
            if relaxation is None:
                continue
            child_upper, point = relaxation

            # The relaxation's (x,y) coordinates satisfy all original linear
            # constraints, so x*y is a valid feasible lower bound.
            x, y = float(point[0]), float(point[1])
            feasible_value = x * y
            if feasible_value > incumbent:
                incumbent_x, incumbent_y, incumbent = x, y, feasible_value

            if child_upper > incumbent + tolerance:
                counter += 1
                heapq.heappush(queue, (-child_upper, counter, child))

    if queue:
        global_upper = max(incumbent, -queue[0][0])
    else:
        global_upper = incumbent

    gap = global_upper - incumbent
    if gap > tolerance and explored >= max_nodes:
        raise RuntimeError("Node limit reached before requested global gap closed")

    return GlobalResult(
        x=incumbent_x,
        y=incumbent_y,
        objective=incumbent,
        upper_bound=global_upper,
        absolute_gap=gap,
        explored_nodes=explored,
    )
