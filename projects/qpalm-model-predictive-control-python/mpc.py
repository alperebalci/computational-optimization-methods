from __future__ import annotations

import numpy as np
from qpsolvers import solve_qp


def prediction_matrices(a: float, b: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    sx = np.array([a ** (k + 1) for k in range(horizon)], dtype=float)
    su = np.zeros((horizon, horizon), dtype=float)
    for k in range(horizon):
        for j in range(k + 1):
            su[k, j] = (a ** (k - j)) * b
    return sx, su


def build_mpc_qp(
    x0: float,
    horizon: int = 12,
    a: float = 1.08,
    b: float = 0.45,
    q_state: float = 2.0,
    q_terminal: float = 6.0,
    r_input: float = 0.15,
    x_max: float = 4.0,
    u_max: float = 1.4,
):
    sx, su = prediction_matrices(a, b, horizon)

    q_diag = np.full(horizon, q_state)
    q_diag[-1] = q_terminal
    qbar = np.diag(q_diag)
    rbar = r_input * np.eye(horizon)

    # qpsolvers uses 0.5*u'Pu + q'u.
    p = 2.0 * (su.T @ qbar @ su + rbar)
    q = 2.0 * su.T @ qbar @ (sx * x0)

    # Predicted-state bounds: -x_max <= sx*x0 + su*u <= x_max.
    g = np.vstack((su, -su))
    h = np.concatenate((x_max - sx * x0, x_max + sx * x0))

    lb = -u_max * np.ones(horizon)
    ub = u_max * np.ones(horizon)
    return p, q, g, h, lb, ub, sx, su


def solve_mpc(x0: float = 3.2) -> tuple[np.ndarray, np.ndarray]:
    p, q, g, h, lb, ub, sx, su = build_mpc_qp(x0)
    u = solve_qp(p, q, g, h, lb=lb, ub=ub, solver="qpalm")
    if u is None:
        raise RuntimeError("QPALM did not return a solution")
    x = sx * x0 + su @ u
    return np.asarray(u), np.asarray(x)


if __name__ == "__main__":
    x0 = 3.2
    u, x = solve_mpc(x0)

    x_max = 4.0
    violation = float(np.max(np.maximum(np.abs(x) - x_max, 0.0)))

    print(f"first control action   = {u[0]:.8f}")
    print("control trajectory     =", np.round(u, 6).tolist())
    print("predicted states       =", np.round(x, 6).tolist())
    print(f"max state violation    = {violation:.3e}")
