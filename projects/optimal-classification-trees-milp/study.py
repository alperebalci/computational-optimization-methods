"""Optimal fixed-depth trees over a train-only finite split catalog (HiGHS MILP)."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def validate(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite matrix")
    if y is not None:
        y = np.asarray(y)
        if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
            raise ValueError("binary labels with one value per row required")
    return X, y


def catalog(X):
    """Use training data only. Constant features need no genuine split."""
    X, _ = validate(X)
    splits = [
        (j, float(a / 2 + b / 2))
        for j in range(X.shape[1])
        for a, b in zip(np.unique(X[:, j])[:-1], np.unique(X[:, j])[1:], strict=False)
    ]
    return splits or [(0, float(X[0, 0]))]


def path(leaf, depth):
    node = 0
    for bit in f"{leaf:0{depth}b}":
        right = int(bit)
        yield node, right
        node = 2 * node + 1 + right


def predict(model, X):
    X, _ = validate(X)
    if X.shape[1] != model["features"]:
        raise ValueError("feature count mismatch")
    B = len(model["splits"])
    result = []
    for row in X:
        node = 0
        while node < B:
            j, t = model["splits"][node]
            node = 2 * node + 1 + int(row[j] > t)
        result.append(model["labels"][node - B])
    return np.asarray(result)


def fit(X, y, depth=2, time_limit=30.0):
    X, y = validate(X, y)
    if depth not in (1, 2, 3) or not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("depth must be 1..3; positive finite time limit required")
    C = catalog(X)
    n, K, B, L = len(X), len(C), 2**depth - 1, 2**depth
    # Split choices a[b,k], leaf routes z[i,l], leaf label p[l,c], errors e[i].
    az, ap, ae = B * K, B * K + n * L, B * K + n * L + 2 * L
    N = ae + n
    rows, cols, vals, lb, ub = [], [], [], [], []

    def add(terms, low=-np.inf, high=np.inf):
        r = len(lb)
        for col, value in terms:
            rows.append(r)
            cols.append(col)
            vals.append(value)
        lb.append(low)
        ub.append(high)

    for b in range(B):
        add([(b * K + k, 1) for k in range(K)], 1, 1)
    for local_value in range(L):
        add([(ap + 2 * local_value + c, 1) for c in (0, 1)], 1, 1)
    for i in range(n):
        add([(az + i * L + local_value, 1) for local_value in range(L)], 1, 1)
        for local_value in range(L):
            for b, right in path(local_value, depth):
                consistent = [k for k, (j, t) in enumerate(C) if int(X[i, j] > t) == right]
                add([(az + i * L + local_value, 1)] + [(b * K + k, -1) for k in consistent], high=0)
            add(
                [
                    (az + i * L + local_value, 1),
                    (ap + 2 * local_value + int(y[i]), -1),
                    (ae + i, -1),
                ],
                high=0,
            )
    A = coo_matrix((vals, (rows, cols)), shape=(len(lb), N)).tocsc()
    c = np.zeros(N)
    c[ae:] = 1 / n
    start = time.perf_counter()
    r = milp(
        c,
        integrality=np.r_[np.ones(ae), np.zeros(n)],
        bounds=Bounds(np.zeros(N), np.ones(N)),
        constraints=LinearConstraint(A, lb, ub),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if r.x is None:
        raise RuntimeError(f"No tree incumbent: {r.message}")
    m = {
        "features": X.shape[1],
        "depth": depth,
        "splits": [C[int(np.argmax(r.x[b * K : (b + 1) * K]))] for b in range(B)],
        "labels": [
            int(np.argmax(r.x[ap + 2 * local_value : ap + 2 * local_value + 2]))
            for local_value in range(L)
        ],
        "status": int(r.status),
        "proven_optimal": r.status == 0,
        "mip_gap": float(r.mip_gap),
        "dual_bound": float(r.mip_dual_bound),
        "train_error": float(np.mean(predict_raw(r.x, C, B, K, ap, L, X, depth) != y)),
        "seconds": time.perf_counter() - start,
        "catalog_size": K,
    }
    if abs(m["train_error"] - r.fun) > 1e-6:
        raise RuntimeError("Decoded tree disagrees with MILP objective")
    return m


def predict_raw(v, C, B, K, ap, L, X, depth):
    m = {
        "features": X.shape[1],
        "splits": [C[int(np.argmax(v[b * K : (b + 1) * K]))] for b in range(B)],
        "labels": [
            int(np.argmax(v[ap + 2 * local_value : ap + 2 * local_value + 2]))
            for local_value in range(L)
        ],
    }
    return predict(m, X)


def exhaustive_error(X, y, depth=2):
    X, y = validate(X, y)
    C = catalog(X)
    B, L = 2**depth - 1, 2**depth
    if len(C) ** B > 100000:
        raise ValueError("Exhaustive oracle limited to 100000 split structures")
    best = 1.0
    for choices in itertools.product(C, repeat=B):
        counts = np.zeros((L, 2), dtype=int)
        for row, label in zip(X, y, strict=False):
            node = 0
            while node < B:
                j, t = choices[node]
                node = 2 * node + 1 + int(row[j] > t)
            counts[node - B, int(label)] += 1
        best = min(best, float(counts.min(axis=1).sum() / len(X)))
    return best


def benchmark():
    from sklearn.tree import DecisionTreeClassifier

    records = []
    for seed in (101, 102, 103):
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 2, (48, 4))
        y = X[:, 0] ^ X[:, 1]
        Xt = rng.integers(0, 2, (400, 4))
        yt = Xt[:, 0] ^ Xt[:, 1]
        m = fit(X, y)
        cart = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(X, y)
        records.append(
            {
                "seed": seed,
                "optimal_tree": m,
                "test_error": float(np.mean(predict(m, Xt) != yt)),
                "cart_train_error": float(np.mean(cart.predict(X) != y)),
                "cart_test_error": float(np.mean(cart.predict(Xt) != yt)),
            }
        )
    return {
        "scope": "binary XOR smoke; optimality restricted to fixed catalog and depth",
        "results": records,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results.json")
    args = p.parse_args()
    Path(args.output).write_text(json.dumps(benchmark(), indent=2))
