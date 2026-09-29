# Embedded Convex Optimization for Model Predictive Control

A reproducible **parametric MPC** benchmark that compiles one constrained quadratic program and repeatedly solves it as the initial state changes.

The same mathematical MPC model is solved with three open-source convex backends exposed by CVXPY:

- **OSQP** — operator-splitting QP solver;
- **Clarabel** — interior-point conic solver with QP support;
- **SCS** — first-order conic solver.

The project is about formulation fidelity, repeated-solve behavior, residuals, and solver interoperability. It does not hard-code a latency winner.

## MPC formulation

For linear dynamics

```text
x_(t+1) = A x_t + B u_t
```

the finite-horizon controller solves

```text
minimize
    sum_t (x_t - x_ref)^T Q (x_t - x_ref)
  + sum_t u_t^T R u_t
  + (x_N - x_ref)^T Qf (x_N - x_ref)

subject to
    x_0 = current_state
    x_(t+1) = A x_t + B u_t
    x_lower <= x_t <= x_upper
    u_lower <= u_t <= u_upper.
```

Only `x_0` is a CVXPY `Parameter`. The dynamics and cost matrices remain fixed, which matches a common embedded/receding-horizon use case and allows the canonical problem structure to be reused.

## Repeated solve contract

`ParametricMPC` builds the CVXPY problem once. Each control step:

1. updates the initial-state parameter;
2. solves with a selected backend;
3. independently recomputes dynamics residuals;
4. independently checks state/input bounds;
5. applies only the first control;
6. advances the plant state and repeats.

The benchmark passes `warm_start=True` after the first solve, but does not infer a universal speedup from one short CPU run.

## Solver parity

Tests use OSQP as a numerical reference for the benchmark QP and require Clarabel and SCS to return closely matching objective values and first controls within declared tolerances.

This is not a claim that the algorithms are equivalent. It is a check that all three backends are solving the same model to useful numerical accuracy.

## Run

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m embedded_mpc.experiment
```

The experiment performs a receding-horizon trajectory with each solver and records:

- objective;
- first control;
- dynamics residual;
- bound violation;
- solver-reported solve time when available;
- wall time.

The report explicitly contains `warm_start_timing_claimed: false`.

## Why this fills a separate portfolio gap

The portfolio already contains large-scale LP first-order methods and GPU/cuOpt benchmarking. Embedded MPC asks a different computational question: how should a **small structured convex program be solved repeatedly with changing parameters and strict per-solve correctness checks?**

This project therefore complements rather than duplicates the matrix-free PDHG and GPU benchmark work.

## Current boundary

v0.1 uses a constrained double-integrator QP and CVXPY's solver interfaces. It does not yet implement:

- code-generated bare-metal solvers;
- direct OSQP C API integration;
- explicit factorization-cache instrumentation;
- SOCP MPC;
- nonlinear sequential convex programming;
- hardware-in-the-loop timing;
- real-time deadline scheduling.

Natural extensions are a conic MPC case, direct solver APIs, code generation, and controlled latency experiments on embedded hardware.

## Reference implementation notes

CVXPY documents OSQP, Clarabel, and SCS as supported open-source solvers and supports repeated parameterized solves with `warm_start=True`. Solver statistics are recorded through `problem.solver_stats`.

## License

This project inherits the umbrella repository's non-commercial/source-available licensing terms.
