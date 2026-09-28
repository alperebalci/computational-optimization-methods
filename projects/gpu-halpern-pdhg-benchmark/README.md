# GPU / Halpern-PDHG Benchmark Laboratory

A small, auditable benchmark for studying **PDHG**, **Halpern-style fixed-point iteration**, periodic restart, and optional GPU execution on box-constrained equality-form linear programs.

The project is motivated by the modern GPU first-order LP line that includes PDLP, restarted Halpern PDHG, and cuPDLPx. It is **not** a reproduction of cuPDLPx.

## Problem class

```text
minimize    c^T x
subject to  A x = b
            lower <= x <= upper.
```

The implementation uses only matrix-vector products, vector arithmetic, and box projection inside the first-order iteration.

## Base PDHG operator

The fixed-point operator uses

```text
x+ = projection_[l,u](x - tau * (c + A^T y))
xbar = 2 x+ - x
y+ = y + sigma * (A xbar - b)
```

with conservative steps satisfying

```text
tau * sigma * ||A||_2^2 < 1.
```

## Halpern modes

The plain Halpern experiment applies

```text
z_(k+1) = alpha_k * z_anchor + (1-alpha_k) * T(z_k)
alpha_k = 1 / (k + 2).
```

A second research baseline periodically replaces the anchor with the current iterate and resets the local Halpern counter.

That periodic rule is deliberately simple. It is **not** the restart criterion or primal-weight controller used by cuPDLPx.

## Why this is separate from the existing matrix-free PDHG project

The existing `matrix-free-pdhg-large-scale-linear-programming-python` project explains a conventional CPU PDHG implementation and its residual diagnostics.

This project asks a different question:

- what changes when the same primal-dual fixed-point map is Halpern-averaged?
- what happens under a transparent periodic restart?
- can the exact same arithmetic be executed with NumPy or optional CuPy?
- how should objective/residual accuracy be compared with an independent HiGHS reference without turning a CPU CI run into a GPU speedup claim?

## Optional CuPy backend

Install only on a compatible CUDA environment:

```bash
pip install -e ".[cuda12]"
```

Then the same solver can be called with `backend="cupy"`. Arrays are transferred to the GPU before iteration and copied back only for convergence diagnostics and the final result.

GitHub-hosted CPU CI does not install CuPy and does not report a GPU speedup.

## Benchmark contract

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m halpern_pdhg.experiment
```

The JSON report contains:

- HiGHS reference objective;
- PDHG result;
- Halpern result;
- periodically restarted Halpern result;
- residuals and projected stationarity;
- iterations and wall time;
- optional CuPy result when available;
- `gpu_speedup_claimed: false`.

Wall times on different backends are not interpreted as a scientific speedup without controlled hardware, warm-up, transfer accounting, repeated runs, and comparable stopping conditions.

## Relationship to current GPU LP solvers

The 2025 cuPDLPx work builds on restarted Halpern PDHG and adds techniques such as a new restart criterion and PID-controlled primal-weight updates. NVIDIA cuOpt 26.08 exposes PDLP, multiple precision modes, and multi-GPU capabilities.

This repository isolates only the basic fixed-point ideas so their behavior can be inspected. It does not implement cuPDLPx's full restart logic, presolve, scaling, adaptive primal weighting, GPU sparse kernels, or production stopping tests.

## Natural extensions

- sparse SciPy / cupyx operators;
- adaptive restart based on normalized KKT progress;
- primal/dual weight adaptation;
- diagonal scaling and presolve;
- MPS benchmark ingestion;
- controlled self-hosted GPU experiments;
- direct external comparison with public cuPDLPx and NVIDIA cuOpt.

## References

- Lu, Peng, and Yang, *cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming*, 2025.
- PDLP / PDHG literature for large-scale linear programming.
- NVIDIA cuOpt documentation for PDLP precision and multi-GPU modes.

## License

This project inherits the umbrella repository's non-commercial/source-available licensing terms.
