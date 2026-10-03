# COSMO Nearest Correlation Matrix SDP

A Julia/JuMP project using **COSMO** to repair an indefinite correlation-like matrix by projecting it onto the set of positive-semidefinite matrices with unit diagonal.

## Model

Given a symmetric matrix `C`, solve

```text
minimize    ||X - C||_F^2
subject to  X is positive semidefinite
            diag(X) = 1
```

This is a convex semidefinite optimization problem. It is useful when empirical or manually assembled correlation matrices are internally inconsistent.

## Run

```bash
julia --project=. -e 'using Pkg; Pkg.add(["JuMP","COSMO"])'
julia --project=. run.jl
```

The script reports the objective, minimum eigenvalue, and the repaired matrix.

## Why COSMO

COSMO is an operator-splitting conic solver with semidefinite-cone support and chordal-decomposition capabilities. The nearest-correlation problem is a direct SDP workload rather than a generic LP/QP example.

## Notes

The input matrix is deliberately synthetic and indefinite so that the PSD repair is observable.
