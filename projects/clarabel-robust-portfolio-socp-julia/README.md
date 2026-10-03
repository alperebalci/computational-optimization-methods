# Clarabel Robust Portfolio SOCP

A compact Julia/JuMP project that uses **Clarabel** to solve a long-only portfolio model with a second-order-cone risk limit and an L1 turnover penalty.

## Model

Decision variables are portfolio weights `w_i` and turnover auxiliaries `z_i`.

```text
maximize    mu' w - lambda * sum(z)
subject to  sum(w) = 1
            w >= 0
            ||L w||_2 <= risk_cap
            z_i >=  w_i - w0_i
            z_i >= -w_i + w0_i
```

The matrix `L` is a factor-risk loading matrix, so `||Lw||_2` is the portfolio risk proxy. The risk constraint is represented natively as a second-order cone.

## Run

```bash
julia --project=. -e 'using Pkg; Pkg.add(["JuMP","Clarabel"])'
julia --project=. run.jl
```

The script prints solver status, objective value, expected return, realized risk, turnover, and the optimal weights.

## Why Clarabel

Clarabel is an open-source interior-point conic solver with direct support for LP, QP, SOCP and additional cone families. This example isolates a solver-native SOCP use case instead of hiding the conic structure behind a black-box portfolio library.

## Notes

This is a reproducible numerical example, not investment advice. Returns and factor loadings are synthetic.
