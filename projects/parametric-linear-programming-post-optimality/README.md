# Parametric Linear Programming and Post-Optimality

This project treats sensitivity analysis as a computational object rather than a paragraph after an optimization model.

Benchmark:

```text
minimize  x_cheap + 3 x_flex

subject to
    x_cheap + x_flex >= demand
    x_cheap <= 4
    x_cheap, x_flex >= 0
```

The right-hand-side parameter `demand` changes continuously.

For demand below the cheap-capacity breakpoint, the optimal value grows with slope 1. Once cheap capacity is saturated, the value function changes regime and grows with slope 3.

The implementation records for each parameter value:

- primal solution;
- objective value;
- solver-derived demand shadow price;
- binding/active constraints;
- active-set regime changes across a parameter sweep.

The test suite independently estimates the derivative of the optimal value by centered finite differences and checks that it agrees with the LP dual multiplier away from the breakpoint.

## Why this matters

An optimal solution at one nominal data point does not tell a decision-maker whether the plan is stable. Parametric optimization exposes:

- where the current structural regime remains valid;
- where a capacity constraint becomes binding;
- how the marginal value of demand/capacity changes;
- where the value function is nondifferentiable.

## Run

```bash
python -m pip install -e '.[dev]'
pytest
```

## Scope

This benchmark focuses on RHS parametric LP sensitivity. Objective-coefficient ranges, multiparametric LP/QP critical regions, MILP stability, basis continuation, and differentiable solution mappings are separate extensions.
