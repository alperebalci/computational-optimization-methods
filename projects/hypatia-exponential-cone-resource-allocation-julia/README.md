# Hypatia Exponential-Cone Resource Allocation

A Julia/JuMP example using **Hypatia** for a utility-maximizing resource-allocation model represented with exponential cones.

## Model

For allocations `x_i > 0`, maximize weighted logarithmic utility:

```text
maximize    sum(alpha_i * log(x_i))
subject to  sum(cost_i * x_i) <= budget
            sum(x_i) <= capacity
            x_i >= lower_i
```

The logarithm is modeled without a nonlinear black box. Auxiliary variables `t_i` satisfy

```text
exp(t_i) <= x_i
```

through the exponential cone, and the objective maximizes `sum(alpha_i t_i)`.

## Run

```bash
julia --project=. -e 'using Pkg; Pkg.add(["JuMP","Hypatia"])'
julia --project=. run.jl
```

## Why Hypatia

Hypatia is a Julia interior-point solver for generic conic optimization with a broad collection of cone families. This project demonstrates a cone-native nonlinear utility model rather than translating it to a generic NLP.

## Interpretation

Weighted log utility encourages diversification because marginal utility decreases with allocation. All data are synthetic.
