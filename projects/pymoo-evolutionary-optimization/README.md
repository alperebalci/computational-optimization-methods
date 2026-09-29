# pymoo Sustainable Factory Design

A reproducible multi-objective evolutionary optimization project built with **pymoo 0.6.2**.

The model chooses two normalized design levers: automation intensity and production speed. NSGA-II searches for a nondominated set balancing annualized cost and emissions while satisfying a minimum-throughput constraint.

## Model

Decision vector: `x = [automation, speed]`.

- automation is bounded to `[0, 1]`
- speed is bounded to `[0.5, 1.5]`
- objective 1 minimizes annualized cost
- objective 2 minimizes emissions
- constraint requires normalized throughput to be at least 1.0

This is intentionally a small transparent model: the purpose is to demonstrate correct pymoo problem construction, constraint handling, seeded NSGA-II execution, and Pareto-set inspection.

## Run

```bash
python -m pip install -e '.[dev]'
python -m pymoo_factory.model
pytest
```

The optimization is stochastic but seeded. Do not interpret a single run as evidence that NSGA-II is superior to another optimizer; use repeated trials and quality indicators for algorithm comparisons.
