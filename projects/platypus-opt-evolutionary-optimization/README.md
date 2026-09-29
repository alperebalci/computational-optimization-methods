# Platypus Bi-objective Engineering Trade-off

A small multi-objective engineering-design study using **Platypus-Opt** and NSGA-II.

Two continuous design variables drive two competing quadratic objectives. The example demonstrates Platypus problem construction, real-valued decision types, NSGA-II execution, and nondominated result extraction without adding plotting dependencies.

## Run

```bash
python -m pip install -e '.[dev]'
python -m platypus_tradeoff.model
pytest
```
