# MEALPY Production-Mix Genetic Algorithm

A reproducible production-mix optimization example using **MEALPY 3.0.3**.

The model chooses quantities of three products. The objective is negative contribution margin plus quadratic penalties for labor and machine-capacity violations, so the minimizer balances profit against feasibility. The project uses `GA.OriginalGA` and explicit `FloatVar` bounds.

## Run

```bash
python -m pip install -e '.[dev]'
python -m mealpy_production.model
pytest
```

The penalty formulation is pedagogical rather than a replacement for an exact LP/MIP formulation. Use it to study stochastic search behavior, not to claim that GA is the preferred solver for linear production planning.
