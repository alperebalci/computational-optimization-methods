# jMetalPy NSGA-II Benchmark Lab

A compact multi-objective benchmark project using **jMetalPy 1.9.0**.

The project runs NSGA-II on ZDT1, extracts the nondominated approximation, and exposes the objective vectors for later quality-indicator studies. ZDT1 is used deliberately: it separates framework verification from application-specific modeling and gives a known convex Pareto-front structure.

## Run

```bash
python -m pip install -e '.[dev]'
python -m jmetal_zdt.model
pytest
```

For serious algorithm studies, extend this project with repeated seeds plus hypervolume, IGD/IGD+, epsilon, and statistical comparisons rather than judging an optimizer from one front.
