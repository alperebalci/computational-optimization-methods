# inspyred Evolutionary Optimization Lab

Bio-inspired evolutionary computation experiments with inspyred.

## Purpose

This project is a focused Jors Academy lab for evaluating **inspyred** as an evolutionary/metaheuristic optimization framework. It is intentionally separated from the umbrella's mathematical-programming benchmark so library-specific APIs, representations, operators, and benchmarking assumptions remain explicit.

## Research checklist

- reproduce a small continuous optimization baseline;
- document population/termination/seed settings;
- distinguish objective evaluations from wall-clock time;
- add a constrained or discrete case where the library supports it naturally;
- for multi-objective libraries, report the nondominated set rather than collapsing objectives into an arbitrary scalar;
- compare against at least one independent reference or known benchmark value before making performance claims.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The dependency is deliberately isolated in this project rather than added to the umbrella root environment.

## Status

Scaffolded as a library-specific research project. Add experiments under `experiments/`, reusable code under `src/`, and tests under `tests/` as the study grows.
