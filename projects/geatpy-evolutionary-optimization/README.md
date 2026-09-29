# geatpy Evolutionary Optimization Lab

Evolutionary and multi-objective optimization experiments using GEATpy.

## Purpose

Focused Jors Academy lab for evaluating **geatpy** as an evolutionary/metaheuristic optimization framework.

## Research checklist

- reproduce a small continuous optimization baseline;
- document population, termination and seed settings;
- distinguish objective evaluations from wall-clock time;
- add constrained or discrete cases where supported;
- report nondominated sets for multi-objective studies;
- validate results against an independent reference before performance claims.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The dependency is isolated in this project rather than added to the umbrella root environment.
