# Certified Bilinear Global Optimization with Spatial Branch-and-Bound

This project demonstrates deterministic **global** optimization rather than global-search heuristics.

Benchmark:

```text
maximize x y
subject to
    x + y <= 1.5
    0 <= x,y <= 1
```

The bilinear objective is nonconvex. A local NLP solution is therefore only an incumbent; it is not a global certificate.

The implementation adds an auxiliary variable `w = xy` and, on every bounded node, constructs the four McCormick inequalities. Solving that LP gives a valid upper bound on the nonconvex objective. Spatial branch-and-bound then:

1. keeps a feasible incumbent (lower bound for maximization);
2. solves McCormick LP relaxations for global upper bounds;
3. bisects the widest variable interval;
4. prunes boxes whose upper bound cannot improve the incumbent;
5. terminates only when the global bound gap is below tolerance.

The known optimum `x=y=0.75`, objective `0.5625`, is recovered and certified by the tests, but the algorithm does not hard-code it.

## Why this belongs in an OR portfolio

Differential evolution, GA, simulated annealing, and multistart NLP may search broadly but normally do not provide a finite global-optimality certificate. Spatial branch-and-bound does so by combining valid relaxations with systematic partitioning.

## Run

```bash
python -m pip install -e '.[dev]'
pytest
```

## Scope

The solver is intentionally small and transparent: one bilinear term and one linear coupling constraint. Production global solvers add expression trees, convexification for many nonlinear primitives, bound tightening, branching rules, domain propagation, local-search heuristics, and numerical safeguards.
