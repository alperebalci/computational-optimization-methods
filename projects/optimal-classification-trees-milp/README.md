# Optimal Classification Trees via MILP

Status: implemented bounded research baseline, v0.1 (2026-10-01).

A fixed-depth binary classification tree is learned by minimizing training misclassification over a finite, training-only catalog of axis-aligned thresholds. Split selection, leaf routing, leaf class labels and sample errors are optimized jointly with HiGHS. The decoded tree is checked against the MILP objective; an independent exhaustive split-structure oracle checks small cases.

The benchmark compares the same depth with CART on three seeded XOR datasets. Both methods tie at zero error in this smoke run; no empirical superiority claim is made. The optimum is restricted to the declared split catalog and depth, not to arbitrary tree structures or out-of-sample error. Depth is limited to 1--3. No Benders decomposition, StrongTree reproduction, fairness constraints or industrial-scale claim is included.

Core API: `fit(X, y, depth=2)`, `predict(model, X)`, `exhaustive_error(X, y)`.

## Run

From this directory, use Python 3.11 or newer:

```bash
python -m pip install -r requirements.txt
python -m unittest checks -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python study.py --output local-results.json
```

`checks.py` is explicitly invoked so existing umbrella-level pytest discovery is unchanged. `results.json` contains the executed CPU smoke run, not industrial performance evidence. `VALIDATION.json` records the environment and source fingerprints. Existing modules are not replaced.

## Research lineage

Independent educational implementation, informed by the following primary source; no paper-level reproduction or new theorem is claimed:

https://arxiv.org/abs/2103.15965

## License and provenance

Original addition made inside this repository; not a recovered snapshot. Existing repository licensing applies; this directory grants no separate rights. See `SOURCE_REPOSITORY.md`.
