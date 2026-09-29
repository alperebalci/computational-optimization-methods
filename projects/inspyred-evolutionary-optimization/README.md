# inspyred Rastrigin Evolution Strategy

A benchmark-oriented project using **inspyred 1.0.3**.

The implementation follows inspyred's canonical real-coded evolution-strategy workflow on the Rastrigin benchmark: seeded RNG, bounded candidates, evaluation-count termination, and Gaussian mutation.

## Run

```bash
python -m pip install -e '.[dev]'
python -m inspyred_rastrigin.model
pytest
```

The returned fitness is a benchmark result from one stochastic run. Use repeated seeds when comparing algorithm settings.
