# LEAP Operator-Pipeline Genetic Algorithm

A focused demonstration of **LEAP's composable evolutionary pipeline**.

The project solves the canonical MaxOnes problem with a binary representation, tournament selection, cloning, bit-flip mutation, uniform crossover, evaluation, and pooling. It is intentionally simple because the research target is LEAP's operator-pipeline architecture rather than the optimization problem itself.

## Run

```bash
python -m pip install -e '.[dev]'
python -m leap_maxones.model
pytest
```

The code mirrors LEAP's documented `generational_ea` pattern and makes the representation/operator composition explicit.
