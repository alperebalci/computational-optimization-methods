# GEATpy Ackley Differential Evolution

A compatibility-conscious GEATpy project using the last PyPI release, **GEATpy 2.7.0**.

The project models the Ackley benchmark as a custom `ea.Problem` and solves it with GEATpy's `soea_DE_rand_1_bin_templet`. GEATpy's PyPI wheels stop at CPython 3.10, so this project deliberately pins Python to `>=3.10,<3.11` rather than pretending current interpreter support exists.

## Run

Use Python 3.10:

```bash
python3.10 -m pip install -e '.[dev]'
python -m geatpy_ackley.model
pytest
```

This project is retained for framework comparison and legacy reproducibility. Treat compatibility constraints as part of the experimental record.
