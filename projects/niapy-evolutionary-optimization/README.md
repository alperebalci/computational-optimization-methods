# NiaPy Differential Evolution Benchmark

A continuous global-optimization project using **NiaPy 2.7.1**.

Differential Evolution is run on the Griewank benchmark with a controlled evaluation budget and random seed. The project is designed for extending into algorithm-comparison experiments across NiaPy's nature-inspired optimizer catalog.

## Run

```bash
python -m pip install -e '.[dev]'
python -m niapy_griewank.model
pytest
```

NiaPy 2.7.1 requires Python 3.11 or newer.
