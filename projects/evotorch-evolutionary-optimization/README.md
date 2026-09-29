# EvoTorch Constrained Production Search

A PyTorch-native evolutionary optimization example using **EvoTorch 0.6.1** and functional PGPE.

Three continuous production decisions are optimized for contribution margin. Capacity violations and negative production are penalized explicitly. The project is small enough to inspect mathematically while still demonstrating the main reason to use EvoTorch: tensorized population evaluation that can move to accelerators.

## Run

```bash
python -m pip install -e '.[dev]'
python -m evotorch_production.model
pytest
```

The objective function is vectorized over a population tensor. For larger simulation or neural optimization tasks, the same pattern can be moved to GPU by placing tensors on a CUDA device.
