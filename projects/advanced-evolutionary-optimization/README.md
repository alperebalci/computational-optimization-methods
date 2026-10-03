# Advanced Evolutionary Optimization

Executable reference implementations for advanced evolutionary-computation topics that were missing from the surrounding portfolio.

## Included methods

- **GPU-accelerated GA:** vectorized real-valued GA with NumPy or optional CuPy.
- **Distributed GA:** island-model GA with optional MPI ring migration plus a deterministic serial-island fallback for CI.
- **Interactive Evolutionary Computation:** external scorer callback that can be connected to human ratings or preferences.
- **Coevolutionary algorithms:** competitive two-population zero-sum coevolution.
- **Learning Classifier Systems (LCS):** compact Michigan-style ternary-rule classifier system.
- **Cartesian Genetic Programming:** a (1+lambda) symbolic-regression reference implementation.
- **Grammatical Evolution:** codon-to-grammar symbolic regression with a closed arithmetic grammar.
- **Quality-Diversity / MAP-Elites:** two-dimensional archive that keeps the best solution in each behavioral cell.
- **Novelty Search:** k-nearest-neighbor behavioral novelty with an explicit archive and no objective-based selection.
- **Advanced benchmarking:** repeated-seed summaries with median, IQR, best objective, wall time and optional success rate.

These implementations are deliberately compact but executable and tested. CPU CI validates correctness; it does not claim GPU speedup or multi-machine scaling.

## Install

CPU / CI:

    python -m pip install -e '.[dev]'
    pytest -q

Optional CUDA/CuPy:

    python -m pip install -e '.[gpu]'

Optional MPI:

    python -m pip install -e '.[distributed]'
    mpiexec -n 4 python -c "from advanced_evolutionary import distributed_island_ga; print(distributed_island_ga())"

## Notes

The CuPy extra targets CUDA 12 wheels; use the CuPy package matching the installed CUDA runtime if needed. The MPI implementation performs real inter-process migration only when launched with multiple ranks. Benchmark claims should use identical evaluation budgets, multiple seeds, controlled hardware and an explicit warm-up policy.
