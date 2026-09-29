# Computational Optimization Methods

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`box-constrained-optimization`](projects/box-constrained-optimization/)
- [`gpu-halpern-pdhg-benchmark`](projects/gpu-halpern-pdhg-benchmark/)
- [`matrix-free-pdhg-large-scale-linear-programming-python`](projects/matrix-free-pdhg-large-scale-linear-programming-python/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

A reproducible benchmark harness for comparing a CPU HiGHS reference with NVIDIA cuOpt on the **same LP/MIP mathematical models**. The repository is designed to remain scientifically honest on ordinary GitHub Actions runners: CPU CI validates the model, reference solver, adapter contract and benchmark/reporting logic, but it does **not** invent a GPU speedup when no GPU/cuOpt runtime exists.

## Why this repository exists

GPU optimization benchmarks are easy to overstate. Different formulations, parser costs, warm starts, hardware, stopping rules or MIP optimality semantics can make an apparently simple speedup number misleading. This project makes those distinctions explicit.

NVIDIA cuOpt 26.08 exposes a Python API for routing, LP/QP/convex optimization and MIP. The documented mathematical-optimization API constructs a `Problem`, adds variables/constraints, sets the objective and solves with optional `SolverSettings`. cuOpt also supports remote LP/MIP execution through environment variables, allowing a CPU client to target a GPU server without changing model code.

> **MIP caveat:** NVIDIA documents the cuOpt MIP solver as beta. It currently emphasizes finding high-quality feasible solutions with GPU-accelerated primal heuristics; optimality proof remains under active development. This repository therefore never treats a feasible cuOpt MIP incumbent as proven optimal unless the solver status says so.

## Included benchmark families

### Transportation LP

A seeded balanced transportation problem with equality supply/demand constraints represented in canonical `A x <= b` form. It is solved by SciPy/HiGHS as an independent CPU reference. On a cuOpt runtime, the same coefficient arrays are rebuilt through the cuOpt Python `Problem` API.

### Multidimensional 0-1 knapsack MIP

A seeded binary maximization problem represented internally as minimization of negative profit. It is used to compare incumbent quality, feasibility, status and MIP gap where available.

## Correctness gates

For every backend result the harness records objective, status, wall time and solution vector. It independently recomputes maximum constraint/bound/integrality violation. For LP, an executed cuOpt result must agree with the HiGHS objective to relative tolerance `1e-6` and be feasible to `1e-6`; otherwise the benchmark exits non-zero.

For MIP, HiGHS provides the reference optimum on the CI-sized instance. cuOpt results are recorded as candidate incumbents, but the report keeps solver status/gap visible because cuOpt MIP is beta.

## Install for CPU development / CI

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python scripts/run_benchmarks.py
```

On a normal CPU-only environment, `candidate` will be `null` and `wall_speedup` will be `null`. That is intentional.

## Run with NVIDIA cuOpt

Use a supported cuOpt 26.08 environment following NVIDIA's installation/system-requirement documentation, then run:

```bash
python scripts/cuopt_api_smoke.py
python scripts/run_benchmarks.py --time-limit 60 --output artifacts/gpu-benchmark.json
```

The adapter imports:

```python
from cuopt.linear_programming.problem import CONTINUOUS, INTEGER, MINIMIZE, Problem
from cuopt.linear_programming.solver_settings import SolverSettings
```

and follows NVIDIA's documented `Problem.addVariable`, `Problem.addConstraint`, `Problem.setObjective`, `SolverSettings.set_parameter("time_limit", ...)`, and `Problem.solve` workflow.

For remote LP/MIP execution, NVIDIA documents:

```bash
export CUOPT_REMOTE_HOST=<gpu-server-hostname-or-ip>
export CUOPT_REMOTE_PORT=5001
python scripts/run_benchmarks.py
```

## Benchmark JSON

Each run writes the environment plus CPU reference and cuOpt candidate, if available. A speedup field exists only when cuOpt actually returned a solve result.

```json
{
  "environment": {
    "cuopt_importable": false,
    "nvidia_smi": false
  },
  "results": [
    {
      "model_type": "lp",
      "candidate": null,
      "wall_speedup": null
    }
  ]
}
```

This prevents CPU CI from being misrepresented as evidence of GPU acceleration.

## GitHub Actions

`CI` runs on Ubuntu 24.04 with Python 3.10, 3.11 and 3.12. It executes Ruff, the unit tests, and a CPU benchmark smoke that uploads the JSON artifact. The workflow intentionally does not install cuOpt because GitHub-hosted runners do not provide the supported NVIDIA GPU runtime required for a meaningful local cuOpt benchmark.

## Research limitations

- CI is a **correctness/reproducibility** test, not a GPU performance study.
- Timing includes Python-side solve-call wall time, but not environment provisioning.
- LP comparisons are appropriate for strict objective/feasibility equality.
- MIP comparisons need time-limit, incumbent, gap and status context.
- Small synthetic instances are smoke tests; serious GPU scaling studies should sweep instance size and repeat runs after warm-up.
- Routing is deliberately excluded from v0.1 because cuOpt routing has a different data model and remote-execution path; mixing it into LP/MIP would weaken benchmark comparability.

## Primary references

- NVIDIA cuOpt 26.08 Python API / Quickstart documentation.
- NVIDIA cuOpt 26.08 Convex Optimization examples (`Problem`, `SolverSettings`).
- NVIDIA cuOpt 26.08 MIP documentation and beta-status note.
- NVIDIA cuOpt 26.08 release notes, including multi-GPU PDLP and MIP heuristic updates.
- SciPy `linprog` and `milp` / HiGHS for the independent CPU reference.

See `RESEARCH_NOTES.md` for methodology and interpretation details.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms. NVIDIA cuOpt itself remains subject to NVIDIA's own licensing and system requirements.
