"""Embedded convex optimization MPC benchmark."""

from .model import MPCConfig, MPCResult, ParametricMPC, double_integrator_config
from .timing import RepeatedSolveBenchmark, benchmark_repeated_solves

__all__ = [
    "MPCConfig",
    "MPCResult",
    "ParametricMPC",
    "RepeatedSolveBenchmark",
    "benchmark_repeated_solves",
    "double_integrator_config",
]
