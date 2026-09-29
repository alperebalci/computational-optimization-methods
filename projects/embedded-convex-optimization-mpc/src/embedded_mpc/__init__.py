"""Parametric convex model-predictive control benchmarks."""

from .model import MPCConfig, MPCResult, ParametricMPC, double_integrator_config

__all__ = ["MPCConfig", "MPCResult", "ParametricMPC", "double_integrator_config"]

from .timing import RepeatedSolveBenchmark, benchmark_repeated_solves
