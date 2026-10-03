"""Advanced evolutionary-computation reference implementations."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GAResult:
    best_vector: np.ndarray
    best_fitness: float
    generations: int
    backend: str


@dataclass(frozen=True)
class Elite:
    genome: np.ndarray
    fitness: float
    descriptor: np.ndarray


def _array_backend(name: str):
    if name not in {"auto", "numpy", "cupy"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'cupy'")
    if name == "numpy":
        return np, "numpy"
    try:
        import cupy as cp
    except ImportError:
        if name == "cupy":
            raise RuntimeError("CuPy backend requested but CuPy is not installed") from None
        return np, "numpy"
    return cp, "cupy"


def real_valued_ga(
    *,
    dimensions: int = 16,
    population_size: int = 256,
    generations: int = 100,
    mutation_sigma: float = 0.15,
    elite_fraction: float = 0.10,
    seed: int = 0,
    backend: str = "auto",
) -> GAResult:
    """Minimize Sphere with a vectorized GA on NumPy or optional CuPy."""
    if dimensions <= 0 or population_size < 4 or generations < 0:
        raise ValueError("invalid GA configuration")
    if not 0 < elite_fraction < 0.5:
        raise ValueError("elite_fraction must be in (0, 0.5)")

    xp, backend_name = _array_backend(backend)
    rng = xp.random.RandomState(seed)
    population = rng.uniform(-5.12, 5.12, size=(population_size, dimensions))
    elite_count = max(2, int(population_size * elite_fraction))

    for _ in range(generations):
        fitness = xp.sum(population * population, axis=1)
        elite_idx = xp.argsort(fitness)[:elite_count]
        elites = population[elite_idx]
        count = population_size - elite_count
        p1 = elites[rng.randint(0, elite_count, size=count)]
        p2 = elites[rng.randint(0, elite_count, size=count)]
        mask = rng.random_sample(size=p1.shape) < 0.5
        children = xp.where(mask, p1, p2)
        children += rng.normal(0.0, mutation_sigma, size=children.shape)
        children = xp.clip(children, -5.12, 5.12)
        population = xp.concatenate([elites, children], axis=0)

    fitness = xp.sum(population * population, axis=1)
    idx = int(xp.argmin(fitness).item())
    best = population[idx]
    if backend_name == "cupy":
        best = xp.asnumpy(best)
    return GAResult(np.asarray(best), float(fitness[idx].item()), generations, backend_name)


def _evolve_island(
    population: np.ndarray,
    *,
    generations: int,
    rng: np.random.Generator,
    mutation_sigma: float,
) -> np.ndarray:
    size = len(population)
    elite_count = max(2, size // 5)
    for _ in range(generations):
        fitness = np.sum(population * population, axis=1)
        elites = population[np.argsort(fitness)[:elite_count]]
        pairs = rng.integers(0, elite_count, size=(size - elite_count, 2))
        a, b = elites[pairs[:, 0]], elites[pairs[:, 1]]
        mask = rng.random(a.shape) < 0.5
        children = np.where(mask, a, b) + rng.normal(0.0, mutation_sigma, size=a.shape)
        population = np.vstack([elites, children])
    return population


def distributed_island_ga(
    *,
    dimensions: int = 12,
    island_size: int = 64,
    epochs: int = 8,
    generations_per_epoch: int = 10,
    migrants: int = 2,
    mutation_sigma: float = 0.20,
    seed: int = 0,
    use_mpi: bool = True,
) -> GAResult:
    """Run an island GA; use MPI ring migration when launched with multiple ranks."""
    if dimensions <= 0 or island_size < 4 or epochs <= 0 or generations_per_epoch <= 0:
        raise ValueError("invalid island-GA configuration")
    migrants = min(max(1, migrants), island_size // 2)

    mpi = None
    if use_mpi:
        try:
            from mpi4py import MPI

            mpi = MPI
        except ImportError:
            mpi = None

    if mpi is not None and mpi.COMM_WORLD.Get_size() > 1:
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        rng = np.random.default_rng(seed + rank)
        population = rng.uniform(-5.12, 5.12, size=(island_size, dimensions))
        for _ in range(epochs):
            population = _evolve_island(
                population,
                generations=generations_per_epoch,
                rng=rng,
                mutation_sigma=mutation_sigma,
            )
            order = np.argsort(np.sum(population * population, axis=1))
            outgoing = population[order[:migrants]].copy()
            incoming = np.empty_like(outgoing)
            comm.Sendrecv(
                outgoing,
                dest=(rank + 1) % size,
                recvbuf=incoming,
                source=(rank - 1) % size,
            )
            population[order[-migrants:]] = incoming

        idx = int(np.argmin(np.sum(population * population, axis=1)))
        local = (float(np.sum(population[idx] ** 2)), population[idx].copy())
        gathered = comm.gather(local, root=0)
        best = min(gathered, key=lambda item: item[0]) if rank == 0 else local
        best = comm.bcast(best, root=0)
        return GAResult(np.asarray(best[1]), float(best[0]), epochs * generations_per_epoch, "mpi")

    island_count = 4
    rngs = [np.random.default_rng(seed + i) for i in range(island_count)]
    populations = [
        rng.uniform(-5.12, 5.12, size=(island_size, dimensions)) for rng in rngs
    ]
    for _ in range(epochs):
        for i in range(island_count):
            populations[i] = _evolve_island(
                populations[i],
                generations=generations_per_epoch,
                rng=rngs[i],
                mutation_sigma=mutation_sigma,
            )
        outgoing = []
        for population in populations:
            order = np.argsort(np.sum(population * population, axis=1))
            outgoing.append(population[order[:migrants]].copy())
        for i, population in enumerate(populations):
            order = np.argsort(np.sum(population * population, axis=1))
            population[order[-migrants:]] = outgoing[(i - 1) % island_count]

    combined = np.vstack(populations)
    fitness = np.sum(combined * combined, axis=1)
    idx = int(np.argmin(fitness))
    return GAResult(
        combined[idx].copy(),
        float(fitness[idx]),
        epochs * generations_per_epoch,
        "serial-islands",
    )


def map_elites(
    fitness_fn: Callable[[np.ndarray], float],
    descriptor_fn: Callable[[np.ndarray], np.ndarray],
    *,
    dimensions: int = 8,
    bins: tuple[int, int] = (12, 12),
    descriptor_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-1.0, 1.0),
        (-1.0, 1.0),
    ),
    iterations: int = 2_000,
    seed: int = 0,
) -> dict[tuple[int, int], Elite]:
    """Build a two-dimensional MAP-Elites archive."""
    if dimensions < 2 or min(bins) <= 0 or iterations <= 0:
        raise ValueError("invalid MAP-Elites configuration")
    rng = np.random.default_rng(seed)
    archive: dict[tuple[int, int], Elite] = {}

    def cell(descriptor: np.ndarray) -> tuple[int, int] | None:
        descriptor = np.asarray(descriptor, dtype=float)
        if descriptor.shape != (2,):
            raise ValueError("descriptor_fn must return shape (2,)")
        coords = []
        for value, bounds, count in zip(descriptor, descriptor_bounds, bins, strict=True):
            low, high = bounds
            if not low <= value <= high:
                return None
            scaled = (value - low) / (high - low)
            coords.append(min(count - 1, int(scaled * count)))
        return int(coords[0]), int(coords[1])

    for _ in range(iterations):
        if archive and rng.random() < 0.8:
            keys = list(archive)
            parent = archive[keys[int(rng.integers(0, len(keys)))]].genome
            genome = np.clip(parent + rng.normal(0.0, 0.10, dimensions), -1.0, 1.0)
        else:
            genome = rng.uniform(-1.0, 1.0, dimensions)
        descriptor = np.asarray(descriptor_fn(genome), dtype=float)
        key = cell(descriptor)
        if key is None:
            continue
        fitness = float(fitness_fn(genome))
        incumbent = archive.get(key)
        if incumbent is None or fitness > incumbent.fitness:
            archive[key] = Elite(genome.copy(), fitness, descriptor.copy())
    return archive


def novelty_search(
    behavior_fn: Callable[[np.ndarray], np.ndarray],
    *,
    dimensions: int = 8,
    population_size: int = 80,
    generations: int = 40,
    archive_additions: int = 5,
    k_neighbors: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Search for behavioral novelty without objective-based selection."""
    if dimensions <= 0 or population_size < 4 or generations <= 0:
        raise ValueError("invalid novelty-search configuration")
    rng = np.random.default_rng(seed)
    population = rng.uniform(-1.0, 1.0, size=(population_size, dimensions))
    genome_archive: list[np.ndarray] = []
    behavior_archive: list[np.ndarray] = []

    for _ in range(generations):
        behaviors = np.asarray([behavior_fn(genome) for genome in population], dtype=float)
        if behaviors.ndim != 2:
            raise ValueError("behavior_fn must return fixed-size one-dimensional descriptors")
        reference = behaviors
        if behavior_archive:
            reference = np.vstack([behaviors, np.asarray(behavior_archive)])
        novelty = np.zeros(population_size)
        for i, behavior in enumerate(behaviors):
            distances = np.linalg.norm(reference - behavior, axis=1)
            distances.sort()
            start = 1
            count = min(k_neighbors, len(distances) - start)
            neighbors = distances[start : start + count]
            novelty[i] = float(np.mean(neighbors)) if len(neighbors) else 0.0
        order = np.argsort(novelty)[::-1]
        for idx in order[: min(archive_additions, population_size)]:
            genome_archive.append(population[idx].copy())
            behavior_archive.append(behaviors[idx].copy())
        elite_count = max(2, population_size // 4)
        elites = population[order[:elite_count]]
        parents = elites[rng.integers(0, elite_count, size=population_size - elite_count)]
        children = np.clip(parents + rng.normal(0.0, 0.15, size=parents.shape), -1.0, 1.0)
        population = np.vstack([elites, children])

    return np.asarray(genome_archive), np.asarray(behavior_archive)


def benchmark_methods(
    methods: dict[str, Callable[[int], float]],
    *,
    seeds: Iterable[int] = range(10),
    target: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Benchmark scalar minimization methods over repeated seeds."""
    seed_list = list(seeds)
    if not methods or not seed_list:
        raise ValueError("methods and seeds must not be empty")
    report: dict[str, dict[str, Any]] = {}
    for name, method in methods.items():
        values: list[float] = []
        times: list[float] = []
        for seed in seed_list:
            start = time.perf_counter()
            values.append(float(method(int(seed))))
            times.append(time.perf_counter() - start)
        ordered = sorted(values)
        q1 = ordered[len(ordered) // 4]
        q3 = ordered[(3 * len(ordered)) // 4]
        summary: dict[str, Any] = {
            "runs": len(values),
            "median_objective": statistics.median(values),
            "mean_objective": statistics.fmean(values),
            "best_objective": min(values),
            "iqr_objective": q3 - q1,
            "median_wall_seconds": statistics.median(times),
        }
        if target is not None:
            summary["success_rate"] = sum(value <= target for value in values) / len(values)
        report[name] = summary
    return report


from .extended import (
    CGPGenome,
    LearningClassifierSystem,
    competitive_coevolution,
    evolve_cgp,
    evolve_grammar,
    interactive_evolution,
)


__all__ = [
    "CGPGenome",
    "LearningClassifierSystem",
    "competitive_coevolution",
    "evolve_cgp",
    "evolve_grammar",
    "interactive_evolution",
    "Elite",
    "GAResult",
    "benchmark_methods",
    "distributed_island_ga",
    "map_elites",
    "novelty_search",
    "real_valued_ga",
]
