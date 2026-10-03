import numpy as np

from advanced_evolutionary import (
    benchmark_methods,
    distributed_island_ga,
    map_elites,
    novelty_search,
    real_valued_ga,
)


def test_cpu_ga_improves_sphere():
    result = real_valued_ga(
        dimensions=6,
        population_size=80,
        generations=60,
        mutation_sigma=0.08,
        backend="numpy",
        seed=3,
    )
    assert result.backend == "numpy"
    assert result.best_fitness < 0.2


def test_serial_island_ga_runs():
    result = distributed_island_ga(
        dimensions=5,
        island_size=24,
        epochs=3,
        generations_per_epoch=4,
        use_mpi=False,
        seed=2,
    )
    assert result.backend == "serial-islands"
    assert np.isfinite(result.best_fitness)


def test_map_elites_populates_archive():
    archive = map_elites(
        lambda genome: -float(np.sum(genome * genome)),
        lambda genome: genome[:2],
        dimensions=4,
        bins=(5, 5),
        iterations=300,
        seed=0,
    )
    assert len(archive) > 5


def test_novelty_search_builds_archive():
    genomes, behaviors = novelty_search(
        lambda genome: genome[:2],
        dimensions=4,
        population_size=30,
        generations=6,
        archive_additions=3,
        seed=0,
    )
    assert genomes.shape[0] == 18
    assert behaviors.shape == (18, 2)


def test_benchmark_harness():
    report = benchmark_methods(
        {"a": lambda seed: float(seed), "b": lambda seed: float(seed + 1)},
        seeds=[0, 1, 2, 3],
        target=2.0,
    )
    assert report["a"]["runs"] == 4
    assert report["a"]["success_rate"] == 0.75
