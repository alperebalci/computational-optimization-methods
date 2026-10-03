import numpy as np

from advanced_evolutionary import (
    LearningClassifierSystem,
    benchmark_methods,
    competitive_coevolution,
    distributed_island_ga,
    evolve_cgp,
    evolve_grammar,
    interactive_evolution,
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


def test_interactive_coevolution_and_lcs():
    target = np.array([0.4, -0.2, 0.1])
    best, score = interactive_evolution(
        lambda x: -float(np.sum((x - target) ** 2)),
        dimensions=3,
        population_size=10,
        generations=8,
        seed=1,
    )
    assert best.shape == (3,)
    assert np.isfinite(score)

    a, b = competitive_coevolution(lambda x, y: x - y, generations=5, seed=1)
    assert -1.0 <= a <= 1.0
    assert -1.0 <= b <= 1.0

    data = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
    lcs = LearningClassifierSystem(2, seed=4).fit(data, epochs=40)
    assert sum(lcs.predict(x) == y for x, y in data) >= 3


def test_cgp_and_grammatical_evolution_run():
    x = np.linspace(-1.0, 1.0, 21)
    y = x * x
    genome, cgp_loss = evolve_cgp(x, y, n_nodes=8, generations=20, offspring=5, seed=0)
    assert genome.evaluate([x]).shape == x.shape
    assert np.isfinite(cgp_loss)

    expression, ge_loss = evolve_grammar(
        x,
        y,
        population_size=30,
        generations=15,
        seed=0,
    )
    assert isinstance(expression, str)
    assert np.isfinite(ge_loss)
