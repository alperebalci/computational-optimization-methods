from __future__ import annotations

import numpy as np
import leap_ec.ops as ops
from leap_ec import Representation
from leap_ec.algorithm import generational_ea
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import MaxOnes


def run_maxones(length: int = 40, generations: int = 30, population_size: int = 40):
    if length <= 0 or generations <= 0 or population_size < 2:
        raise ValueError("length and generations must be positive; population_size must be at least 2")
    final_population = generational_ea(
        max_generations=generations,
        pop_size=population_size,
        problem=MaxOnes(),
        representation=Representation(initialize=create_binary_sequence(length=length)),
        pipeline=[
            ops.tournament_selection,
            ops.clone,
            mutate_bitflip(expected_num_mutations=1),
            ops.UniformCrossover(p_swap=0.4),
            ops.evaluate,
            ops.pool(size=population_size),
        ],
    )
    best = max(final_population)
    return np.asarray(best.genome, dtype=int), float(best.fitness)


if __name__ == "__main__":
    genome, fitness = run_maxones()
    print("fitness:", fitness)
    print("genome:", genome.tolist())
