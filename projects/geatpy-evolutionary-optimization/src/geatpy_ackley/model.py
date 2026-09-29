from __future__ import annotations

import numpy as np
import geatpy as ea


def ackley(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / n)) - np.exp(
        np.sum(np.cos(2.0 * np.pi * x), axis=1) / n
    ) + np.e + 20.0


class AckleyProblem(ea.Problem):
    def __init__(self, dimension: int = 10) -> None:
        super().__init__(
            "Ackley",
            1,
            [1],
            dimension,
            [0] * dimension,
            [-32.768] * dimension,
            [32.768] * dimension,
            [1] * dimension,
            [1] * dimension,
        )

    def aimFunc(self, pop) -> None:
        pop.ObjV = ackley(pop.Phen).reshape(-1, 1)


def run(dimension: int = 10, generations: int = 30, population_size: int = 30):
    problem = AckleyProblem(dimension)
    population = ea.Population(Encoding="RI", NIND=population_size)
    algorithm = ea.soea_DE_rand_1_bin_templet(
        problem,
        population,
        MAXGEN=generations,
        logTras=0,
    )
    algorithm.mutOper.F = 0.5
    algorithm.recOper.XOVR = 0.2
    result = ea.optimize(
        algorithm,
        verbose=False,
        drawing=0,
        outputMsg=False,
        drawLog=False,
        saveFlag=False,
    )
    return result


if __name__ == "__main__":
    result = run()
    print("best objective:", result.get("ObjV"))
