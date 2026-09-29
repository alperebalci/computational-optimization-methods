import numpy as np

from geatpy_ackley.model import AckleyProblem, ackley


def test_ackley_global_optimum_is_zero():
    value = ackley(np.zeros((1, 5)))[0]
    assert abs(value) < 1e-12


def test_problem_metadata():
    problem = AckleyProblem(4)
    assert problem.Dim == 4
    assert problem.M == 1
