import numpy as np

from infeas_or import (
    LinearInequalityModel,
    l1_feasibility_relaxation,
    minimum_constraint_deletion,
    minimal_infeasible_subset,
)


def model():
    # x >= 5, x <= 3, x <= 10. The third row is redundant to the conflict.
    return LinearInequalityModel(
        a_ub=np.array([[-1.0], [1.0], [1.0]]),
        b_ub=np.array([-5.0, 3.0, 10.0]),
        bounds=((0.0, None),),
    )


def test_deletion_filter_removes_redundant_constraint_from_conflict():
    conflict = minimal_infeasible_subset(model())
    assert set(conflict) == {0, 1}


def test_minimum_cardinality_repair_deletes_one_conflicting_row():
    deleted = minimum_constraint_deletion(model())
    assert len(deleted) == 1
    assert deleted[0] in {0, 1}


def test_weighted_l1_relaxation_explains_required_violation():
    result = l1_feasibility_relaxation(model(), weights=np.array([10.0, 1.0, 1.0]))
    # Keeping x >= 5 is expensive to violate, so the cheaper x <= 3 row absorbs 2 units.
    assert np.isclose(result.x[0], 5.0, atol=1e-8)
    assert np.isclose(result.slacks[1], 2.0, atol=1e-8)
    assert np.isclose(result.weighted_violation, 2.0, atol=1e-8)
