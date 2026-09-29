from embedded_mpc.experiment import compare_warm_cold


def test_warm_and_cold_solves_preserve_objective() -> None:
    result=compare_warm_cold("OSQP",repeats=3)
    assert result["objective_max_abs_difference"] < 1e-4
    assert result["speedup_claimed"] is False
