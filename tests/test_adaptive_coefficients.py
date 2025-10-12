import pandas as pd
import pytest

from src.preprocessing import compute_adaptive_coefficients


def test_adaptive_coefficients_respect_monotonicity() -> None:
    balances = pd.Series(
        [-20.0, -5.0, 0.0, 15.0, 40.0],
        index=["E1", "E2", "E3", "E4", "E5"],
        name="start_balance",
    )

    coeffs = compute_adaptive_coefficients(balances)

    assert set(coeffs.columns) == {"c_under", "c_over"}
    assert coeffs["c_under"].between(1.0, 2.0).all()
    assert coeffs["c_over"].between(1.0, 2.0).all()
    assert coeffs.loc["E1", "c_under"] >= coeffs.loc["E2", "c_under"] >= coeffs.loc["E4", "c_under"]
    assert coeffs.loc["E5", "c_over"] >= coeffs.loc["E3", "c_over"]


def test_adaptive_coefficients_equal_balances_are_neutral() -> None:
    balances = pd.Series([7.0, 7.0, 7.0], index=["A", "B", "C"])

    coeffs = compute_adaptive_coefficients(balances)

    assert pytest.approx(coeffs["c_under"].iloc[0], rel=1e-6) == 1.5
    assert (coeffs["c_under"] == 1.5).all()
    assert (coeffs["c_over"] == 1.5).all()


def test_adaptive_coefficients_dataframe_input_with_clamp() -> None:
    data = pd.DataFrame(
        {
            "employee_id": ["H1", "H2"],
            "start_balance": [200.0, -180.0],
        }
    )

    coeffs = compute_adaptive_coefficients(
        data,
        abs_min=-100.0,
        abs_max=100.0,
        min_range=20.0,
    )

    assert pytest.approx(coeffs.loc["H1", "c_over"], rel=1e-6) == 2.0
    assert pytest.approx(coeffs.loc["H2", "c_under"], rel=1e-6) == 2.0
