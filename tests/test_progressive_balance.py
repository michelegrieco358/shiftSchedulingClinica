from datetime import date

import pandas as pd

from src.preprocessing import compute_month_progressive_balance


def _build_employees():
    return pd.DataFrame(
        {
            "employee_id": ["A", "B"],
            "hours_due_month": [160, 150],
            "start_balance": [5, -2],
        }
    )


def test_full_month_horizon():
    employees = _build_employees()
    history = pd.DataFrame(columns=["employee_id", "date", "hours_effective"])
    plan = pd.DataFrame(
        {
            "employee_id": ["A", "B"],
            "date": [date(2025, 3, 1), date(2025, 3, 15)],
            "hours_planned": [160, 150],
        }
    )

    result = compute_month_progressive_balance(
        employees,
        history,
        plan,
        month_start=date(2025, 3, 1),
        month_end=date(2025, 3, 31),
        horizon_start=date(2025, 3, 1),
        horizon_end=date(2025, 3, 31),
    )

    assert result.set_index("employee_id").loc["A", "end_balance"] == 5
    assert result.set_index("employee_id").loc["B", "hours_due_period"] == 150


def test_partial_month_horizon_with_prorata():
    employees = _build_employees()
    history = pd.DataFrame(columns=["employee_id", "date", "hours_effective"])
    plan = pd.DataFrame(
        {
            "employee_id": ["A"],
            "date": [date(2025, 3, 10)],
            "hours_planned": [80],
        }
    )

    result = compute_month_progressive_balance(
        employees,
        history,
        plan,
        month_start=date(2025, 3, 1),
        month_end=date(2025, 3, 31),
        horizon_start=date(2025, 3, 1),
        horizon_end=date(2025, 3, 15),
    )

    a_row = result.set_index("employee_id").loc["A"]
    assert a_row["hours_due_period"] == 160 * (15 / 31)
    assert a_row["end_balance"] == 5 + (80 - a_row["hours_due_period"])


def test_horizon_start_mid_month():
    employees = _build_employees()
    history = pd.DataFrame(
        {
            "employee_id": ["A", "A"],
            "date": [date(2025, 3, 2), date(2025, 3, 10)],
            "hours_effective": [8, 7],
        }
    )
    plan = pd.DataFrame(
        {
            "employee_id": ["A"],
            "date": [date(2025, 3, 20)],
            "hours_planned": [40],
        }
    )

    result = compute_month_progressive_balance(
        employees,
        history,
        plan,
        month_start=date(2025, 3, 1),
        month_end=date(2025, 3, 31),
        horizon_start=date(2025, 3, 16),
        horizon_end=date(2025, 3, 31),
    )

    a_row = result.set_index("employee_id").loc["A"]
    assert a_row["hours_hist_to_hstart"] == 15
    assert a_row["hours_eff_to_date"] == 55
