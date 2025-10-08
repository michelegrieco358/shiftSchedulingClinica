from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pytest

DATA_DIR = Path(__file__).resolve().parents[1]

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from loader import load_all
from loader.absences import get_absence_hours_from_config
from loader.calendar import build_calendar
from loader.config import load_config
from loader.cross import enrich_employees_with_cross_policy
from loader.employees import (
    enrich_employees_with_fte,
    load_employees,
    resolve_fulltime_baseline,
)
from loader.shifts import load_shift_role_eligibility, load_shifts


def _calendar_info(cfg: dict[str, object]) -> tuple[int, int]:
    start = datetime.strptime(str(cfg["horizon"]["start_date"]), "%Y-%m-%d").date()
    end = datetime.strptime(str(cfg["horizon"]["end_date"]), "%Y-%m-%d").date()
    calendar_df = build_calendar(start, end)
    horizon_days = int(calendar_df["is_in_horizon"].sum())
    weeks_in_horizon = calendar_df.loc[
        calendar_df["is_in_horizon"], "week_id"
    ].nunique()
    return horizon_days, weeks_in_horizon


def test_load_employees_and_cross_policy_overrides() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_df = load_employees(
        str(DATA_DIR / "employees.csv"),
        cfg.get("defaults", {}),
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )

    enriched = enrich_employees_with_cross_policy(employees_df, cfg)
    lookup = enriched.set_index("employee_id")

    assert lookup.loc["E001", "cross_max_shifts_week"] == 1
    assert lookup.loc["E001", "cross_max_shifts_month"] == 3
    assert lookup.loc["E001", "cross_penalty_weight"] == pytest.approx(0.5)

    cross_cfg = cfg["cross"]
    assert lookup.loc["E002", "cross_max_shifts_week"] == cross_cfg["max_shifts_week"]
    assert lookup.loc["E002", "cross_max_shifts_month"] == cross_cfg["max_shifts_month"]
    assert lookup.loc["E002", "cross_penalty_weight"] == pytest.approx(
        cross_cfg["penalty_weight"]
    )

    assert lookup.loc["E003", "cross_max_shifts_week"] == 0
    assert lookup.loc["E003", "cross_max_shifts_month"] == 0
    assert lookup.loc["E003", "cross_penalty_weight"] == pytest.approx(3.0)


def test_resolve_fulltime_baseline_precedence() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))

    assert resolve_fulltime_baseline(cfg, "caposala") == pytest.approx(150)
    assert resolve_fulltime_baseline(cfg, "infermiere") == pytest.approx(168)
    assert resolve_fulltime_baseline(cfg, None) == pytest.approx(165)


def test_enrich_employees_with_fte_uses_payroll_defaults() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)
    employees_df = load_employees(
        str(DATA_DIR / "employees.csv"),
        cfg.get("defaults", {}),
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )

    fte_df = enrich_employees_with_fte(employees_df, cfg)
    lookup = fte_df.set_index("employee_id")

    assert lookup.loc["E001", "fte"] == pytest.approx(1.0)
    assert lookup.loc["E002", "fte"] == pytest.approx(1.0)
    assert lookup.loc["E003", "fte"] == pytest.approx(1.0)


def test_shift_role_eligibility_with_allowed_column() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)
    employees_df = load_employees(
        str(DATA_DIR / "employees.csv"),
        cfg.get("defaults", {}),
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )
    shifts_df = load_shifts(str(DATA_DIR / "shifts.csv"))
    eligibility_df = load_shift_role_eligibility(
        str(DATA_DIR / "shift_role_eligibility.csv"),
        employees_df,
        shifts_df,
        cfg.get("defaults", {}),
    )

    mask = (eligibility_df["shift_code"] == "N") & (
        eligibility_df["role"] == "CAPOSALA"
    )
    assert mask.any()
    assert not bool(eligibility_df.loc[mask, "allowed"].iloc[0])


def test_get_absence_hours_from_config_and_load_all_smoke() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))
    assert get_absence_hours_from_config(cfg) == pytest.approx(6.0)

    loaded = load_all(str(DATA_DIR / "config.yaml"), str(DATA_DIR))
    assert not loaded.employees_df.empty
    assert {
        "cross_max_shifts_week",
        "cross_max_shifts_month",
        "cross_penalty_weight",
    }.issubset(loaded.employees_df.columns)
    assert not loaded.leaves_df.empty
    assert not loaded.availability_df.empty
