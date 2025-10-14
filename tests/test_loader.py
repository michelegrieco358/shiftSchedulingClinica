from __future__ import annotations

import logging
import re
import shutil
import warnings
from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

DATA_DIR = Path(__file__).resolve().parents[1]
DATA_CSV_DIR = DATA_DIR / 'data'

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from loader import load_all
from loader.availability import load_availability
from loader.absences import get_absence_hours_from_config
from loader.calendar import build_calendar
from loader.config import load_config
from loader.coverage import expand_requirements, load_coverage_groups
from loader.cross import enrich_employees_with_cross_policy
from loader.employees import (
    enrich_employees_with_fte,
    load_employees,
    resolve_fulltime_baseline,
)
from loader.leaves import apply_unplanned_leave_durations, load_leaves
from loader.preassignments import load_preassignments
from loader.shifts import (
    build_shift_slots,
    load_shift_role_eligibility,
    load_shifts,
)
from loader.utils import LoaderError


def _calendar_info(cfg: dict[str, object]) -> tuple[int, int]:
    start = datetime.strptime(str(cfg["horizon"]["start_date"]), "%Y-%m-%d").date()
    end = datetime.strptime(str(cfg["horizon"]["end_date"]), "%Y-%m-%d").date()
    calendar_df = build_calendar(start, end)
    horizon_days = int(calendar_df["is_in_horizon"].sum())
    weeks_in_horizon = calendar_df.loc[
        calendar_df["is_in_horizon"], "week_id"
    ].nunique()
    return horizon_days, weeks_in_horizon


def _write_basic_config(
    tmp_path: Path, defaults_extra: dict[str, object] | None = None
) -> Path:
    cfg_dict: dict[str, object] = {
        "horizon": {"start_date": "2025-01-01", "end_date": "2025-01-31"},
        "defaults": {
            "allowed_roles": ["infermiere"],
            "departments": ["dep"],
            "contract_hours_by_role_h": {"infermiere": 160},
            "night": {
                "can_work_night": True,
                "max_per_week": 2,
                "max_per_month": 8,
            },
        },
    }
    if defaults_extra:
        cfg_dict["defaults"].update(defaults_extra)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    return cfg_path


def _write_employees_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    df = pd.DataFrame(rows)
    employees_path = tmp_path / "employees.csv"
    df.to_csv(employees_path, index=False)
    return employees_path


def _write_coverage_groups_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    df = pd.DataFrame(rows)
    path = tmp_path / "coverage_groups.csv"
    df.to_csv(path, index=False)
    return path


def _load_basic_employees(
    tmp_path: Path,
    defaults_extra: dict[str, object] | None = None,
    employees_rows: list[dict[str, object]] | None = None,
    *,
    caplog: pytest.LogCaptureFixture | None = None,
) -> pd.DataFrame:
    cfg_path = _write_basic_config(tmp_path, defaults_extra)
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    rows = employees_rows or [
        {
            "employee_id": "E1",
            "nome": "Anna",
            "role": "infermiere",
            "reparto_id": "dep",
            "ore_dovute_mese_h": 160,
            "saldo_prog_iniziale_h": 0,
        }
    ]
    employees_path = _write_employees_csv(tmp_path, rows)

    if caplog is not None:
        caplog.set_level(logging.INFO)

    return load_employees(
        str(employees_path),
        cfg.get("defaults", {}),
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )


def _basic_shifts_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shift_id": "M",
                "start": "06:00",
                "end": "14:00",
                "break_min": 0,
                "duration_min": 480,
                "crosses_midnight": 0,
                "start_time": pd.to_timedelta(6, unit="h"),
                "end_time": pd.to_timedelta(14, unit="h"),
            },
            {
                "shift_id": "P",
                "start": "14:00",
                "end": "22:00",
                "break_min": 0,
                "duration_min": 480,
                "crosses_midnight": 0,
                "start_time": pd.to_timedelta(14, unit="h"),
                "end_time": pd.to_timedelta(22, unit="h"),
            },
            {
                "shift_id": "N",
                "start": "22:00",
                "end": "06:00",
                "break_min": 0,
                "duration_min": 480,
                "crosses_midnight": 1,
                "start_time": pd.to_timedelta(22, unit="h"),
                "end_time": pd.to_timedelta(6, unit="h"),
            },
            {
                "shift_id": "X",
                "start": "10:00",
                "end": "18:00",
                "break_min": 0,
                "duration_min": 480,
                "crosses_midnight": 0,
                "start_time": pd.to_timedelta(10, unit="h"),
                "end_time": pd.to_timedelta(18, unit="h"),
            },
        ]
    )


def _empty_dept_map_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "reparto_id",
            "shift_code",
            "enabled",
            "start_override",
            "end_override",
            "start_override_time",
            "end_override_time",
        ]
    )


def test_build_shift_slots_keeps_standard_shifts_enabled_without_override() -> None:
    shifts_df = _basic_shifts_catalog()
    month_plan_df = pd.DataFrame(
        [
            {
                "data": "2025-01-01",
                "data_dt": pd.Timestamp("2025-01-01"),
                "reparto_id": "DEP",
                "shift_code": "M",
                "coverage_code": "COV",
            }
        ]
    )
    defaults = {"departments": ["DEP"]}

    slots_df = build_shift_slots(
        month_plan_df,
        shifts_df,
        _empty_dept_map_df(),
        defaults,
    )

    assert not slots_df.empty
    assert slots_df.loc[0, "shift_code"] == "M"
    assert slots_df.loc[0, "reparto_id"] == "DEP"


def test_build_shift_slots_requires_explicit_department_enable_for_custom_shift() -> None:
    shifts_df = _basic_shifts_catalog()
    month_plan_df = pd.DataFrame(
        [
            {
                "data": "2025-01-01",
                "data_dt": pd.Timestamp("2025-01-01"),
                "reparto_id": "DEP",
                "shift_code": "X",
                "coverage_code": "COV",
            }
        ]
    )
    defaults = {"departments": ["DEP"]}

    with pytest.raises(LoaderError, match="non abilitato"):
        build_shift_slots(
            month_plan_df,
            shifts_df,
            _empty_dept_map_df(),
            defaults,
        )

    dept_map_df = pd.DataFrame(
        [
            {
                "reparto_id": "DEP",
                "shift_code": "X",
                "enabled": True,
                "start_override": "",
                "end_override": "",
                "start_override_time": pd.NaT,
                "end_override_time": pd.NaT,
            }
        ]
    )

    slots_df = build_shift_slots(
        month_plan_df,
        shifts_df,
        dept_map_df,
        defaults,
    )

    assert not slots_df.empty
    assert slots_df.loc[0, "shift_code"] == "X"
    assert slots_df.loc[0, "reparto_id"] == "DEP"


def test_build_shift_slots_applies_break_minutes_to_duration() -> None:
    shifts_df = pd.DataFrame(
        [
            {
                "shift_id": "M",
                "start": "08:00",
                "end": "16:00",
                "break_min": 45,
                "duration_min": 435,
                "crosses_midnight": 0,
                "start_time": pd.to_timedelta(8, unit="h"),
                "end_time": pd.to_timedelta(16, unit="h"),
            }
        ]
    )
    month_plan_df = pd.DataFrame(
        [
            {
                "data": "2025-01-01",
                "data_dt": pd.Timestamp("2025-01-01"),
                "reparto_id": "DEP",
                "shift_code": "M",
                "coverage_code": "COV",
            }
        ]
    )
    defaults = {"departments": ["DEP"]}

    slots_df = build_shift_slots(
        month_plan_df,
        shifts_df,
        _empty_dept_map_df(),
        defaults,
    )

    assert slots_df.loc[0, "duration_min"] == 435


def test_load_employees_and_cross_policy_defaults() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_df = load_employees(
        str(DATA_CSV_DIR / 'employees.csv'),
        cfg.get("defaults", {}),
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )

    enriched = enrich_employees_with_cross_policy(employees_df, cfg)
    lookup = enriched.set_index("employee_id")

    cross_cfg = cfg["cross"]
    assert lookup.loc["E001", "cross_max_shifts_month"] == 3
    assert lookup.loc["E002", "cross_max_shifts_month"] == cross_cfg["max_shifts_month"]
    assert lookup.loc["E003", "cross_max_shifts_month"] == 0
    assert "cross_penalty_weight" not in enriched.columns


def test_load_all_enriches_employees_with_fte() -> None:
    loaded = load_all(str(DATA_DIR / "config.yaml"), str(DATA_CSV_DIR))
    employees_df = loaded.employees_df

    assert "fte" in employees_df.columns
    assert "fte_weight" in employees_df.columns

    lookup = employees_df.set_index("employee_id")
    # Baseline infermiere 168h, caposala 150h (see config.yaml).
    assert lookup.loc["E001", "fte"] == pytest.approx(1.0)
    assert lookup.loc["E047", "fte"] == pytest.approx(1.0)
    # Columns should contain finite, non-negative weights.
    assert (lookup["fte_weight"] >= 0).all()


def test_load_employees_applies_default_hour_caps(tmp_path: Path) -> None:
    employees_df = _load_basic_employees(tmp_path)

    assert len(employees_df) == 1
    row = employees_df.iloc[0]

    base_hours = 160
    expected_month_cap_min = int(round(base_hours * 1.25 * 60))
    assert row["max_month_min"] == expected_month_cap_min

    weekly_theoretical = base_hours / 31 * 7
    expected_week_cap_min = int(round(weekly_theoretical * 1.4 * 60))
    assert row["max_week_min"] == expected_week_cap_min


def test_enrich_employees_with_cross_policy_rejects_penalty_override(tmp_path: Path) -> None:
    cfg_path = _write_basic_config(
        tmp_path,
        defaults_extra={
            "departments": ["dep"],
        },
    )
    cfg = load_config(str(cfg_path))
    cfg["cross"] = {"max_shifts_month": 3, "penalty_weight": 1.0}

    employees_df = pd.DataFrame(
        [
            {
                "employee_id": "E1",
                "nome": "Anna",
                "role": "infermiere",
                "reparto_id": "dep",
                "cross_penalty_weight": 0.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="cross_penalty_weight"):
        enrich_employees_with_cross_policy(employees_df, cfg)


def test_build_calendar_extends_to_month_start_for_partial_plans() -> None:
    cal = build_calendar(date(2025, 11, 15), date(2025, 11, 20))
    assert cal["data"].min() == "2025-11-01"
    assert "2025-11-05" in cal["data"].tolist()


def test_build_calendar_keeps_ten_day_history_when_month_starts_horizon() -> None:
    cal = build_calendar(date(2025, 11, 1), date(2025, 11, 3))
    assert cal["data"].min() == "2025-10-22"
    assert "2025-10-22" in cal["data"].tolist()


def test_resolve_fulltime_baseline_reads_defaults() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))

    assert resolve_fulltime_baseline(cfg, "caposala") == pytest.approx(150)
    assert resolve_fulltime_baseline(cfg, "infermiere") == pytest.approx(168)

    with pytest.raises(LoaderError, match="contract_hours_by_role_h non definito per il ruolo sconosciuto"):
        resolve_fulltime_baseline(cfg, "sconosciuto")


def test_resolve_fulltime_baseline_requires_defined_role() -> None:
    cfg = {
        "defaults": {
            "contract_hours_by_role_h": {
                "INFERMIERE": 160,
                "CAPOSALA": 150,
            }
        }
    }

    assert resolve_fulltime_baseline(cfg, "INFERMIERE") == pytest.approx(160)
    assert resolve_fulltime_baseline(cfg, "CAPOSALA") == pytest.approx(150)

    with pytest.raises(LoaderError, match="contract_hours_by_role_h non definito per il ruolo OSS"):
        resolve_fulltime_baseline(cfg, "OSS")


def test_load_employees_rest11h_defaults_and_overrides(tmp_path: Path) -> None:
    cfg_path = _write_basic_config(
        tmp_path,
        {
            "rest11h": {
                "max_monthly_exceptions": 2,
                "max_consecutive_exceptions": 1,
            }
        },
    )
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_path = _write_employees_csv(
        tmp_path,
        [
            {
                "employee_id": "E1",
                "nome": "Anna",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
            },
            {
                "employee_id": "E2",
                "nome": "Marco",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "rest11h_max_monthly_exceptions": 5,
                "rest11h_max_consecutive_exceptions": 3,
            },
            {
                "employee_id": "E3",
                "nome": "Luca",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "rest11h_max_monthly_exceptions": " ",
                "rest11h_max_consecutive_exceptions": "",
            },
        ],
    )

    employees_df = load_employees(
        str(employees_path),
        cfg.get("defaults", {}),
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )

    lookup = employees_df.set_index("employee_id")

    assert lookup.loc["E1", "rest11h_max_monthly_exceptions"] == 2
    assert lookup.loc["E1", "rest11h_max_consecutive_exceptions"] == 1

    assert lookup.loc["E2", "rest11h_max_monthly_exceptions"] == 5
    assert lookup.loc["E2", "rest11h_max_consecutive_exceptions"] == 3

    assert lookup.loc["E3", "rest11h_max_monthly_exceptions"] == 2
    assert lookup.loc["E3", "rest11h_max_consecutive_exceptions"] == 1


def test_absence_full_day_hours_global_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    employees_df = _load_basic_employees(
        tmp_path,
        {"absences": {"fallback_contract_daily_avg_h": 7.5}},
        caplog=caplog,
    )

    row = employees_df.iloc[0]
    assert pd.isna(row["absence_full_day_hours_h"])
    assert row["absence_full_day_hours_effective_h"] == pytest.approx(7.5)

    warning_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.WARNING and "absence_full_day_hours_h" in record.message
    ]
    assert warning_messages, "Expected fallback warning when role mapping is missing"


def test_absence_full_day_hours_role_mapping(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    employees_df = _load_basic_employees(
        tmp_path,
        {
            "absences": {
                "full_day_hours_by_role_h": {"INFERMIERE": 7.25},
                "fallback_contract_daily_avg_h": 7.5,
            }
        },
        caplog=caplog,
    )

    row = employees_df.iloc[0]
    assert row["absence_full_day_hours_effective_h"] == pytest.approx(7.25)

    assert not any(
        record.levelno == logging.WARNING
        and "absence_full_day_hours_h non definito" in record.message
        for record in caplog.records
    )


def test_absence_full_day_hours_employee_override(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    employees_df = _load_basic_employees(
        tmp_path,
        {
            "absences": {
                "full_day_hours_by_role_h": {"infermiere": 7.5},
                "fallback_contract_daily_avg_h": 8.0,
            }
        },
        [
            {
                "employee_id": "E1",
                "nome": "Anna",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "absence_full_day_hours_h": 6.0,
            }
        ],
        caplog=caplog,
    )

    row = employees_df.iloc[0]
    assert row["absence_full_day_hours_h"] == pytest.approx(6.0)
    assert row["absence_full_day_hours_effective_h"] == pytest.approx(6.0)

    assert any(
        record.levelno == logging.INFO
        and "absence_full_day_hours_h override" in record.message
        for record in caplog.records
    )


def test_absence_full_day_hours_invalid_value(tmp_path: Path) -> None:
    with pytest.raises(
        LoaderError,
        match=r"Valore non valido per absence_full_day_hours_h per dipendente",
    ):
        _load_basic_employees(
            tmp_path,
            {
                "absences": {
                    "full_day_hours_by_role_h": {"infermiere": 7.5},
                    "fallback_contract_daily_avg_h": 7.5,
                }
            },
            [
                {
                    "employee_id": "E1",
                    "nome": "Anna",
                    "role": "infermiere",
                    "reparto_id": "dep",
                    "ore_dovute_mese_h": 160,
                    "saldo_prog_iniziale_h": 0,
                    "absence_full_day_hours_h": "abc",
                }
            ],
        )

def test_load_employees_rest11h_invalid_value(tmp_path: Path) -> None:
    cfg_path = _write_basic_config(tmp_path, {"rest11h": {"max_monthly_exceptions": 2}})
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_path = _write_employees_csv(
        tmp_path,
        [
            {
                "employee_id": "E1",
                "nome": "Anna",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "rest11h_max_monthly_exceptions": -1,
            }
        ],
    )

    with pytest.raises(
        LoaderError,
        match="Valore non valido per rest11h_max_monthly_exceptions per dipendente E1: -1",
    ):
        load_employees(
            str(employees_path),
            cfg.get("defaults", {}),
            cfg.get("roles", {}) or {},
            weeks_in_horizon,
            horizon_days,
        )


def test_load_employees_balance_delta_defaults_and_overrides(tmp_path: Path) -> None:
    cfg_path = _write_basic_config(
        tmp_path,
        {"balance": {"max_balance_delta_month_h": 12}},
    )
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_path = _write_employees_csv(
        tmp_path,
        [
            {
                "employee_id": "E1",
                "nome": "Anna",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
            },
            {
                "employee_id": "E2",
                "nome": "Marco",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "max_balance_delta_month_h": 8,
            },
            {
                "employee_id": "E3",
                "nome": "Luca",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "max_balance_delta_month_h": "  ",
            },
        ],
    )

    employees_df = load_employees(
        str(employees_path),
        cfg.get("defaults", {}),
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )

    lookup = employees_df.set_index("employee_id")
    assert lookup.loc["E1", "max_balance_delta_month_h"] == 12
    assert lookup.loc["E2", "max_balance_delta_month_h"] == 8
    assert lookup.loc["E3", "max_balance_delta_month_h"] == 12


def test_load_employees_balance_delta_invalid_employee_value(tmp_path: Path) -> None:
    cfg_path = _write_basic_config(
        tmp_path,
        {"balance": {"max_balance_delta_month_h": 5}},
    )
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_path = _write_employees_csv(
        tmp_path,
        [
            {
                "employee_id": "E1",
                "nome": "Anna",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "max_balance_delta_month_h": -1,
            }
        ],
    )

    with pytest.raises(
        LoaderError,
        match="Valore non valido per max_balance_delta_month_h per dipendente E1: -1",
    ):
        load_employees(
            str(employees_path),
            cfg.get("defaults", {}),
            cfg.get("roles", {}) or {},
            weeks_in_horizon,
            horizon_days,
        )


def test_load_employees_balance_delta_invalid_default(tmp_path: Path) -> None:
    cfg_path = _write_basic_config(
        tmp_path,
        {"balance": {"max_balance_delta_month_h": -3}},
    )
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_path = _write_employees_csv(
        tmp_path,
        [
            {
                "employee_id": "E1",
                "nome": "Anna",
                "role": "infermiere",
                "reparto_id": "dep",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
            }
        ],
    )

    with pytest.raises(
        LoaderError,
        match="config: defaults.balance.max_balance_delta_month_h deve essere un intero ≥ 0",
    ):
        load_employees(
            str(employees_path),
            cfg.get("defaults", {}),
            cfg.get("roles", {}) or {},
            weeks_in_horizon,
            horizon_days,
        )


def test_load_availability_preserves_cross_midnight_rows_spanning_horizon(tmp_path: Path) -> None:
    calendar_df = build_calendar(date(2024, 4, 1), date(2024, 4, 3))
    employees_df = pd.DataFrame({"employee_id": ["E1"]})
    shifts_df = pd.DataFrame(
        {
            "shift_id": ["N"],
            "break_min": [0],
            "duration_min": [600],
            "crosses_midnight": [1],
            "start_time": [pd.to_timedelta(22, unit="h")],
            "end_time": [pd.to_timedelta(6, unit="h")],
        }
    )

    availability_path = tmp_path / "availability.csv"
    availability_path.write_text(
        "data,employee_id,turno\n"
        "2024-03-31,E1,N\n"
        "2024-03-20,E1,N\n"
    )

    out = load_availability(str(availability_path), employees_df, calendar_df, shifts_df)

    mask_cross = out["data"].eq("2024-03-31")
    assert mask_cross.any(), "La riga a cavallo dell'orizzonte deve essere mantenuta"
    row = out.loc[mask_cross].iloc[0]
    assert not bool(row["is_in_horizon"])
    assert row["shift_end_dt"] == pd.Timestamp("2024-04-01 06:00:00")

    assert "2024-03-20" not in out["data"].tolist()


def test_load_leaves_preserves_cross_midnight_rows_spanning_horizon(tmp_path: Path) -> None:
    calendar_df = build_calendar(date(2024, 4, 1), date(2024, 4, 3))
    employees_df = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "absence_full_day_hours_effective_h": [7.5],
        }
    )
    shifts_df = pd.DataFrame(
        {
            "shift_id": ["N"],
            "break_min": [0],
            "duration_min": [600],
            "crosses_midnight": [1],
            "start_time": [pd.to_timedelta(22, unit="h")],
            "end_time": [pd.to_timedelta(6, unit="h")],
        }
    )

    leaves_path = tmp_path / "leaves.csv"
    leaves_path.write_text(
        "employee_id,date_from,date_to,type\n"
        "E1,2024-04-01,2024-04-01,FERIE\n"
    )

    shift_out, _ = load_leaves(str(leaves_path), employees_df, shifts_df, calendar_df)

    mask_cross = shift_out["data"].eq("2024-03-31")
    assert mask_cross.any(), "La riga precedente l'orizzonte deve essere presente"
    row = shift_out.loc[mask_cross].iloc[0]
    assert not bool(row["is_in_horizon"])
    assert row["shift_end_dt"] == pd.Timestamp("2024-04-01 06:00:00")


def test_load_leaves_uses_custom_absence_hours(tmp_path: Path) -> None:
    calendar_df = build_calendar(date(2024, 4, 1), date(2024, 4, 3))
    employees_df = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "absence_full_day_hours_effective_h": [7.5],
        }
    )
    shifts_df = pd.DataFrame(
        {
            "shift_id": ["M"],
            "break_min": [0],
            "duration_min": [480],
            "crosses_midnight": [0],
            "start_time": [pd.to_timedelta(8, unit="h")],
            "end_time": [pd.to_timedelta(16, unit="h")],
        }
    )

    leaves_path = tmp_path / "leaves.csv"
    leaves_path.write_text(
        "employee_id,date_from,date_to,type\n"
        "E1,2024-04-01,2024-04-01,FERIE\n"
    )

    _, day_out = load_leaves(str(leaves_path), employees_df, shifts_df, calendar_df)

    assert not day_out.empty
    assert day_out.loc[0, "absence_hours_h"] == pytest.approx(7.5)


def test_load_leaves_keeps_month_history_prior_to_horizon(tmp_path: Path) -> None:
    calendar_df = build_calendar(date(2024, 4, 15), date(2024, 4, 20))
    employees_df = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "absence_full_day_hours_effective_h": [6.0],
        }
    )
    shifts_df = pd.DataFrame(
        {
            "shift_id": ["M"],
            "break_min": [0],
            "duration_min": [480],
            "crosses_midnight": [0],
            "start_time": [pd.to_timedelta(8, unit="h")],
            "end_time": [pd.to_timedelta(16, unit="h")],
        }
    )

    leaves_path = tmp_path / "leaves.csv"
    leaves_path.write_text(
        "employee_id,date_from,date_to,type\n"
        "E1,2024-04-05,2024-04-05,FERIE\n"
    )

    _, day_out = load_leaves(str(leaves_path), employees_df, shifts_df, calendar_df)

    assert "2024-04-05" in day_out["data"].tolist()
    row = day_out.loc[day_out["data"].eq("2024-04-05")].iloc[0]
    assert not bool(row["is_in_horizon"])
    assert row["absence_hours_h"] == pytest.approx(6.0)


def test_load_leaves_uses_fallback_absence_hours(tmp_path: Path) -> None:
    calendar_df = build_calendar(date(2024, 4, 1), date(2024, 4, 3))
    employees_df = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "absence_full_day_hours_effective_h": [float("nan")],
        }
    )
    shifts_df = pd.DataFrame(
        {
            "shift_id": ["M"],
            "break_min": [0],
            "duration_min": [480],
            "crosses_midnight": [0],
            "start_time": [pd.to_timedelta(8, unit="h")],
            "end_time": [pd.to_timedelta(16, unit="h")],
        }
    )

    leaves_path = tmp_path / "leaves.csv"
    leaves_path.write_text(
        "employee_id,date_from,date_to,type\n"
        "E1,2024-04-01,2024-04-01,FERIE\n"
    )

    _, day_out = load_leaves(
        str(leaves_path),
        employees_df,
        shifts_df,
        calendar_df,
        absence_hours_h=7.25,
    )

    assert not day_out.empty
    assert day_out.loc[0, "absence_hours_h"] == pytest.approx(7.25)


def test_apply_unplanned_leave_durations_overrides_hours(tmp_path: Path) -> None:
    calendar_df = build_calendar(date(2024, 4, 1), date(2024, 4, 3))
    employees_df = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "absence_full_day_hours_effective_h": [7.5],
        }
    )
    shifts_df = pd.DataFrame(
        {
            "shift_id": ["M", "P"],
            "break_min": [0, 0],
            "duration_min": [480, 420],
            "crosses_midnight": [0, 0],
            "start_time": [pd.to_timedelta(8, unit="h"), pd.to_timedelta(14, unit="h")],
            "end_time": [pd.to_timedelta(16, unit="h"), pd.to_timedelta(22, unit="h")],
        }
    )

    leaves_path = tmp_path / "leaves.csv"
    leaves_path.write_text(
        "employee_id,date_from,date_to,type,is_planned\n"
        "E1,2024-04-01,2024-04-01,FERIE,false\n",
        encoding="utf-8",
    )

    shift_out, day_out = load_leaves(
        str(leaves_path),
        employees_df,
        shifts_df,
        calendar_df,
    )

    assert pytest.approx(day_out.loc[0, "absence_hours_h"], rel=1e-6) == 7.5

    preassignments_df = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "date": [date(2024, 4, 1)],
            "state_code": ["P"],
        }
    )

    adjusted_shift, adjusted_day = apply_unplanned_leave_durations(
        shift_out,
        day_out,
        preassignments_df,
        shifts_df,
    )

    target_row = adjusted_day.loc[adjusted_day["data"].eq("2024-04-01")].iloc[0]
    assert target_row["absence_hours_h"] == pytest.approx(7.0)
    assert not bool(target_row["is_planned"])

    same_day = adjusted_shift[adjusted_shift["data"].eq("2024-04-01")]
    assert not same_day.empty
    p_row = same_day.loc[same_day["turno"].eq("P")].iloc[0]
    assert p_row["shift_duration_min"] == pytest.approx(420)
    assert not bool(p_row["is_planned"])
    m_row = same_day.loc[same_day["turno"].eq("M")].iloc[0]
    assert m_row["shift_duration_min"] == pytest.approx(0.0)


def test_load_all_applies_unplanned_absence_hours(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        (DATA_DIR / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for src in DATA_CSV_DIR.glob("*.csv"):
        shutil.copy(src, data_dir / src.name)

    leaves_path = data_dir / "leaves.csv"
    leaves_df = pd.read_csv(leaves_path)
    mask = (leaves_df["employee_id"] == "E006") & (
        leaves_df["date_from"] == "2025-11-10"
    )
    assert mask.any(), "leaves.csv sample should contain E006 on 2025-11-10"
    leaves_df.loc[mask, "is_planned"] = False
    leaves_df.to_csv(leaves_path, index=False)

    pd.DataFrame(
        {
            "employee_id": ["E006"],
            "date": ["2025-11-10"],
            "state_code": ["N"],
        }
    ).to_csv(data_dir / "preassignments.csv", index=False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"locks.csv: caricati .*",
            category=UserWarning,
        )
        loaded = load_all(str(cfg_path), str(data_dir))

    day_rows = loaded.leaves_days_df
    assert not day_rows.empty

    target_day = day_rows[
        (day_rows["employee_id"] == "E006")
        & (day_rows["data"] == "2025-11-10")
    ].iloc[0]
    assert target_day["absence_hours_h"] == pytest.approx(10.5)
    assert not bool(target_day["is_planned"])

    fallback_day = day_rows[
        (day_rows["employee_id"] == "E006")
        & (day_rows["data"] == "2025-11-11")
    ].iloc[0]
    assert fallback_day["absence_hours_h"] == pytest.approx(7.5)

    shift_rows = loaded.leaves_df[
        (loaded.leaves_df["employee_id"] == "E006")
        & (loaded.leaves_df["data"] == "2025-11-10")
    ]
    assert not shift_rows.empty

    night_row = shift_rows.loc[shift_rows["turno"].eq("N")].iloc[0]
    assert night_row["shift_duration_min"] == pytest.approx(630.0)
    assert not bool(night_row["is_planned"])

    others = shift_rows.loc[~shift_rows["turno"].eq("N")]
    assert not others.empty
    assert (others["shift_duration_min"] == 0.0).all()
def test_enrich_employees_with_fte_uses_defaults() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)
    employees_df = load_employees(
        str(DATA_CSV_DIR / 'employees.csv'),
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


def test_load_employees_weekly_rest_uses_default(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg_path = _write_basic_config(tmp_path)
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_path = _write_employees_csv(
        tmp_path,
        [
            {
                "employee_id": "E001",
                "nome": "Mario Rossi",
                "reparto_id": "dep",
                "role": "infermiere",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
            }
        ],
    )

    with caplog.at_level(logging.INFO, logger="loader.employees"):
        out = load_employees(
            str(employees_path),
            cfg.get("defaults", {}),
            cfg.get("roles", {}) or {},
            weeks_in_horizon,
            horizon_days,
        )

    assert out.loc[0, "weekly_rest_min_days"] == 1
    assert any(
        "weekly_rest_min_days default applicato" in message
        for message in caplog.messages
    )


def test_load_employees_weekly_rest_override(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg_path = _write_basic_config(tmp_path, defaults_extra={"weekly_rest_min_days": 2})
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_path = _write_employees_csv(
        tmp_path,
        [
            {
                "employee_id": "E002",
                "nome": "Luigi Verdi",
                "reparto_id": "dep",
                "role": "infermiere",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "weekly_rest_min_days": 5,
            }
        ],
    )

    with caplog.at_level(logging.INFO, logger="loader.employees"):
        out = load_employees(
            str(employees_path),
            cfg.get("defaults", {}),
            cfg.get("roles", {}) or {},
            weeks_in_horizon,
            horizon_days,
        )

    assert out.loc[0, "weekly_rest_min_days"] == 5
    assert any(
        "weekly_rest_min_days override" in message for message in caplog.messages
    )


@pytest.mark.parametrize("invalid_value", ["-1", "abc"])
def test_load_employees_weekly_rest_invalid_values(invalid_value: str, tmp_path: Path) -> None:
    cfg_path = _write_basic_config(tmp_path)
    cfg = load_config(str(cfg_path))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)

    employees_path = _write_employees_csv(
        tmp_path,
        [
            {
                "employee_id": "E003",
                "nome": "Anna Bianchi",
                "reparto_id": "dep",
                "role": "infermiere",
                "ore_dovute_mese_h": 160,
                "saldo_prog_iniziale_h": 0,
                "weekly_rest_min_days": invalid_value,
            }
        ],
    )

    match = re.escape(
        f"Valore non valido per weekly_rest_min_days per dipendente E003: {invalid_value}"
    )
    with pytest.raises(LoaderError, match=match):
        load_employees(
            str(employees_path),
            cfg.get("defaults", {}),
            cfg.get("roles", {}) or {},
            weeks_in_horizon,
            horizon_days,
        )


def test_get_absence_hours_from_config_warns_on_payroll(caplog: pytest.LogCaptureFixture) -> None:
    cfg = {"payroll": {"absence_hours_h": 7}}

    with caplog.at_level(logging.WARNING, logger="loader.absences"):
        value = get_absence_hours_from_config(cfg)

    assert value == pytest.approx(7.0)
    assert any("deprecato" in message for message in caplog.messages)


def test_load_config_warns_on_weekly_rest_hours(tmp_path: Path) -> None:
    cfg_path = _write_basic_config(tmp_path, defaults_extra={"weekly_rest_min_h": 48})

    with pytest.warns(UserWarning, match="defaults.weekly_rest_min_h è deprecato"):
        cfg = load_config(str(cfg_path))

    defaults = cfg.get("defaults", {})
    assert "weekly_rest_min_h" not in defaults
    assert defaults["weekly_rest_min_days"] == 1


def test_shift_role_eligibility_with_allowed_column() -> None:
    cfg = load_config(str(DATA_DIR / "config.yaml"))
    horizon_days, weeks_in_horizon = _calendar_info(cfg)
    employees_df = load_employees(
        str(DATA_CSV_DIR / 'employees.csv'),
        cfg.get("defaults", {}),
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )
    shifts_df = load_shifts(str(DATA_CSV_DIR / 'shifts.csv'))
    eligibility_df = load_shift_role_eligibility(
        str(DATA_CSV_DIR / 'shift_role_eligibility.csv'),
        employees_df,
        shifts_df,
        cfg.get("defaults", {}),
    )

    mask = (eligibility_df["shift_code"] == "N") & (
        eligibility_df["role"] == "CAPOSALA"
    )
    assert mask.any()
    assert not bool(eligibility_df.loc[mask, "allowed"].iloc[0])


def test_load_coverage_groups_applies_default_overstaff_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg_path = _write_basic_config(
        tmp_path,
        defaults_extra={"overstaffing": {"enabled": True, "group_cap_default": 3}},
    )
    cfg = load_config(str(cfg_path))

    coverage_groups_path = _write_coverage_groups_csv(
        tmp_path,
        [
            {
                "coverage_code": "COV",
                "shift_code": "S1",
                "reparto_id": "DEP",
                "gruppo": "G1",
                "total_staff": 5,
                "ruoli_totale": "INFERMIERE",
                "overstaff_cap": "",
            }
        ],
    )

    caplog.set_level(logging.INFO, logger="loader.coverage")
    df = load_coverage_groups(str(coverage_groups_path), cfg.get("defaults", {}))

    assert df.loc[0, "overstaff_cap_effective"] == 8
    assert "overstaffing abilitato" in " ".join(caplog.messages)


def test_expand_requirements_propagates_overstaff_cap(tmp_path: Path) -> None:
    cfg_path = _write_basic_config(
        tmp_path,
        defaults_extra={"overstaffing": {"enabled": True, "group_cap_default": 4}},
    )
    cfg = load_config(str(cfg_path))

    coverage_groups_path = _write_coverage_groups_csv(
        tmp_path,
        [
            {
                "coverage_code": "COV",
                "shift_code": "S1",
                "reparto_id": "DEP",
                "gruppo": "G1",
                "total_staff": 6,
                "ruoli_totale": "INFERMIERE",
                "overstaff_cap": "",
            }
        ],
    )
    groups_df = load_coverage_groups(str(coverage_groups_path), cfg.get("defaults", {}))

    month_plan = pd.DataFrame(
        [
            {
                "data": "2025-01-01",
                "reparto_id": "DEP",
                "shift_code": "S1",
                "coverage_code": "COV",
            }
        ]
    )
    roles_df = pd.DataFrame(
        [
            {
                "coverage_code": "COV",
                "shift_code": "S1",
                "reparto_id": "DEP",
                "gruppo": "G1",
                "role": "INFERMIERE",
                "min_ruolo": 2,
            }
        ]
    )

    groups_total, roles_total = expand_requirements(month_plan, groups_df, roles_df)

    assert "overstaff_cap_effective" in groups_total.columns
    assert groups_total.loc[0, "overstaff_cap_effective"] == 10
    assert roles_total.loc[0, "gruppo"] == "G1"


def test_load_preassignments_ignores_out_of_horizon(tmp_path: Path) -> None:
    calendar = build_calendar(date(2025, 11, 15), date(2025, 11, 16))
    employees = pd.DataFrame({"employee_id": ["E1"]})
    pre_df = pd.DataFrame(
        {
            "employee_id": ["E1", "E1"],
            "data": ["2025-11-14", "2025-11-15"],
            "state_code": ["M", "P"],
        }
    )
    path = tmp_path / "preassignments.csv"
    pre_df.to_csv(path, index=False)

    loaded = load_preassignments(str(path), employees, calendar)

    assert list(loaded["data"]) == ["2025-11-15"]
    assert list(loaded["state_code"]) == ["P"]


def test_get_absence_hours_from_config_and_load_all_smoke(tmp_path: Path) -> None:
    cfg = load_config(str(DATA_DIR / 'config.yaml'))
    assert get_absence_hours_from_config(cfg) == pytest.approx(6.0)

    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text((DATA_DIR / 'config.yaml').read_text(encoding='utf-8'), encoding='utf-8')

    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    for src in DATA_CSV_DIR.glob('*.csv'):
        shutil.copy(src, data_dir / src.name)

    pd.DataFrame(
        columns=['date', 'reparto_id', 'shift_code', 'employee_id', 'lock_type']
    ).to_csv(data_dir / 'locks.csv', index=False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'locks.csv: caricati .*',
            category=UserWarning,
        )
        loaded = load_all(str(cfg_path), str(data_dir))
    assert not loaded.employees_df.empty
    assert {'cross_max_shifts_month'}.issubset(loaded.employees_df.columns)
    assert 'cross_penalty_weight' not in loaded.employees_df.columns
    assert not loaded.leaves_df.empty
    assert not loaded.availability_df.empty
