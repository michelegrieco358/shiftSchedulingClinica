from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

DATA_DIR = Path(__file__).resolve().parents[1]

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from loader import load_all
from loader.availability import load_availability
from loader.absences import get_absence_hours_from_config
from loader.calendar import build_calendar
from loader.config import load_config
from loader.cross import enrich_employees_with_cross_policy
from loader.employees import (
    enrich_employees_with_fte,
    load_employees,
    resolve_fulltime_baseline,
)
from loader.leaves import load_leaves
from loader.shifts import load_shift_role_eligibility, load_shifts
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
    employees_df = pd.DataFrame({"employee_id": ["E1"]})
    shifts_df = pd.DataFrame(
        {
            "shift_id": ["N"],
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
    employees_df = pd.DataFrame({"employee_id": ["E1"]})
    shifts_df = pd.DataFrame(
        {
            "shift_id": ["M"],
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
        absence_hours_h=7.5,
    )

    assert not day_out.empty
    assert day_out.loc[0, "absence_hours_h"] == pytest.approx(7.5)
def test_enrich_employees_with_fte_uses_defaults() -> None:
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
