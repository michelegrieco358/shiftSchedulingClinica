from __future__ import annotations

import os
from dataclasses import dataclass

try:  # pragma: no cover - dipendenza runtime
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - messaggio esplicativo
    raise ModuleNotFoundError(
        "Il pacchetto 'pandas' è richiesto per eseguire il loader. "
        "Installarlo con `pip install -r requirements.txt`."
    ) from exc

from .absences import (
    build_absence_masks,
    explode_absences_by_day,
    get_absence_hours_from_config,
    load_absences,
)
from .availability import load_availability
from .calendar import attach_calendar, build_calendar, enrich_shift_slots_calendar
from .config import load_config, load_holidays
from .coverage import (
    build_slot_requirements,
    expand_requirements,
    load_coverage_groups,
    load_coverage_roles,
    load_month_plan,
    validate_groups_roles,
)
from .employees import (
    build_department_compatibility,
    load_employees,
    load_role_dept_pools,
)
from .gap_pairs import build_gap_pairs
from .history import load_history
from .leaves import load_leaves
from .shifts import (
    build_shift_slots,
    load_department_shift_map,
    load_shift_role_eligibility,
    load_shifts,
)
from .utils import LoaderError, _parse_date


@dataclass
class LoadedData:
    cfg: dict
    calendar_df: pd.DataFrame
    employees_df: pd.DataFrame
    shifts_df: pd.DataFrame
    shift_slots_df: pd.DataFrame
    slot_requirements_df: pd.DataFrame
    eligibility_df: pd.DataFrame
    month_plan_df: pd.DataFrame
    groups_total_expanded: pd.DataFrame
    groups_role_min_expanded: pd.DataFrame
    history_df: pd.DataFrame
    availability_df: pd.DataFrame
    leaves_df: pd.DataFrame
    leaves_days_df: pd.DataFrame
    holidays_df: pd.DataFrame
    role_dept_pools_df: pd.DataFrame
    dept_compat_df: pd.DataFrame
    gap_pairs_df: pd.DataFrame


def load_all(config_path: str, data_dir: str) -> LoadedData:
    cfg = load_config(config_path)
    start_date = _parse_date(cfg["horizon"]["start_date"])
    end_date = _parse_date(cfg["horizon"]["end_date"])

    defaults = cfg.get("defaults", {})

    holidays_df = load_holidays(os.path.join(data_dir, "holidays.csv"))

    calendar_df = build_calendar(
        start_date, end_date, holidays_df if not holidays_df.empty else None
    )

    horizon_days = int(calendar_df["is_in_horizon"].sum())
    weeks_in_horizon = (
        calendar_df.loc[calendar_df["is_in_horizon"], "week_id"].nunique()
    )
    if weeks_in_horizon <= 0:
        raise LoaderError(
            "calendar: nessuna settimana trovata nell'orizzonte configurato"
        )
    if horizon_days <= 0:
        raise LoaderError(
            "calendar: nessun giorno trovato nell'orizzonte configurato"
        )

    employees_df = load_employees(
        os.path.join(data_dir, "employees.csv"),
        defaults,
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )
    shifts_df = load_shifts(os.path.join(data_dir, "shifts.csv"))
    dept_shift_map_df = load_department_shift_map(
        os.path.join(data_dir, "reparto_shift_map.csv"), defaults, shifts_df
    )
    eligibility_df = load_shift_role_eligibility(
        os.path.join(data_dir, "shift_role_eligibility.csv"), employees_df, shifts_df, defaults
    )
    role_dept_pools_df = load_role_dept_pools(
        os.path.join(data_dir, "role_dept_pools.csv"),
        defaults,
        employees_df,
    )
    dept_compat_df = build_department_compatibility(
        defaults,
        role_dept_pools_df,
        employees_df,
    )
    month_plan_df = load_month_plan(
        os.path.join(data_dir, "month_plan.csv"), shifts_df, defaults
    )
    groups_df = load_coverage_groups(os.path.join(data_dir, "coverage_groups.csv"))
    roles_df = load_coverage_roles(os.path.join(data_dir, "coverage_roles.csv"))
    validate_groups_roles(groups_df, roles_df, eligibility_df)

    month_plan_df = attach_calendar(month_plan_df, calendar_df)
    shift_slots_df = build_shift_slots(
        month_plan_df, shifts_df, dept_shift_map_df, defaults
    )

    groups_total_expanded, groups_role_min_expanded = expand_requirements(
        month_plan_df, groups_df, roles_df
    )
    groups_total_expanded = attach_calendar(groups_total_expanded, calendar_df)
    groups_role_min_expanded = attach_calendar(groups_role_min_expanded, calendar_df)

    slot_requirements_df = build_slot_requirements(shift_slots_df, roles_df)

    rest_rules = cfg.get("rest_rules", {})
    max_gap_window = max(
        [
            value
            for value in (
                rest_rules.get("post_night_rest_h"),
                rest_rules.get("weekly_rest_min_h"),
                rest_rules.get("min_between_shifts_h"),
            )
            if isinstance(value, (int, float)) and value > 0
        ]
        or [15]
    )
    gap_pairs_df = build_gap_pairs(
        shift_slots_df,
        max_check_window_h=int(max_gap_window),
        add_debug=False,
    )

    history_df = load_history(
        os.path.join(data_dir, "history.csv"),
        employees_df,
        shifts_df,
        calendar_df,
    )
    leaves_df, leaves_days_df = load_leaves(
        os.path.join(data_dir, "leaves.csv"),
        employees_df,
        shifts_df,
        calendar_df,
    )
    availability_df = load_availability(
        os.path.join(data_dir, "availability.csv"),
        employees_df,
        calendar_df,
        shifts_df,
    )

    return LoadedData(
        cfg=cfg,
        calendar_df=calendar_df,
        employees_df=employees_df,
        shifts_df=shifts_df,
        shift_slots_df=shift_slots_df,
        slot_requirements_df=slot_requirements_df,
        eligibility_df=eligibility_df,
        month_plan_df=month_plan_df,
        groups_total_expanded=groups_total_expanded,
        groups_role_min_expanded=groups_role_min_expanded,
        history_df=history_df,
        availability_df=availability_df,
        leaves_df=leaves_df,
        leaves_days_df=leaves_days_df,
        holidays_df=holidays_df,
        role_dept_pools_df=role_dept_pools_df,
        dept_compat_df=dept_compat_df,
        gap_pairs_df=gap_pairs_df,
    )


__all__ = [
    "LoaderError",
    "LoadedData",
    "load_all",
    "attach_calendar",
    "build_absence_masks",
    "build_calendar",
    "explode_absences_by_day",
    "get_absence_hours_from_config",
    "enrich_shift_slots_calendar",
    "load_absences",
]
