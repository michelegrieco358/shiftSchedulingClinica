from __future__ import annotations

import os

import pandas as pd

from .absences import explode_absences_by_day, load_absences
from .calendar import attach_calendar
from .utils import LoaderError, _compute_horizon_window


def load_leaves(
    path: str,
    employees_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    *,
    absence_hours_h: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_turns = tuple(
        pd.Series(shifts_df.loc[shifts_df["duration_min"] > 0, "shift_id"])
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if not allowed_turns:
        raise LoaderError(
            "shifts.csv: nessun turno con duration_min>0 — impossibile espandere leaves.csv."
        )

    shift_columns = [
        "employee_id",
        "data",
        "data_dt",
        "turno",
        "tipo",
        "is_planned",
        "shift_start_time",
        "shift_end_time",
        "shift_start_dt",
        "shift_end_dt",
        "shift_duration_min",
        "shift_crosses_midnight",
        "dow_iso",
        "week_start_date",
        "week_start_date_dt",
        "week_id",
        "week_idx",
        "is_in_horizon",
        "is_weekend",
        "is_weekday_holiday",
        "holiday_desc",
    ]

    day_columns = [
        "employee_id",
        "data",
        "data_dt",
        "tipo_set",
        "is_leave_day",
        "is_absent",
        "absence_hours_h",
        "is_planned",
        "dow_iso",
        "week_start_date",
        "week_start_date_dt",
        "week_id",
        "week_idx",
        "is_in_horizon",
        "is_weekend",
        "is_weekday_holiday",
        "holiday_desc",
    ]

    if not os.path.exists(path):
        empty_shift = pd.DataFrame(columns=shift_columns)
        empty_day = pd.DataFrame(columns=day_columns)
        return empty_shift, empty_day

    absences_df = load_absences(path)

    if absence_hours_h is not None:
        try:
            absence_hours_h = float(absence_hours_h)
        except (TypeError, ValueError) as exc:
            raise LoaderError("absence_hours_h deve essere numerico") from exc
        if absence_hours_h <= 0:
            raise LoaderError("absence_hours_h deve essere positivo")

    horizon_start_ts, horizon_end_ts = _compute_horizon_window(calendar_df)
    horizon_start_date = horizon_start_ts.date()
    horizon_end_date = (horizon_end_ts - pd.Timedelta(days=1)).date()
    month_start_date = horizon_start_date.replace(day=1)

    employees_df = employees_df.copy()
    employees_df["employee_id"] = employees_df["employee_id"].astype(str).str.strip()

    if "absence_full_day_hours_effective_h" not in employees_df.columns:
        raise LoaderError(
            "employees_df deve contenere la colonna absence_full_day_hours_effective_h"
        )

    hours_series = pd.to_numeric(
        employees_df.set_index("employee_id")["absence_full_day_hours_effective_h"],
        errors="coerce",
    )

    abs_employee_ids = absences_df["employee_id"].unique().tolist()
    hours_for_absences = hours_series.reindex(abs_employee_ids)
    if absence_hours_h is not None:
        hours_for_absences = hours_for_absences.fillna(absence_hours_h)

    known_emp = set(employees_df["employee_id"].unique())
    unknown = sorted(set(absences_df["employee_id"].unique()) - known_emp)
    if unknown:
        raise LoaderError(f"leaves.csv: employee_id sconosciuti: {unknown}")

    missing_hours = sorted(
        {emp for emp, value in hours_for_absences.items() if pd.isna(value)}
    )
    if missing_hours:
        raise LoaderError(
            "leaves.csv: ore di assenza non definite per i dipendenti: "
            f"{missing_hours}"
        )

    invalid_hours = sorted(
        {
            emp for emp, value in hours_for_absences.items() if value is not None and value <= 0
        }
    )
    if invalid_hours:
        raise LoaderError(
            "leaves.csv: ore di assenza non positive per i dipendenti: "
            f"{invalid_hours}"
        )

    hours_series = hours_series.copy()
    for emp, value in hours_for_absences.items():
        hours_series.loc[emp] = value

    absences_df["start_date_dt"] = pd.to_datetime(absences_df["date_from"], format="%Y-%m-%d")
    absences_df["end_date_dt"] = pd.to_datetime(absences_df["date_to"], format="%Y-%m-%d")
    absences_df["tipo"] = absences_df["type"]
    explode_fallback = (
        float(hours_for_absences.min())
        if hours_for_absences.size and not pd.isna(hours_for_absences.min())
        else (absence_hours_h if absence_hours_h is not None else 6.0)
    )
    day_explode_columns = ["employee_id", "date_from", "date_to", "type"]
    if "is_planned" in absences_df.columns:
        day_explode_columns.append("is_planned")

    abs_by_day = explode_absences_by_day(
        absences_df.loc[:, day_explode_columns],
        min_date=min(month_start_date, horizon_start_date),
        max_date=horizon_end_date,
        absence_hours_h=explode_fallback,
    )

    if not abs_by_day.empty:
        abs_by_day["employee_id"] = abs_by_day["employee_id"].astype(str).str.strip()
        abs_by_day["absence_hours_h"] = abs_by_day["employee_id"].map(hours_series)
        if "is_planned" not in abs_by_day.columns:
            abs_by_day["is_planned"] = True

    shift_info = shifts_df.loc[
        shifts_df["shift_id"].isin(allowed_turns),
        ["shift_id", "start_time", "end_time", "crosses_midnight"],
    ].copy()
    shift_rows = list(shift_info.itertuples(index=False))

    records = []
    for row in absences_df.itertuples(index=False):
        absence_start_day = row.start_date_dt.normalize()
        absence_end_day = row.end_date_dt.normalize()
        absence_interval_start = absence_start_day
        absence_interval_end = absence_end_day + pd.Timedelta(days=1)

        day = absence_start_day - pd.Timedelta(days=1)
        last_day = absence_end_day

        while day <= last_day:
            day_str = day.date().isoformat()
            for shift in shift_rows:
                if pd.isna(shift.start_time) or pd.isna(shift.end_time):
                    continue

                shift_start_dt = day + shift.start_time
                shift_end_dt = day + shift.end_time
                if int(shift.crosses_midnight) == 1:
                    shift_end_dt = shift_end_dt + pd.Timedelta(days=1)

                if shift_end_dt > absence_interval_start and shift_start_dt < absence_interval_end:
                    records.append(
                        {
                            "employee_id": row.employee_id,
                            "data": day_str,
                            "turno": shift.shift_id,
                            "tipo": row.tipo,
                            "is_planned": getattr(row, "is_planned", True),
                        }
                    )
            day += pd.Timedelta(days=1)

    if records:
        shift_out = pd.DataFrame.from_records(records)
        shift_out["data_dt"] = pd.to_datetime(shift_out["data"], format="%Y-%m-%d")

        shift_out = shift_out.drop_duplicates(
            subset=["employee_id", "data", "turno", "tipo"]
        ).reset_index(drop=True)

        shift_out = attach_calendar(shift_out, calendar_df)

        shift_cols = [
            "shift_id",
            "start_time",
            "end_time",
            "duration_min",
            "crosses_midnight",
        ]
        shift_info = shifts_df[shift_cols].rename(
            columns={
                "shift_id": "turno",
                "start_time": "shift_start_time",
                "end_time": "shift_end_time",
                "duration_min": "shift_duration_min",
                "crosses_midnight": "shift_crosses_midnight",
            }
        )

        shift_out = shift_out.merge(shift_info, on="turno", how="left", validate="many_to_one")

        shift_out["shift_start_dt"] = shift_out["data_dt"] + shift_out["shift_start_time"]
        shift_out["shift_end_dt"] = shift_out["data_dt"] + shift_out["shift_end_time"]
        crosses_mask = shift_out["shift_crosses_midnight"].fillna(0).astype(int) == 1
        shift_out.loc[crosses_mask, "shift_end_dt"] = (
            shift_out.loc[crosses_mask, "shift_end_dt"] + pd.Timedelta(days=1)
        )

        history_mask = pd.Series(False, index=shift_out.index)
        if "data_dt" in shift_out.columns:
            data_dates = pd.to_datetime(shift_out["data_dt"], errors="coerce")
            history_mask = data_dates.dt.date.ge(month_start_date) & data_dates.dt.date.lt(
                horizon_start_date
            )
            history_mask = history_mask.fillna(False)

        overlaps = shift_out["shift_start_dt"].notna() & shift_out["shift_end_dt"].notna()
        overlaps &= shift_out["shift_end_dt"] > horizon_start_ts
        overlaps &= shift_out["shift_start_dt"] < horizon_end_ts

        in_horizon = shift_out["is_in_horizon"].astype("boolean", copy=False).fillna(False)
        keep_mask = in_horizon.astype(bool) | overlaps | history_mask.astype(bool)
        shift_out = shift_out.loc[keep_mask].copy()

        shift_out = shift_out[
            [
                "employee_id",
                "data",
                "data_dt",
                "turno",
                "tipo",
                "is_planned",
                "shift_start_time",
                "shift_end_time",
                "shift_start_dt",
                "shift_end_dt",
                "shift_duration_min",
                "shift_crosses_midnight",
                "dow_iso",
                "week_start_date",
                "week_start_date_dt",
                "week_id",
                "week_idx",
                "is_in_horizon",
                "is_weekend",
                "is_weekday_holiday",
                "holiday_desc",
            ]
        ].sort_values(["data", "employee_id", "turno"]).reset_index(drop=True)
    else:
        shift_out = pd.DataFrame(columns=shift_columns)

    if not abs_by_day.empty:
        day_out = abs_by_day.copy()
        day_out["data"] = day_out["date"].apply(lambda x: x.isoformat())
        day_out["tipo"] = day_out["type"]
        day_out["data_dt"] = pd.to_datetime(day_out["data"], format="%Y-%m-%d")
        day_out = attach_calendar(day_out, calendar_df)

        history_mask = pd.Series(False, index=day_out.index)
        if "data_dt" in day_out.columns:
            day_dates = pd.to_datetime(day_out["data_dt"], errors="coerce")
            history_mask = day_dates.dt.date.ge(month_start_date) & day_dates.dt.date.lt(
                horizon_start_date
            )
            history_mask = history_mask.fillna(False)

        horizon_mask = day_out["is_in_horizon"].astype("boolean", copy=False).fillna(False)
        keep_mask = horizon_mask.astype(bool) | history_mask.astype(bool)
        day_out = day_out.loc[keep_mask].copy()

        def _join_types(values: pd.Series) -> str:
            unique_vals = sorted({str(v).strip() for v in values if str(v).strip()})
            return "|".join(unique_vals)

        day_out = (
            day_out.groupby(["employee_id", "data"], as_index=False)
            .agg(
                data_dt=("data_dt", "first"),
                tipo_set=("tipo", _join_types),
                is_absent=("is_absent", "max"),
                absence_hours_h=("absence_hours_h", "max"),
                is_planned=("is_planned", "max"),
                dow_iso=("dow_iso", "first"),
                week_start_date=("week_start_date", "first"),
                week_start_date_dt=("week_start_date_dt", "first"),
                week_id=("week_id", "first"),
                week_idx=("week_idx", "first"),
                is_in_horizon=("is_in_horizon", "first"),
                is_weekend=("is_weekend", "first"),
                is_weekday_holiday=("is_weekday_holiday", "first"),
                holiday_desc=("holiday_desc", "first"),
            )
        )
        day_out["is_leave_day"] = 1
        day_out["is_absent"] = day_out["is_absent"].fillna(False).astype(bool)
        day_out["absence_hours_h"] = day_out["absence_hours_h"].fillna(0.0)
        day_out["is_planned"] = day_out["is_planned"].fillna(True).astype(bool)
        day_out = day_out[
            [
                "employee_id",
                "data",
                "data_dt",
                "tipo_set",
                "is_leave_day",
                "is_absent",
                "absence_hours_h",
                "is_planned",
                "dow_iso",
                "week_start_date",
                "week_start_date_dt",
                "week_id",
                "week_idx",
                "is_in_horizon",
                "is_weekend",
                "is_weekday_holiday",
                "holiday_desc",
            ]
        ].sort_values(["data", "employee_id"]).reset_index(drop=True)
    else:
        day_out = pd.DataFrame(columns=day_columns)

    return shift_out, day_out


def apply_unplanned_leave_durations(
    leaves_df: pd.DataFrame,
    leaves_days_df: pd.DataFrame,
    preassignments_df: pd.DataFrame | None,
    shifts_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Override absence hours for unplanned leaves using preassigned shifts."""

    if leaves_days_df is None or leaves_days_df.empty:
        return leaves_df, leaves_days_df
    if "is_planned" not in leaves_days_df.columns:
        return leaves_df, leaves_days_df
    planned_series = leaves_days_df["is_planned"].astype("boolean", copy=False)
    if planned_series.fillna(True).all():
        return leaves_df, leaves_days_df
    if preassignments_df is None or preassignments_df.empty:
        return leaves_df, leaves_days_df
    if "employee_id" not in preassignments_df.columns or "state_code" not in preassignments_df.columns:
        return leaves_df, leaves_days_df

    day_work = leaves_days_df.copy()
    day_work["employee_id"] = day_work["employee_id"].astype(str).str.strip()
    if "data_dt" in day_work.columns:
        day_work["_date"] = pd.to_datetime(day_work["data_dt"], errors="coerce").dt.date
    else:
        day_work["_date"] = pd.to_datetime(day_work["data"], errors="coerce").dt.date
    day_work["absence_hours_h"] = pd.to_numeric(
        day_work.get("absence_hours_h"), errors="coerce"
    ).fillna(0.0)

    unplanned_mask = ~planned_series.fillna(True)
    if "is_in_horizon" in day_work.columns:
        unplanned_mask &= day_work["is_in_horizon"].astype("boolean", copy=False).fillna(False)
    if "is_absent" in day_work.columns:
        unplanned_mask &= day_work["is_absent"].astype("boolean", copy=False).fillna(True)
    unplanned_mask &= day_work["_date"].notna()

    if not unplanned_mask.any():
        day_work = day_work.drop(columns="_date")
        return leaves_df, day_work

    shift_minutes_map: dict[str, float] = {}
    if shifts_df is not None and not shifts_df.empty and {"shift_id", "duration_min"}.issubset(shifts_df.columns):
        mapping_frame = shifts_df.loc[:, ["shift_id", "duration_min"]].copy()
        mapping_frame["shift_id"] = mapping_frame["shift_id"].astype(str).str.strip().str.upper()
        mapping_frame["duration_min"] = pd.to_numeric(
            mapping_frame["duration_min"], errors="coerce"
        )
        mapping_frame = mapping_frame.dropna(subset=["shift_id", "duration_min"])
        shift_minutes_map = mapping_frame.set_index("shift_id")["duration_min"].to_dict()
    elif leaves_df is not None and not leaves_df.empty:
        if {"turno", "shift_duration_min"}.issubset(leaves_df.columns):
            mapping_frame = leaves_df.loc[:, ["turno", "shift_duration_min"]].copy()
            mapping_frame["turno"] = mapping_frame["turno"].astype(str).str.strip().str.upper()
            mapping_frame["shift_duration_min"] = pd.to_numeric(
                mapping_frame["shift_duration_min"], errors="coerce"
            )
            mapping_frame = mapping_frame.dropna(subset=["turno", "shift_duration_min"])
            shift_minutes_map = mapping_frame.set_index("turno")["shift_duration_min"].to_dict()

    pre_columns = ["employee_id", "state_code"]
    date_column = None
    for candidate in ("date", "data_dt", "data"):
        if candidate in preassignments_df.columns:
            date_column = candidate
            pre_columns.append(candidate)
            break
    if date_column is None:
        return leaves_df, leaves_days_df

    pre_work = preassignments_df.loc[:, pre_columns].copy()
    pre_work["employee_id"] = pre_work["employee_id"].astype(str).str.strip()
    pre_work["state_code"] = pre_work["state_code"].astype(str).str.strip().str.upper()
    pre_work["_date"] = pd.to_datetime(pre_work[date_column], errors="coerce").dt.date
    pre_work = pre_work.dropna(subset=["_date"])
    if pre_work.empty:
        day_work = day_work.drop(columns="_date")
        return leaves_df, day_work

    pre_work["duration_min"] = pre_work["state_code"].map(shift_minutes_map)

    targets = day_work.loc[unplanned_mask, ["employee_id", "_date"]].merge(
        pre_work.loc[:, ["employee_id", "_date", "state_code", "duration_min"]],
        on=["employee_id", "_date"],
        how="left",
    )
    if targets.empty:
        day_work = day_work.drop(columns="_date")
        return leaves_df, day_work

    targets["duration_min"] = pd.to_numeric(targets["duration_min"], errors="coerce")
    targets["override_hours"] = targets["duration_min"] / 60.0

    overrides = targets.dropna(subset=["override_hours"]).set_index(["employee_id", "_date"])
    if not overrides.empty:
        day_indexed = day_work.set_index(["employee_id", "_date"])
        day_indexed.loc[overrides.index, "absence_hours_h"] = overrides["override_hours"]
        day_indexed.loc[overrides.index, "is_planned"] = False
        day_work = day_indexed.reset_index()

    if leaves_df is None or leaves_df.empty:
        day_work = day_work.drop(columns="_date")
        return leaves_df, day_work

    shift_work = leaves_df.copy()
    shift_work["employee_id"] = shift_work["employee_id"].astype(str).str.strip()
    if "data_dt" in shift_work.columns:
        shift_work["_date"] = pd.to_datetime(shift_work["data_dt"], errors="coerce").dt.date
    else:
        shift_work["_date"] = pd.to_datetime(shift_work["data"], errors="coerce").dt.date
    shift_work["turno"] = shift_work["turno"].astype(str).str.strip().str.upper()
    shift_work["shift_duration_min"] = pd.to_numeric(
        shift_work.get("shift_duration_min"), errors="coerce"
    )

    override_records = targets.dropna(subset=["state_code", "duration_min"])
    if not override_records.empty:
        for _, record in override_records.iterrows():
            emp_id = record["employee_id"]
            day_value = record["_date"]
            shift_code = record["state_code"]
            duration_value = record["duration_min"]
            key_mask = (
                (shift_work["employee_id"] == emp_id)
                & (shift_work["_date"] == day_value)
            )
            if not key_mask.any():
                continue
            match_mask = key_mask & shift_work["turno"].eq(shift_code)
            if match_mask.any():
                shift_work.loc[match_mask, "shift_duration_min"] = duration_value
            shift_work.loc[key_mask & ~match_mask, "shift_duration_min"] = 0.0
            shift_work.loc[key_mask, "is_planned"] = False

    shift_work = shift_work.drop(columns="_date")
    day_work = day_work.drop(columns="_date")

    return shift_work, day_work


__all__ = ["load_leaves", "apply_unplanned_leave_durations"]
