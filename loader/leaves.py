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
    abs_by_day = explode_absences_by_day(
        absences_df.loc[:, ["employee_id", "date_from", "date_to", "type"]],
        min_date=min(month_start_date, horizon_start_date),
        max_date=horizon_end_date,
        absence_hours_h=explode_fallback,
    )

    if not abs_by_day.empty:
        abs_by_day["employee_id"] = abs_by_day["employee_id"].astype(str).str.strip()
        abs_by_day["absence_hours_h"] = abs_by_day["employee_id"].map(hours_series)

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
        day_out = day_out[
            [
                "employee_id",
                "data",
                "data_dt",
                "tipo_set",
                "is_leave_day",
                "is_absent",
                "absence_hours_h",
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
