from __future__ import annotations

import os

import pandas as pd

from .calendar import attach_calendar
from .utils import LoaderError, _ensure_cols


def load_leaves(
    path: str,
    employees_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
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

    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(df, {"employee_id", "start_date", "end_date", "tipo"}, "leaves.csv")

    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["start_date"] = df["start_date"].astype(str).str.strip()
    df["end_date"] = df["end_date"].astype(str).str.strip()
    df["tipo"] = df["tipo"].astype(str).str.strip()

    known_emp = set(employees_df["employee_id"].unique())
    unknown = sorted(set(df["employee_id"].unique()) - known_emp)
    if unknown:
        raise LoaderError(f"leaves.csv: employee_id sconosciuti: {unknown}")

    try:
        df["start_date_dt"] = pd.to_datetime(df["start_date"], format="%Y-%m-%d", errors="raise")
        df["end_date_dt"] = pd.to_datetime(df["end_date"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"leaves.csv: formato data non valido: {exc}")

    bad_interval = df["start_date_dt"] > df["end_date_dt"]
    if bad_interval.any():
        bad_rows = df.loc[bad_interval, ["employee_id", "start_date", "end_date"]]
        raise LoaderError(
            "leaves.csv: intervallo con start_date > end_date per le righe:\n"
            f"{bad_rows}"
        )

    shift_info = shifts_df.loc[
        shifts_df["shift_id"].isin(allowed_turns),
        ["shift_id", "start_time", "end_time", "crosses_midnight"],
    ].copy()
    shift_rows = list(shift_info.itertuples(index=False))

    records = []
    day_records = []
    for row in df.itertuples(index=False):
        absence_start_day = row.start_date_dt.normalize()
        absence_end_day = row.end_date_dt.normalize()
        absence_interval_start = absence_start_day
        absence_interval_end = absence_end_day + pd.Timedelta(days=1)

        current_day = absence_start_day
        while current_day <= absence_end_day:
            day_records.append(
                {
                    "employee_id": row.employee_id,
                    "data": current_day.date().isoformat(),
                    "tipo": row.tipo,
                }
            )
            current_day += pd.Timedelta(days=1)

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
        shift_out = shift_out[shift_out["is_in_horizon"].fillna(False)].copy()

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

    if day_records:
        day_out = pd.DataFrame.from_records(day_records)
        day_out = day_out.drop_duplicates().reset_index(drop=True)
        day_out["data_dt"] = pd.to_datetime(day_out["data"], format="%Y-%m-%d")
        day_out = attach_calendar(day_out, calendar_df)
        day_out = day_out[day_out["is_in_horizon"].fillna(False)].copy()

        def _join_types(values: pd.Series) -> str:
            unique_vals = sorted({str(v).strip() for v in values if str(v).strip()})
            return "|".join(unique_vals)

        day_out = (
            day_out.groupby(["employee_id", "data"], as_index=False)
            .agg(
                data_dt=("data_dt", "first"),
                tipo_set=("tipo", _join_types),
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
        day_out = day_out[
            [
                "employee_id",
                "data",
                "data_dt",
                "tipo_set",
                "is_leave_day",
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
