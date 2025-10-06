from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from .calendar import attach_calendar
from .utils import LoaderError, _ensure_cols


def load_history(
    path: str,
    employees_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "data",
                "data_dt",
                "employee_id",
                "turno",
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
        )

    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(df, {"data", "employee_id", "turno"}, "history.csv")

    df["data"] = df["data"].astype(str).str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["turno"] = df["turno"].astype(str).str.strip()
    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"history.csv: formato data non valido: {exc}")

    def _is_iso_date(s: str) -> bool:
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return True
        except Exception:
            return False

    bad_date_mask = ~df["data"].apply(_is_iso_date)
    if bad_date_mask.any():
        bad_rows = df.loc[bad_date_mask, ["data", "employee_id", "turno"]].head()
        raise LoaderError(
            "history.csv: formato data non valido (atteso YYYY-MM-DD) per le righe:\n"
            f"{bad_rows}"
        )

    known_shifts = set(shifts_df["shift_id"].astype(str).str.strip().unique())
    bad_turni = sorted(set(df["turno"].unique()) - known_shifts)
    if bad_turni:
        raise LoaderError(f"history.csv: turni non presenti in shifts.csv: {bad_turni}")

    known_emp = set(employees_df["employee_id"].unique())
    unknown = sorted(set(df["employee_id"].unique()) - known_emp)
    if unknown:
        raise LoaderError(
            f"history.csv: employee_id sconosciuti rispetto a employees.csv: {unknown}"
        )

    df = df.drop_duplicates(subset=["data", "employee_id", "turno"]).reset_index(drop=True)

    conflicts = (
        df.groupby(["data", "employee_id"])["turno"].nunique().reset_index(name="n_turni")
    )
    conflicts = conflicts[conflicts["n_turni"] > 1]
    if not conflicts.empty:
        keys = set(map(tuple, conflicts[["data", "employee_id"]].to_records(index=False)))
        sample = (
            df[df.set_index(["data", "employee_id"]).index.isin(keys)]
            .sort_values(["data", "employee_id", "turno"])
            .head(20)
        )
        raise LoaderError(
            "history.csv: più di un turno per lo stesso dipendente nello stesso giorno (max 1 turno/giorno). "
            "Esempi delle righe in conflitto:\n"
            f"{sample[['data','employee_id','turno']]}"
        )

    df = attach_calendar(df, calendar_df)

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
    df = df.merge(shift_info, on="turno", how="left", validate="many_to_one")

    df["shift_start_dt"] = df["data_dt"] + df["shift_start_time"]
    df["shift_end_dt"] = df["data_dt"] + df["shift_end_time"]
    crosses_mask = df["shift_crosses_midnight"].fillna(0).astype(int) == 1
    df.loc[crosses_mask, "shift_end_dt"] = df.loc[crosses_mask, "shift_end_dt"] + pd.Timedelta(days=1)

    return df[[
        "data",
        "data_dt",
        "employee_id",
        "turno",
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
    ]]
