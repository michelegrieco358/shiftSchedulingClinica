from __future__ import annotations

import os

import pandas as pd

from .calendar import attach_calendar
from .utils import LoaderError, _ensure_cols


def _resolve_column(df: pd.DataFrame, *candidates: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise LoaderError(
        "preassignments.csv: colonne mancanti. "
        f"Richieste: uno fra {candidates}."
    )


def load_preassignments(
    path: str,
    employees_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Load previously generated assignments for the new horizon."""

    base_columns = [
        "employee_id",
        "data",
        "data_dt",
        "date",
        "state_code",
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
        return pd.DataFrame(columns=base_columns)

    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(df, {"employee_id"}, "preassignments.csv")

    date_col = _resolve_column(df, "data", "date", "day")
    state_col = _resolve_column(df, "state_code", "state", "turno", "shift_code")

    work = df.loc[:, ["employee_id", date_col, state_col]].copy()
    work["employee_id"] = work["employee_id"].astype(str).str.strip()

    work[date_col] = work[date_col].astype(str).str.strip()
    work["data_dt"] = pd.to_datetime(work[date_col], errors="coerce")
    if work["data_dt"].isna().any():
        bad_rows = work.loc[work["data_dt"].isna(), ["employee_id", date_col]].head()
        raise LoaderError(
            "preassignments.csv: formato data non valido (atteso YYYY-MM-DD) per:\n"
            f"{bad_rows}"
        )

    work["data"] = work["data_dt"].dt.date.apply(lambda d: d.isoformat())

    work["state_code"] = (
        work[state_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    work = work.loc[work["state_code"].ne("")].copy()

    known_emp = set(employees_df["employee_id"].astype(str).str.strip())
    unknown = sorted(set(work["employee_id"]) - known_emp)
    if unknown:
        raise LoaderError(
            "preassignments.csv: employee_id sconosciuti rispetto a employees.csv: "
            f"{unknown}"
        )

    duplicates = work.duplicated(subset=["employee_id", "data"], keep=False)
    if duplicates.any():
        sample = work.loc[
            duplicates, ["employee_id", "data", "state_code"]
        ].head()
        raise LoaderError(
            "preassignments.csv: più di uno stato indicato per lo stesso "
            "dipendente nella stessa data. Esempio:\n"
            f"{sample}"
        )

    if work.empty:
        return pd.DataFrame(columns=base_columns)

    work = attach_calendar(work, calendar_df)

    work["data_dt"] = pd.to_datetime(work["data"], errors="coerce")
    work = work.loc[work["data_dt"].notna()].copy()
    work["date"] = work["data_dt"].dt.date

    horizon_mask = work["is_in_horizon"].astype("boolean", copy=False).fillna(False)
    work = work.loc[horizon_mask].copy()

    if work.empty:
        return pd.DataFrame(columns=base_columns)

    ordered = work[[
        "employee_id",
        "data",
        "data_dt",
        "date",
        "state_code",
        "dow_iso",
        "week_start_date",
        "week_start_date_dt",
        "week_id",
        "week_idx",
        "is_in_horizon",
        "is_weekend",
        "is_weekday_holiday",
        "holiday_desc",
    ]].reset_index(drop=True)

    return ordered


__all__ = ["load_preassignments"]
