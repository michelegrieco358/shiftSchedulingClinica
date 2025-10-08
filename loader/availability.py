from __future__ import annotations

import os

import pandas as pd

from .calendar import attach_calendar
from .utils import LoaderError, _compute_horizon_window, _ensure_cols


def load_availability(
    path: str,
    employees_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
) -> pd.DataFrame:
    allowed_turns = tuple(
        pd.Series(shifts_df.loc[shifts_df["duration_min"] > 0, "shift_id"])
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if not allowed_turns:
        raise LoaderError(
            "shifts.csv: nessun turno con duration_min>0 (niente turni lavorativi disponibili)."
        )

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
    _ensure_cols(df, {"data", "employee_id"}, "availability.csv")

    df["data"] = df["data"].astype(str).str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"availability.csv: formato data non valido: {exc}")

    if "turno" not in df.columns:
        df["turno"] = ""
    df["turno"] = df["turno"].astype(str).str.strip().str.upper()

    known_emp = set(employees_df["employee_id"].unique())
    unknown = sorted(set(df["employee_id"].unique()) - known_emp)
    if unknown:
        raise LoaderError(f"availability.csv: employee_id sconosciuti: {unknown}")

    allday_mask = (df["turno"] == "") | (df["turno"] == "ALL") | (df["turno"] == "*")
    perturno_mask = ~allday_mask

    bad_turni = sorted(set(df.loc[perturno_mask, "turno"].unique()) - set(allowed_turns))
    if bad_turni:
        raise LoaderError(
            "availability.csv: turni non ammessi: "
            f"{bad_turni}. Ammessi (da shifts.csv con duration_min>0): {sorted(allowed_turns)} "
            "oppure ALL/*/vuoto per indisponibilità tutto il giorno."
        )

    rows = []
    for _, r in df.iterrows():
        base_row = {"data": r["data"], "data_dt": r["data_dt"], "employee_id": r["employee_id"]}
        turno = r["turno"]
        if turno in allowed_turns:
            rows.append({**base_row, "turno": turno})
        else:
            for tt in allowed_turns:
                rows.append({**base_row, "turno": tt})

    out = pd.DataFrame(rows).drop_duplicates(subset=["data", "employee_id", "turno"]).reset_index(drop=True)

    out = attach_calendar(out, calendar_df)

    shift_cols = ["shift_id", "start_time", "end_time", "duration_min", "crosses_midnight"]
    shift_info = shifts_df[shift_cols].rename(
        columns={
            "shift_id": "turno",
            "start_time": "shift_start_time",
            "end_time": "shift_end_time",
            "duration_min": "shift_duration_min",
            "crosses_midnight": "shift_crosses_midnight",
        }
    )
    out = out.merge(shift_info, on="turno", how="left", validate="many_to_one")

    out["shift_start_dt"] = out["data_dt"] + out["shift_start_time"]
    out["shift_end_dt"] = out["data_dt"] + out["shift_end_time"]
    crosses_mask = out["shift_crosses_midnight"].fillna(0).astype(int) == 1
    out.loc[crosses_mask, "shift_end_dt"] = (
        out.loc[crosses_mask, "shift_end_dt"] + pd.Timedelta(days=1)
    )

    horizon_start, horizon_end = _compute_horizon_window(calendar_df)
    overlaps = out["shift_start_dt"].notna() & out["shift_end_dt"].notna()
    overlaps &= out["shift_end_dt"] > horizon_start
    overlaps &= out["shift_start_dt"] < horizon_end

    in_horizon = out["is_in_horizon"].astype("boolean", copy=False).fillna(False)
    keep_mask = in_horizon.astype(bool) | overlaps
    out = out.loc[keep_mask].copy()

    return out[[
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
