from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .utils import LoaderError, _ensure_cols, TURNI_DOMANDA


def load_shifts(path: str) -> pd.DataFrame:
    hhmm = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(
        df, {"shift_id", "start", "end", "duration_min", "crosses_midnight"}, "shifts.csv"
    )

    for c in ["shift_id", "start", "end"]:
        df[c] = df[c].astype(str).str.strip()

    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="raise").astype(int)
    df["crosses_midnight"] = pd.to_numeric(df["crosses_midnight"], errors="raise").astype(int)

    bad_cm = sorted(set(df["crosses_midnight"].unique()) - {0, 1})
    if bad_cm:
        raise LoaderError(
            f"shifts.csv: crosses_midnight deve essere 0 o 1, trovati: {bad_cm}"
        )

    key_cols = ["shift_id", "start", "end", "duration_min", "crosses_midnight"]
    if df["shift_id"].duplicated().any():
        grp = df.groupby("shift_id")[key_cols].nunique()
        diverging = grp[(grp > 1).any(axis=1)]
        if not diverging.empty:
            raise LoaderError(
                "shifts.csv: shift_id duplicati con definizioni diverse: "
                + ", ".join(diverging.index.tolist())
            )
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

    zero_duration_shifts = {"R", "SN", "F"}
    mask_zero = df["shift_id"].isin(zero_duration_shifts)
    if not df.loc[mask_zero, "duration_min"].eq(0).all() or not df.loc[
        mask_zero, "crosses_midnight"
    ].eq(0).all():
        raise LoaderError(
            "shifts.csv: R, SN e F devono avere duration_min=0 e crosses_midnight=0"
        )

    if (df.loc[mask_zero, ["start", "end"]] != "").any().any():
        raise LoaderError("shifts.csv: R, SN e F devono avere start/end vuoti")

    mask_nzero = ~mask_zero
    if (df.loc[mask_nzero, "duration_min"] <= 0).any():
        bad = (
            df.loc[mask_nzero & (df["duration_min"] <= 0), "shift_id"].unique().tolist()
        )
        raise LoaderError(
            f"shifts.csv: turni con duration_min <= 0 non ammessi (eccetto R/SN): {bad}"
        )

    bad_start = df.loc[mask_nzero, "start"].apply(lambda s: bool(hhmm.fullmatch(s))).eq(False)
    bad_end = df.loc[mask_nzero, "end"].apply(lambda s: bool(hhmm.fullmatch(s))).eq(False)
    if bad_start.any() or bad_end.any():
        bad_rows = df.loc[mask_nzero & (bad_start | bad_end), ["shift_id", "start", "end"]]
        raise LoaderError(
            f"shifts.csv: start/end non validi (HH:MM) per turni non-zero:\n{bad_rows}"
        )

    def to_minutes(s: str) -> int:
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    for sid, s, e, cm in df.loc[
        mask_nzero, ["shift_id", "start", "end", "crosses_midnight"]
    ].itertuples(index=False):
        sm = to_minutes(s)
        em = to_minutes(e)
        if cm == 0 and not (em > sm):
            raise LoaderError(
                f"shifts.csv: per turno {sid} crosses_midnight=0 ma end <= start ({e} <= {s})"
            )
        if cm == 1 and not (em < sm):
            raise LoaderError(
                f"shifts.csv: per turno {sid} crosses_midnight=1 ma end >= start ({e} >= {s})"
            )

    def to_timedelta_or_nat(s: str):
        if not s:
            return pd.NaT
        h, m = s.split(":")
        return pd.to_timedelta(int(h), unit="h") + pd.to_timedelta(int(m), unit="m")

    df["start_time"] = df["start"].apply(to_timedelta_or_nat)
    df["end_time"] = df["end"].apply(to_timedelta_or_nat)

    return df[[
        "shift_id",
        "start",
        "end",
        "duration_min",
        "crosses_midnight",
        "start_time",
        "end_time",
    ]]


def load_shift_role_eligibility(
    path: str, employees_df: pd.DataFrame, shifts_df: pd.DataFrame
) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"shift_id", "ruolo"}, "shift_role_eligibility.csv")

    df["shift_id"] = df["shift_id"].astype(str).str.strip()
    df["ruolo"] = df["ruolo"].astype(str).str.strip()

    if (df["shift_id"] == "").any():
        bad = df.loc[df["shift_id"] == "", :].index.tolist()[:5]
        raise LoaderError(
            f"shift_role_eligibility.csv: shift_id vuoti nelle righe (prime occorrenze): {bad}"
        )
    if (df["ruolo"] == "").any():
        bad = df.loc[df["ruolo"] == "", :].index.tolist()[:5]
        raise LoaderError(
            f"shift_role_eligibility.csv: ruolo vuoto nelle righe (prime occorrenze): {bad}"
        )

    known_roles = set(employees_df["ruolo"].unique())
    bad_roles = sorted(set(df["ruolo"].unique()) - known_roles)
    if bad_roles:
        raise LoaderError(
            "shift_role_eligibility.csv: ruoli sconosciuti rispetto a employees.csv: "
            f"{bad_roles}"
        )

    known_shifts = set(shifts_df["shift_id"].unique())
    bad_shifts = sorted(set(df["shift_id"].unique()) - known_shifts)
    if bad_shifts:
        raise LoaderError(
            "shift_role_eligibility.csv: shift_id sconosciuti rispetto a shifts.csv: "
            f"{bad_shifts}"
        )

    df = df.drop_duplicates(subset=["shift_id", "ruolo"]).reset_index(drop=True)

    demand_shifts_in_catalog = sorted(TURNI_DOMANDA & known_shifts)
    for sid in demand_shifts_in_catalog:
        if df.loc[df["shift_id"] == sid].empty:
            raise LoaderError(
                "shift_role_eligibility.csv: nessun ruolo idoneo definito per turno di domanda "
                f"'{sid}'"
            )

    df = df.sort_values(["shift_id", "ruolo"]).reset_index(drop=True)
    return df[["shift_id", "ruolo"]]
