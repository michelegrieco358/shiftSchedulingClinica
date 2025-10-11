"""Pre-processing utilities to build candidate assignment tables."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _normalise(series: pd.Series) -> pd.Series:
    """Return an upper-cased, stripped string representation of ``series``."""

    return series.astype(str).str.strip().str.upper()


def _validate_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name}: colonne mancanti: {', '.join(sorted(missing))}")


def build_candidate_assignments(
    employees: pd.DataFrame,
    employee_allowed_reparti: pd.DataFrame,
    shift_slots: pd.DataFrame,
    shift_role_eligibility: pd.DataFrame,
    absences_by_day: pd.DataFrame | None = None,
    must_locks: pd.DataFrame | None = None,
    forbid_locks: pd.DataFrame | None = None,
    in_scope_only: bool = True,
) -> pd.DataFrame:
    """Build the candidate assignment table for the solver pre-processing stage.

    The output contains one row for each assignable pair ``(employee_id, slot_id)``
    after applying department compatibility, role eligibility, absences and
    optional lock constraints.
    """

    _validate_columns(
        employees, ["employee_id", "reparto_id", "role"], "employees"
    )
    _validate_columns(
        employee_allowed_reparti,
        ["employee_id", "reparto_id_allowed"],
        "employee_allowed_reparti",
    )
    _validate_columns(
        shift_slots,
        ["slot_id", "reparto_id", "shift_code", "start_dt", "is_night"],
        "shift_slots",
    )
    _validate_columns(
        shift_role_eligibility,
        ["shift_code", "role", "allowed"],
        "shift_role_eligibility",
    )

    employees_df = employees.copy()
    employees_df["_employee_key"] = _normalise(employees_df["employee_id"])
    employees_df["_reparto_home_key"] = _normalise(employees_df["reparto_id"])
    employees_df["_role_key"] = _normalise(employees_df["role"])

    allowed_df = employee_allowed_reparti.copy()
    allowed_df["_employee_key"] = _normalise(allowed_df["employee_id"])
    allowed_df["_reparto_allowed_key"] = _normalise(
        allowed_df["reparto_id_allowed"]
    )

    slots_df = shift_slots.copy()
    if in_scope_only and "in_scope" in slots_df.columns:
        slots_df = slots_df.loc[slots_df["in_scope"].fillna(False)].copy()

    slots_df = slots_df.rename(columns={"reparto_id": "slot_reparto_id"})
    slots_df["_slot_reparto_key"] = _normalise(slots_df["slot_reparto_id"])
    slots_df["_shift_code_key"] = _normalise(slots_df["shift_code"])
    slots_df["_slot_id_key"] = _normalise(slots_df["slot_id"])
    slots_df["start_dt"] = pd.to_datetime(slots_df["start_dt"], errors="coerce")
    if slots_df["start_dt"].isna().any():
        raise ValueError("shift_slots: valori non validi in start_dt")
    slots_df["date"] = slots_df["start_dt"].dt.date

    role_elig_df = shift_role_eligibility.copy()
    role_elig_df = role_elig_df.loc[
        role_elig_df["allowed"].fillna(False).astype(bool)
    ].copy()
    role_elig_df["_role_key"] = _normalise(role_elig_df["role"])
    role_elig_df["_shift_code_key"] = _normalise(role_elig_df["shift_code"])
    role_elig_df = role_elig_df.drop_duplicates(subset=["_role_key", "_shift_code_key"])

    employees_allowed = employees_df.merge(
        allowed_df[["_employee_key", "_reparto_allowed_key", "reparto_id_allowed"]],
        on="_employee_key",
        how="inner",
    )

    candidate = employees_allowed.merge(
        slots_df,
        left_on="_reparto_allowed_key",
        right_on="_slot_reparto_key",
        how="inner",
    )

    candidate = candidate.merge(
        role_elig_df[["_role_key", "_shift_code_key"]],
        on=["_role_key", "_shift_code_key"],
        how="inner",
    )

    candidate["_employee_slot_key"] = (
        candidate["_employee_key"] + "||" + candidate["_slot_id_key"]
    )

    if absences_by_day is not None and not absences_by_day.empty:
        _validate_columns(
            absences_by_day, ["employee_id", "date", "is_absent"], "absences_by_day"
        )
        abs_df = absences_by_day.copy()
        abs_df["_employee_key"] = _normalise(abs_df["employee_id"])
        abs_df["date"] = pd.to_datetime(abs_df["date"], errors="coerce").dt.date
        if abs_df["date"].isna().any():
            raise ValueError("absences_by_day: valori non validi in date")
        abs_df = abs_df.loc[abs_df["is_absent"].fillna(False).astype(bool)]
        if not abs_df.empty:
            abs_pairs = abs_df[["_employee_key", "date"]].drop_duplicates()
            candidate = candidate.merge(
                abs_pairs,
                on=["_employee_key", "date"],
                how="left",
                indicator=True,
            )
            candidate = candidate.loc[candidate["_merge"] == "left_only"].drop(
                columns="_merge"
            )

    must_df = None
    forbid_df = None
    if must_locks is not None and not must_locks.empty:
        _validate_columns(must_locks, ["employee_id", "slot_id"], "must_locks")
        must_df = must_locks.copy()
        must_df["_employee_key"] = _normalise(must_df["employee_id"])
        must_df["_slot_id_key"] = _normalise(must_df["slot_id"])
        must_df = must_df.drop_duplicates(subset=["_employee_key", "_slot_id_key"])
    if forbid_locks is not None and not forbid_locks.empty:
        _validate_columns(forbid_locks, ["employee_id", "slot_id"], "forbid_locks")
        forbid_df = forbid_locks.copy()
        forbid_df["_employee_key"] = _normalise(forbid_df["employee_id"])
        forbid_df["_slot_id_key"] = _normalise(forbid_df["slot_id"])
        forbid_df = forbid_df.drop_duplicates(subset=["_employee_key", "_slot_id_key"])

    if must_df is not None and forbid_df is not None:
        conflicts = must_df.merge(
            forbid_df, on=["_employee_key", "_slot_id_key"], how="inner"
        )
        if not conflicts.empty:
            conflict_pairs = [
                f"(employee_id={row['employee_id_x']}, slot_id={row['slot_id_x']})"
                for _, row in conflicts.iterrows()
            ]
            raise ValueError(
                "locks: conflitto tra must e forbid per: "
                + ", ".join(sorted(conflict_pairs))
            )

    if forbid_df is not None and not forbid_df.empty:
        forbid_keys = set(
            forbid_df["_employee_key"] + "||" + forbid_df["_slot_id_key"]
        )
        candidate = candidate.loc[
            ~candidate["_employee_slot_key"].isin(forbid_keys)
        ].copy()

    candidate["must"] = 0
    if must_df is not None and not must_df.empty:
        must_keys = set(must_df["_employee_key"] + "||" + must_df["_slot_id_key"])
        candidate.loc[
            candidate["_employee_slot_key"].isin(must_keys), "must"
        ] = 1

    candidate["reparto_home"] = candidate["reparto_id"]
    candidate["is_cross"] = (
        candidate["_slot_reparto_key"] != candidate["_reparto_home_key"]
    )

    final_cols = [
        "employee_id",
        "slot_id",
        "reparto_home",
        "slot_reparto_id",
        "is_cross",
        "role",
        "shift_code",
        "date",
        "is_night",
        "must",
    ]

    candidate = candidate.sort_values(
        ["employee_id", "slot_id", "must"], ascending=[True, True, False]
    )
    candidate = candidate.drop_duplicates(subset=["employee_id", "slot_id"], keep="first")
    candidate = candidate.sort_values(["employee_id", "date", "slot_id"]).reset_index(
        drop=True
    )

    helper_cols = [col for col in candidate.columns if col.startswith("_")]
    if helper_cols:
        candidate = candidate.drop(columns=helper_cols)

    missing_final = [col for col in final_cols if col not in candidate.columns]
    for missing in missing_final:
        candidate[missing] = pd.Series(dtype="object")

    candidate = candidate.loc[:, final_cols]
    candidate["must"] = candidate["must"].fillna(0).astype(int)
    candidate["is_cross"] = candidate["is_cross"].fillna(False).astype(bool)

    return candidate

