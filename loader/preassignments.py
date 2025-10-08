from __future__ import annotations

import os
import warnings

import pandas as pd


_REQUIRED_COLUMNS = {"employee_id", "slot_id", "lock"}


def load_preassignments(path: str) -> pd.DataFrame:
    """Load and validate preassignment locks from ``preassignments.csv``."""

    if not os.path.exists(path):
        return pd.DataFrame(columns=["employee_id", "slot_id", "lock", "note"])

    df = pd.read_csv(path, dtype=str).fillna("")

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "preassignments.csv: colonne mancanti: " + str(sorted(missing))
        )

    columns_order = [
        "employee_id",
        "slot_id",
        "lock",
        *[col for col in df.columns if col not in _REQUIRED_COLUMNS],
    ]
    work_df = df.loc[:, columns_order].copy()

    work_df["employee_id"] = work_df["employee_id"].astype(str).str.strip()
    if work_df["employee_id"].eq("").any():
        bad_idx = work_df.index[work_df["employee_id"].eq("")].tolist()[:5]
        raise ValueError(
            "preassignments.csv: employee_id non può essere vuoto. "
            f"Prime occorrenze: {bad_idx}"
        )

    slot_series = work_df["slot_id"].astype(str).str.strip()
    if slot_series.eq("").any():
        bad_idx = work_df.index[slot_series.eq("")].tolist()[:5]
        raise ValueError(
            "preassignments.csv: slot_id non può essere vuoto. "
            f"Prime occorrenze: {bad_idx}"
        )
    try:
        work_df["slot_id"] = pd.to_numeric(slot_series, errors="raise").astype("int64")
    except ValueError as exc:
        raise ValueError(
            "preassignments.csv: slot_id deve essere numerico intero"
        ) from exc

    lock_series = work_df["lock"].astype(str).str.strip()
    if lock_series.eq("").any():
        bad_idx = work_df.index[lock_series.eq("")].tolist()[:5]
        raise ValueError(
            "preassignments.csv: lock non può essere vuoto. "
            f"Prime occorrenze: {bad_idx}"
        )
    try:
        work_df["lock"] = pd.to_numeric(lock_series, errors="raise").astype(int)
    except ValueError as exc:
        raise ValueError("preassignments.csv: lock deve essere numerico") from exc

    bad_lock = sorted(set(work_df["lock"].unique()) - {1, -1})
    if bad_lock:
        raise ValueError(
            "preassignments.csv: lock deve valere 1 (assegna) o -1 (veto). "
            f"Valori non validi: {bad_lock}"
        )

    if "note" in work_df.columns:
        work_df["note"] = work_df["note"].astype(str).str.strip()

    work_df = work_df.drop_duplicates().reset_index(drop=True)

    conflicts = (
        work_df.groupby(["employee_id", "slot_id"])["lock"].nunique().reset_index()
    )
    conflicts = conflicts[conflicts["lock"] > 1]
    if not conflicts.empty:
        keys = [
            {"employee_id": row.employee_id, "slot_id": int(row.slot_id)}
            for row in conflicts.itertuples(index=False)
        ]
        raise ValueError(
            "preassignments.csv: lock contrastanti per le chiavi specificate: "
            f"{keys}"
        )

    work_df = work_df.drop_duplicates(subset=["employee_id", "slot_id", "lock"])

    return work_df.reset_index(drop=True)


def validate_preassignments(
    pre_df: pd.DataFrame,
    employees: pd.DataFrame,
    shift_slots: pd.DataFrame,
    absences_by_day: pd.DataFrame | None = None,
    shift_role_eligibility: pd.DataFrame | None = None,
    cross_reparto_enabled: bool = False,
) -> pd.DataFrame:
    """Cross-check preassignments against employees, slots, roles and absences."""

    if pre_df.empty:
        result_columns = list(pre_df.columns)
        for col in ("date", "reparto_id", "shift_code"):
            if col not in result_columns:
                result_columns.append(col)
        return pd.DataFrame(columns=result_columns)

    clean = pre_df.copy()

    clean["employee_id"] = clean["employee_id"].astype(str).str.strip()
    clean["slot_id"] = clean["slot_id"].astype("int64")

    employees_lookup = employees.loc[:, ["employee_id", "reparto_id", "ruolo"]].copy()
    employees_lookup["employee_id"] = (
        employees_lookup["employee_id"].astype(str).str.strip()
    )
    employees_lookup["reparto_id"] = (
        employees_lookup["reparto_id"].astype(str).str.strip()
    )
    employees_lookup["ruolo"] = employees_lookup["ruolo"].astype(str).str.strip()
    employees_lookup = employees_lookup.rename(
        columns={"reparto_id": "employee_reparto_id", "ruolo": "employee_ruolo"}
    )

    slots_lookup = shift_slots.loc[
        :, ["slot_id", "reparto_id", "shift_code", "start_dt"]
    ].copy()
    slots_lookup["slot_id"] = slots_lookup["slot_id"].astype("int64")
    slots_lookup["reparto_id"] = slots_lookup["reparto_id"].astype(str).str.strip()
    slots_lookup["shift_code"] = slots_lookup["shift_code"].astype(str).str.strip()
    slots_lookup = slots_lookup.rename(columns={"reparto_id": "slot_reparto_id"})

    merged = clean.merge(
        employees_lookup,
        on="employee_id",
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        slots_lookup,
        on="slot_id",
        how="left",
        validate="many_to_one",
    )

    missing_emp = merged["employee_ruolo"].isna()
    if missing_emp.any():
        bad = merged.loc[missing_emp, ["employee_id", "slot_id"]].drop_duplicates()
        keys = [
            {"employee_id": row.employee_id, "slot_id": int(row.slot_id)}
            for row in bad.itertuples(index=False)
        ]
        raise ValueError(
            "preassignments: employee_id inesistente rispetto a employees.csv: "
            f"{keys}"
        )

    missing_slot = merged["shift_code"].isna()
    if missing_slot.any():
        bad = merged.loc[missing_slot, ["employee_id", "slot_id"]].drop_duplicates()
        keys = [
            {"employee_id": row.employee_id, "slot_id": int(row.slot_id)}
            for row in bad.itertuples(index=False)
        ]
        raise ValueError(
            "preassignments: slot_id inesistente rispetto a shift_slots: " + str(keys)
        )

    reparto_mismatch = merged["employee_reparto_id"] != merged["slot_reparto_id"]
    if not cross_reparto_enabled:
        if reparto_mismatch.any():
            bad = merged.loc[
                reparto_mismatch,
                ["employee_id", "slot_id", "employee_reparto_id", "slot_reparto_id"],
            ].drop_duplicates()
            keys = [
                {
                    "employee_id": row.employee_id,
                    "slot_id": int(row.slot_id),
                    "employee_reparto_id": row.employee_reparto_id,
                    "slot_reparto_id": row.slot_reparto_id,
                }
                for row in bad.itertuples(index=False)
            ]
            raise ValueError(
                "preassignments: reparto diverso tra dipendente e slot (cross-reparto disabilitato): "
                f"{keys}"
            )
    elif reparto_mismatch.any():
        bad = merged.loc[
            reparto_mismatch,
            ["employee_id", "slot_id", "employee_reparto_id", "slot_reparto_id"],
        ].drop_duplicates()
        warnings.warn(
            "preassignments: reparto diverso tra dipendente e slot con cross-reparto abilitato: "
            f"{bad.to_dict(orient='records')}",
            UserWarning,
            stacklevel=2,
        )

    if shift_role_eligibility is not None and not shift_role_eligibility.empty:
        elig = shift_role_eligibility.loc[:, ["shift_id", "ruolo"]].copy()
        elig["shift_id"] = elig["shift_id"].astype(str).str.strip()
        elig["ruolo"] = elig["ruolo"].astype(str).str.strip()
        elig = elig.drop_duplicates().rename(
            columns={"shift_id": "shift_code", "ruolo": "employee_ruolo"}
        )

        merged = merged.merge(
            elig,
            on=["shift_code", "employee_ruolo"],
            how="left",
            indicator="_elig_merge",
            validate="many_to_many",
        )

        invalid_role = merged["_elig_merge"] == "left_only"
        if invalid_role.any():
            bad = merged.loc[
                invalid_role,
                ["employee_id", "slot_id", "shift_code", "employee_ruolo"],
            ].drop_duplicates()
            keys = [
                {
                    "employee_id": row.employee_id,
                    "slot_id": int(row.slot_id),
                    "shift_code": row.shift_code,
                    "ruolo": row.employee_ruolo,
                }
                for row in bad.itertuples(index=False)
            ]
            raise ValueError(
                "preassignments: ruolo del dipendente non idoneo per il turno dello slot: "
                f"{keys}"
            )

    start_ts = pd.to_datetime(merged["start_dt"], errors="coerce")
    if start_ts.isna().any():
        bad = merged.loc[start_ts.isna(), ["employee_id", "slot_id"]]
        keys = [
            {"employee_id": row.employee_id, "slot_id": int(row.slot_id)}
            for row in bad.drop_duplicates().itertuples(index=False)
        ]
        raise ValueError(
            "preassignments: impossibile determinare la data dello slot per le chiavi: "
            f"{keys}"
        )
    tz = getattr(start_ts.dt, "tz", None)
    if tz is not None:
        start_ts = start_ts.dt.tz_convert(None)
    merged["date"] = start_ts.dt.date

    if absences_by_day is not None and not absences_by_day.empty:
        abs_df = absences_by_day.loc[:, ["employee_id", "date"]].copy()
        abs_df["employee_id"] = abs_df["employee_id"].astype(str).str.strip()
        abs_df["date"] = pd.to_datetime(abs_df["date"]).dt.date
        abs_df = abs_df.drop_duplicates().assign(is_absent=True)

        merged = merged.merge(
            abs_df,
            on=["employee_id", "date"],
            how="left",
            validate="many_to_many",
        )
        conflict = merged["lock"].eq(1) & merged["is_absent"].fillna(False)
        if conflict.any():
            bad = merged.loc[
                conflict,
                ["employee_id", "slot_id", "date", "shift_code"],
            ].drop_duplicates()
            keys = [
                {
                    "employee_id": row.employee_id,
                    "slot_id": int(row.slot_id),
                    "date": row.date,
                    "shift_code": row.shift_code,
                }
                for row in bad.itertuples(index=False)
            ]
            raise ValueError(
                "preassignments: lock=1 su giorni di assenza del dipendente: "
                f"{keys}"
            )
        merged = merged.drop(columns=["is_absent"])

    if "_elig_merge" in merged.columns:
        merged = merged.drop(columns=["_elig_merge"])

    merged = merged.rename(columns={"slot_reparto_id": "reparto_id"})
    merged = merged.drop(columns=["employee_reparto_id", "employee_ruolo", "start_dt"])

    result_columns = list(clean.columns)
    for col in ("date", "reparto_id", "shift_code"):
        if col not in result_columns:
            result_columns.append(col)

    return merged.loc[:, result_columns].reset_index(drop=True)


def split_preassignments(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validated preassignments into must-assign and forbid DataFrames."""

    must_df = (
        clean_df.loc[clean_df["lock"] == 1, ["employee_id", "slot_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    forbid_df = (
        clean_df.loc[clean_df["lock"] == -1, ["employee_id", "slot_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return must_df, forbid_df
