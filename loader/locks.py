from __future__ import annotations

import os
import warnings

import pandas as pd


_REQUIRED_COLUMNS = {"employee_id", "slot_id", "lock"}
_LOCK_TYPE_MAP = {"MUST_DO": 1, "FORBIDDEN": -1}


def load_locks(path: str, shift_slots: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load and normalize hard assignment locks from ``locks.csv``."""

    if not os.path.exists(path):
        warnings.warn(
            "locks.csv non trovato: caricati 0 record da locks.csv",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["employee_id", "slot_id", "lock", "note"])

    raw_df = pd.read_csv(path, dtype=str).fillna("")
    records: list[pd.DataFrame] = []

    has_direct_slot = _REQUIRED_COLUMNS.issubset(raw_df.columns)
    has_symbolic = {"date", "reparto_id", "shift_code", "employee_id", "lock_type"}.issubset(
        raw_df.columns
    )

    if not has_direct_slot and not has_symbolic:
        missing_direct = _REQUIRED_COLUMNS - set(raw_df.columns)
        missing_symbolic = {
            "date",
            "reparto_id",
            "shift_code",
            "employee_id",
            "lock_type",
        } - set(raw_df.columns)
        raise ValueError(
            "locks.csv: formato non valido. "
            "Richieste le colonne {employee_id, slot_id, lock} oppure "
            "{date, reparto_id, shift_code, employee_id, lock_type}. "
            f"Colonne mancanti: direct={sorted(missing_direct)} symbolic={sorted(missing_symbolic)}"
        )

    if has_direct_slot:
        direct_columns = [
            "employee_id",
            "slot_id",
            "lock",
            *[col for col in raw_df.columns if col not in _REQUIRED_COLUMNS],
        ]
        direct_df = raw_df.loc[:, direct_columns].copy()

        direct_df["employee_id"] = direct_df["employee_id"].astype(str).str.strip()
        if direct_df["employee_id"].eq("").any():
            bad_idx = direct_df.index[direct_df["employee_id"].eq("")].tolist()[:5]
            raise ValueError(
                "locks.csv: employee_id non può essere vuoto (sezione slot_id). "
                f"Prime occorrenze: {bad_idx}"
            )

        slot_series = direct_df["slot_id"].astype(str).str.strip()
        if slot_series.eq("").any():
            bad_idx = direct_df.index[slot_series.eq("")].tolist()[:5]
            raise ValueError(
                "locks.csv: slot_id non può essere vuoto (sezione slot_id). "
                f"Prime occorrenze: {bad_idx}"
            )
        try:
            direct_df["slot_id"] = pd.to_numeric(slot_series, errors="raise").astype(
                "int64"
            )
        except ValueError as exc:
            raise ValueError(
                "locks.csv: slot_id deve essere numerico intero (sezione slot_id)."
            ) from exc

        lock_series = direct_df["lock"].astype(str).str.strip()
        if lock_series.eq("").any():
            bad_idx = direct_df.index[lock_series.eq("")].tolist()[:5]
            raise ValueError(
                "locks.csv: lock non può essere vuoto (sezione slot_id). "
                f"Prime occorrenze: {bad_idx}"
            )
        try:
            direct_df["lock"] = pd.to_numeric(lock_series, errors="raise").astype(int)
        except ValueError as exc:
            raise ValueError(
                "locks.csv: lock deve essere numerico (sezione slot_id)."
            ) from exc

        bad_lock = sorted(set(direct_df["lock"].unique()) - {1, -1})
        if bad_lock:
            raise ValueError(
                "locks.csv: lock deve valere 1 (assegna) o -1 (veto). "
                f"Valori non validi: {bad_lock}"
            )

        if "note" in direct_df.columns:
            direct_df["note"] = direct_df["note"].astype(str).str.strip()

        records.append(direct_df.reset_index(drop=True))

    if has_symbolic:
        if shift_slots is None or shift_slots.empty:
            raise ValueError(
                "locks.csv: impossibile risolvere gli slot senza shift_slots caricati"
            )

        locks_df = raw_df.loc[
            :,
            ["date", "reparto_id", "shift_code", "employee_id", "lock_type", "note"]
            if "note" in raw_df.columns
            else ["date", "reparto_id", "shift_code", "employee_id", "lock_type"],
        ].copy()

        locks_df["employee_id"] = locks_df["employee_id"].astype(str).str.strip()
        if locks_df["employee_id"].eq("").any():
            bad_idx = locks_df.index[locks_df["employee_id"].eq("")].tolist()[:5]
            raise ValueError(
                "locks.csv: employee_id non può essere vuoto (sezione simbolica). "
                f"Prime occorrenze: {bad_idx}"
            )

        date_series = pd.to_datetime(locks_df["date"], errors="coerce")
        if date_series.isna().any():
            bad_idx = locks_df.index[date_series.isna()].tolist()[:5]
            raise ValueError(
                "locks.csv: date non valide (sezione simbolica) per le righe: "
                + str(bad_idx)
            )
        locks_df["date"] = date_series.dt.date

        locks_df["reparto_id"] = locks_df["reparto_id"].astype(str).str.strip()
        if locks_df["reparto_id"].eq("").any():
            bad_idx = locks_df.index[locks_df["reparto_id"].eq("")].tolist()[:5]
            raise ValueError(
                "locks.csv: reparto_id non può essere vuoto (sezione simbolica). "
                f"Prime occorrenze: {bad_idx}"
            )

        locks_df["shift_code"] = locks_df["shift_code"].astype(str).str.strip()
        if locks_df["shift_code"].eq("").any():
            bad_idx = locks_df.index[locks_df["shift_code"].eq("")].tolist()[:5]
            raise ValueError(
                "locks.csv: shift_code non può essere vuoto (sezione simbolica). "
                f"Prime occorrenze: {bad_idx}"
            )

        lock_map = locks_df["lock_type"].astype(str).str.strip()
        invalid = sorted(set(lock_map.unique()) - set(_LOCK_TYPE_MAP))
        if invalid:
            raise ValueError(
                "locks.csv: lock_type deve valere 'MUST_DO' o 'FORBIDDEN'. "
                f"Valori non validi: {invalid}"
            )
        locks_df["lock"] = lock_map.map(_LOCK_TYPE_MAP).astype(int)

        slots_lookup = shift_slots.loc[
            :, ["slot_id", "reparto_id", "shift_code", "start_dt"]
        ].copy()
        slots_lookup["slot_id"] = slots_lookup["slot_id"].astype("int64")
        slots_lookup["reparto_id"] = slots_lookup["reparto_id"].astype(str).str.strip()
        slots_lookup["shift_code"] = slots_lookup["shift_code"].astype(str).str.strip()
        start_ts = pd.to_datetime(slots_lookup["start_dt"], errors="coerce")
        if start_ts.isna().any():
            raise ValueError(
                "shift_slots: valori start_dt non validi durante il caricamento di locks.csv"
            )
        tz = getattr(start_ts.dt, "tz", None)
        if tz is not None:
            start_ts = start_ts.dt.tz_convert(None)
        slots_lookup["date"] = start_ts.dt.date
        slots_lookup = slots_lookup.drop(columns=["start_dt"]).drop_duplicates()

        merged = locks_df.merge(
            slots_lookup,
            on=["date", "reparto_id", "shift_code"],
            how="left",
            validate="many_to_one",
        )

        missing_slot = merged["slot_id"].isna()
        if missing_slot.any():
            missing_rows = (
                merged.loc[
                    missing_slot,
                    ["date", "reparto_id", "shift_code", "employee_id"],
                ]
                .drop_duplicates()
                .to_dict(orient="records")
            )
            warnings.warn(
                "locks.csv: slot non trovato per le combinazioni: "
                f"{missing_rows}",
                UserWarning,
                stacklevel=2,
            )

        valid = merged.loc[~missing_slot, ["employee_id", "slot_id", "lock"]].copy()
        if "note" in merged.columns:
            valid["note"] = merged.loc[~missing_slot, "note"].astype(str).str.strip()
        if not valid.empty:
            valid["slot_id"] = valid["slot_id"].astype("int64")
            records.append(valid.reset_index(drop=True))

    if not records:
        return pd.DataFrame(columns=["employee_id", "slot_id", "lock", "note"])

    combined = pd.concat(records, ignore_index=True, sort=False)
    combined["employee_id"] = combined["employee_id"].astype(str).str.strip()
    combined["slot_id"] = combined["slot_id"].astype("int64")
    combined["lock"] = combined["lock"].astype(int)

    if "note" not in combined.columns:
        combined["note"] = ""

    combined = combined.drop_duplicates().reset_index(drop=True)

    conflicts = (
        combined.groupby(["employee_id", "slot_id"])["lock"].nunique().reset_index()
    )
    conflicts = conflicts[conflicts["lock"] > 1]
    if not conflicts.empty:
        keys = [
            {"employee_id": row.employee_id, "slot_id": int(row.slot_id)}
            for row in conflicts.itertuples(index=False)
        ]
        raise ValueError(
            "locks: lock contrastanti per le chiavi specificate: " + str(keys)
        )

    combined = combined.drop_duplicates(subset=["employee_id", "slot_id", "lock"])
    columns_order = [
        "employee_id",
        "slot_id",
        "lock",
        *[col for col in combined.columns if col not in _REQUIRED_COLUMNS],
    ]
    combined = combined.loc[:, columns_order].reset_index(drop=True)

    warnings.warn(
        "locks.csv: caricati %s record" % len(combined),
        UserWarning,
        stacklevel=2,
    )

    return combined


def validate_locks(
    locks_df: pd.DataFrame,
    employees: pd.DataFrame,
    shift_slots: pd.DataFrame,
    absences_by_day: pd.DataFrame | None = None,
    shift_role_eligibility: pd.DataFrame | None = None,
    cross_reparto_enabled: bool = False,
) -> pd.DataFrame:
    """Cross-check locks against employees, slots, roles and absences."""

    if locks_df.empty:
        result_columns = list(locks_df.columns)
        for col in ("date", "reparto_id", "shift_code"):
            if col not in result_columns:
                result_columns.append(col)
        return pd.DataFrame(columns=result_columns)

    clean = locks_df.copy()

    clean["employee_id"] = clean["employee_id"].astype(str).str.strip()
    clean["slot_id"] = clean["slot_id"].astype("int64")

    employees_lookup = employees.loc[:, ["employee_id", "reparto_id", "role"]].copy()
    employees_lookup["employee_id"] = (
        employees_lookup["employee_id"].astype(str).str.strip()
    )
    employees_lookup["reparto_id"] = (
        employees_lookup["reparto_id"].astype(str).str.strip()
    )
    employees_lookup["role"] = employees_lookup["role"].astype(str).str.strip()
    employees_lookup = employees_lookup.rename(
        columns={"reparto_id": "employee_reparto_id", "role": "employee_role"}
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

    missing_emp = merged["employee_role"].isna()
    if missing_emp.any():
        bad = merged.loc[missing_emp, ["employee_id", "slot_id"]].drop_duplicates()
        keys = [
            {"employee_id": row.employee_id, "slot_id": int(row.slot_id)}
            for row in bad.itertuples(index=False)
        ]
        raise ValueError(
            "locks: employee_id inesistente rispetto a employees.csv: "
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
            "locks: slot_id inesistente rispetto a shift_slots: " + str(keys)
        )

    merged["shift_code"] = merged["shift_code"].astype(str).str.strip().str.upper()
    merged["employee_role"] = merged["employee_role"].astype(str).str.strip().str.upper()

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
                "locks: reparto diverso tra dipendente e slot (cross-reparto disabilitato): "
                f"{keys}"
            )
    elif reparto_mismatch.any():
        bad = merged.loc[
            reparto_mismatch,
            ["employee_id", "slot_id", "employee_reparto_id", "slot_reparto_id"],
        ].drop_duplicates()
        warnings.warn(
            "locks: reparto diverso tra dipendente e slot con cross-reparto abilitato: "
            f"{bad.to_dict(orient='records')}",
            UserWarning,
            stacklevel=2,
        )

    if shift_role_eligibility is not None and not shift_role_eligibility.empty:
        elig = shift_role_eligibility.loc[:, ["shift_code", "role", "allowed"]].copy()
        elig = elig.loc[elig["allowed"].fillna(False).astype(bool)]
        elig["shift_code"] = elig["shift_code"].astype(str).str.strip().str.upper()
        elig["role"] = elig["role"].astype(str).str.strip().str.upper()
        elig = (
            elig.drop_duplicates(subset=["shift_code", "role"])
            .rename(columns={"role": "employee_role"})
            .drop(columns=["allowed"])
        )

        merged = merged.merge(
            elig,
            on=["shift_code", "employee_role"],
            how="left",
            indicator="_elig_merge",
            validate="many_to_many",
        )

        invalid_role = merged["_elig_merge"] == "left_only"
        if invalid_role.any():
            bad = merged.loc[
                invalid_role,
                ["employee_id", "slot_id", "shift_code", "employee_role"],
            ].drop_duplicates()
            keys = [
                {
                    "employee_id": row.employee_id,
                    "slot_id": int(row.slot_id),
                    "shift_code": row.shift_code,
                    "role": row.employee_role,
                }
                for row in bad.itertuples(index=False)
            ]
            raise ValueError(
                "locks: role del dipendente non idoneo per il turno dello slot: "
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
            "locks: impossibile determinare la data dello slot per le chiavi: "
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
        is_absent = merged["is_absent"].astype("boolean", copy=False).fillna(False)
        conflict = merged["lock"].eq(1) & is_absent.astype(bool)
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
                "locks: lock=1 su giorni di assenza del dipendente: "
                f"{keys}"
            )
        merged = merged.drop(columns=["is_absent"])

    if "_elig_merge" in merged.columns:
        merged = merged.drop(columns=["_elig_merge"])

    merged = merged.rename(columns={"slot_reparto_id": "reparto_id"})
    merged = merged.drop(columns=["employee_reparto_id", "employee_role", "start_dt"])

    result_columns = list(clean.columns)
    for col in ("date", "reparto_id", "shift_code"):
        if col not in result_columns:
            result_columns.append(col)

    return merged.loc[:, result_columns].reset_index(drop=True)


def split_locks(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validated locks into must-assign and forbid DataFrames."""

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
