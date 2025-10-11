"""Utilities for cross-department policies, validation, and reporting."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _normalise(series: pd.Series) -> pd.Series:
    """Return an upper-cased, stripped string representation of ``series``."""

    return series.astype(str).str.strip().str.upper()


def _ensure_non_negative_default(
    config: dict[str, Any], key: str, *, is_float: bool
) -> float:
    if "cross" not in config or not isinstance(config["cross"], dict):
        raise ValueError("config: sezione 'cross' mancante o non valida")

    value = config["cross"].get(key)
    if value is None:
        raise ValueError(f"config: valore predefinito mancante per '{key}'")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - messaggio esplicito
        raise ValueError(
            f"config: valore non numerico per '{key}': {value!r}"
        ) from exc

    if numeric_value < 0:
        raise ValueError(
            f"config: '{key}' deve essere maggiore o uguale a zero (trovato {value!r})"
        )

    if not is_float and abs(numeric_value - round(numeric_value)) > 1e-9:
        raise ValueError(
            f"config: '{key}' deve essere un intero non negativo (trovato {value!r})"
        )

    return numeric_value if is_float else float(round(numeric_value))


def enrich_employees_with_cross_policy(
    employees: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Attach cross-department policy defaults to employees."""

    required_cols = ["employee_id", "reparto_id", "role"]
    missing = [col for col in required_cols if col not in employees.columns]
    if missing:
        raise ValueError(
            "employees: colonne mancanti: " + ", ".join(sorted(missing))
        )

    defaults = {
        "cross_max_shifts_month": _ensure_non_negative_default(
            config, "max_shifts_month", is_float=False
        )
    }

    # Ensure the global penalty weight is valid even if overrides are forbidden.
    _ensure_non_negative_default(config, "penalty_weight", is_float=True)

    enriched = employees.copy()

    if "cross_penalty_weight" in enriched.columns:
        raise ValueError(
            "employees: la colonna 'cross_penalty_weight' non è più supportata; "
            "configurare il peso in config.yaml"
        )

    for column, default_value in defaults.items():
        if column in enriched.columns:
            raw_series = enriched[column]
            coerced = pd.to_numeric(raw_series, errors="coerce")
            invalid_non_numeric = (
                raw_series.notna()
                & raw_series.astype(str).str.strip().ne("")
                & coerced.isna()
            )
            if invalid_non_numeric.any():
                raise ValueError(
                    f"employees: '{column}' contiene valori non numerici"
                )
            invalid_fraction = coerced.dropna() % 1 != 0
            if invalid_fraction.any():
                raise ValueError(
                    f"employees: '{column}' deve contenere soli interi non negativi"
                )
            coerced = coerced.round()

            if (coerced.dropna() < 0).any():
                raise ValueError(
                    f"employees: '{column}' non può contenere valori negativi"
                )
        else:
            coerced = pd.Series(
                [float("nan")] * len(enriched), index=enriched.index, dtype="float64"
            )

        filled = coerced.fillna(default_value)
        enriched[column] = filled.astype(int)

    if "cross_max_shifts_month" not in enriched.columns:
        enriched["cross_max_shifts_month"] = defaults["cross_max_shifts_month"]

    return enriched


def validate_candidates_cross(
    candidate_assignments: pd.DataFrame,
    employee_allowed_reparti: pd.DataFrame,
    shift_role_eligibility: pd.DataFrame,
    absences_by_day: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate candidate assignments against pool, role, and absence rules."""

    required_candidate_cols = [
        "employee_id",
        "slot_id",
        "slot_reparto_id",
        "reparto_home",
        "role",
        "shift_code",
        "date",
    ]
    missing_candidate = [
        col for col in required_candidate_cols if col not in candidate_assignments.columns
    ]
    if missing_candidate:
        raise ValueError(
            "candidate_assignments: colonne mancanti: "
            + ", ".join(sorted(missing_candidate))
        )

    required_allowed_cols = ["employee_id", "reparto_id_allowed"]
    missing_allowed = [
        col for col in required_allowed_cols if col not in employee_allowed_reparti.columns
    ]
    if missing_allowed:
        raise ValueError(
            "employee_allowed_reparti: colonne mancanti: "
            + ", ".join(sorted(missing_allowed))
        )

    required_role_cols = ["shift_code", "role", "allowed"]
    missing_role = [
        col for col in required_role_cols if col not in shift_role_eligibility.columns
    ]
    if missing_role:
        raise ValueError(
            "shift_role_eligibility: colonne mancanti: "
            + ", ".join(sorted(missing_role))
        )

    original_columns = candidate_assignments.columns.tolist()
    candidates = candidate_assignments.copy()
    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce").dt.date
    if candidates["date"].isna().any():
        raise ValueError("candidate_assignments: date non valide")

    candidates["_employee_key"] = _normalise(candidates["employee_id"])
    candidates["_slot_reparto_key"] = _normalise(candidates["slot_reparto_id"])
    candidates["_role_key"] = _normalise(candidates["role"])
    candidates["_shift_code_key"] = _normalise(candidates["shift_code"])

    allowed_df = employee_allowed_reparti.copy()
    allowed_df["_employee_key"] = _normalise(allowed_df["employee_id"])
    allowed_df["_reparto_allowed_key"] = _normalise(
        allowed_df["reparto_id_allowed"]
    )

    allowed_index = pd.MultiIndex.from_frame(
        allowed_df[["_employee_key", "_reparto_allowed_key"]].drop_duplicates()
    )
    candidate_pairs = pd.MultiIndex.from_arrays(
        [candidates["_employee_key"], candidates["_slot_reparto_key"]]
    )
    mask_allowed = candidate_pairs.isin(allowed_index)

    role_df = shift_role_eligibility.copy()
    role_df = role_df.loc[role_df["allowed"].fillna(False).astype(bool)]
    role_df["_role_key"] = _normalise(role_df["role"])
    role_df["_shift_code_key"] = _normalise(role_df["shift_code"])
    role_index = pd.MultiIndex.from_frame(
        role_df[["_role_key", "_shift_code_key"]].drop_duplicates()
    )
    candidate_role_pairs = pd.MultiIndex.from_arrays(
        [candidates["_role_key"], candidates["_shift_code_key"]]
    )
    mask_role = candidate_role_pairs.isin(role_index)

    mask_absent = pd.Series(True, index=candidates.index)
    if absences_by_day is not None and not absences_by_day.empty:
        required_abs_cols = ["employee_id", "date", "is_absent"]
        missing_abs_cols = [
            col for col in required_abs_cols if col not in absences_by_day.columns
        ]
        if missing_abs_cols:
            raise ValueError(
                "absences_by_day: colonne mancanti: "
                + ", ".join(sorted(missing_abs_cols))
            )
        abs_df = absences_by_day.copy()
        abs_df["date"] = pd.to_datetime(abs_df["date"], errors="coerce").dt.date
        if abs_df["date"].isna().any():
            raise ValueError("absences_by_day: date non valide")
        abs_df = abs_df.loc[abs_df["is_absent"].fillna(False).astype(bool)]
        if not abs_df.empty:
            abs_df["_employee_key"] = _normalise(abs_df["employee_id"])
            absence_index = pd.MultiIndex.from_frame(
                abs_df[["_employee_key", "date"]].drop_duplicates()
            )
            candidate_abs_pairs = pd.MultiIndex.from_arrays(
                [candidates["_employee_key"], candidates["date"]]
            )
            mask_absent = ~candidate_abs_pairs.isin(absence_index)

    reason = pd.Series(pd.NA, index=candidates.index, dtype="object")
    reason.loc[~mask_allowed] = "reparto_not_allowed"
    reason.loc[mask_allowed & ~mask_role & reason.isna()] = "role_not_allowed"
    reason.loc[
        mask_allowed & mask_role & ~mask_absent & reason.isna()
    ] = "employee_absent"

    candidates_ok = candidates.loc[reason.isna()].copy()
    dropped = candidates.loc[reason.notna()].copy()
    dropped = dropped.assign(reason=reason.loc[reason.notna()].values)

    helper_cols_ok = [col for col in candidates_ok.columns if col.startswith("_")]
    if helper_cols_ok:
        candidates_ok = candidates_ok.drop(columns=helper_cols_ok)

    helper_cols_dropped = [col for col in dropped.columns if col.startswith("_")]
    if helper_cols_dropped:
        dropped = dropped.drop(columns=helper_cols_dropped)

    dropped_cols = [
        "employee_id",
        "slot_id",
        "reason",
        "reparto_home",
        "slot_reparto_id",
        "date",
        "role",
        "shift_code",
    ]
    dropped = dropped.loc[:, dropped_cols]

    candidates_ok = candidates_ok.loc[:, original_columns]

    return candidates_ok.reset_index(drop=True), dropped.reset_index(drop=True)


def cross_reporting(candidate_assignments_ok: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build debug reports about potential cross-department assignments."""

    required_cols = [
        "employee_id",
        "reparto_home",
        "slot_reparto_id",
        "is_cross",
        "must",
        "is_night",
    ]
    missing_cols = [
        col for col in required_cols if col not in candidate_assignments_ok.columns
    ]
    if missing_cols:
        raise ValueError(
            "candidate_assignments: colonne mancanti: "
            + ", ".join(sorted(missing_cols))
        )

    df = candidate_assignments_ok.copy()
    df["is_cross"] = df["is_cross"].fillna(False).astype(bool)
    df["must"] = df["must"].fillna(0).astype(int)
    df["is_night"] = df["is_night"].fillna(False).astype(bool)

    flow_by_reparto = (
        df.groupby(["reparto_home", "slot_reparto_id", "is_cross"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["reparto_home", "slot_reparto_id", "is_cross"])
        .reset_index(drop=True)
    )

    cross_only = df.loc[df["is_cross"]]
    if cross_only.empty:
        flow_matrix = pd.DataFrame(columns=[])
    else:
        flow_matrix = (
            cross_only.groupby(["reparto_home", "slot_reparto_id"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
            .sort_index(axis=1)
        )

    employee_group = df.groupby("employee_id", dropna=False)
    employee_summary = employee_group.agg(
        reparto_home=("reparto_home", "first"),
        cand_total=("slot_id", "size"),
        cand_cross=("is_cross", "sum"),
        must_count=("must", "sum"),
        night_cand=("is_night", "sum"),
    )

    employee_summary["day_cand"] = (
        employee_summary["cand_total"] - employee_summary["night_cand"]
    )

    for column in ["cand_total", "cand_cross", "must_count", "night_cand", "day_cand"]:
        employee_summary[column] = employee_summary[column].astype(int)

    cross_dest = (
        df.loc[df["is_cross"], ["employee_id", "slot_reparto_id"]]
        .dropna()
        .drop_duplicates()
    )
    dest_counts = cross_dest.groupby("employee_id").size()
    employee_summary["dest_count"] = (
        employee_summary.index.map(dest_counts).fillna(0).astype(int)
    )

    employee_summary["share_cross"] = employee_summary["cand_cross"].astype(float)
    non_zero = employee_summary["cand_total"] != 0
    employee_summary.loc[non_zero, "share_cross"] = (
        employee_summary.loc[non_zero, "share_cross"]
        / employee_summary.loc[non_zero, "cand_total"].astype(float)
    )
    employee_summary.loc[~non_zero, "share_cross"] = 0.0

    employee_summary = (
        employee_summary.reset_index()
        .loc[
            :,
            [
                "employee_id",
                "reparto_home",
                "cand_total",
                "cand_cross",
                "share_cross",
                "dest_count",
                "must_count",
                "night_cand",
                "day_cand",
            ],
        ]
    )

    return {
        "flow_by_reparto": flow_by_reparto,
        "flow_matrix": flow_matrix,
        "employee_summary": employee_summary,
    }

