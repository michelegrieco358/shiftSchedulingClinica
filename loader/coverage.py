from __future__ import annotations

import pandas as pd

from .utils import LoaderError, _ensure_cols, _resolve_allowed_departments


def load_month_plan(
    path: str, shifts_df: pd.DataFrame, defaults: dict[str, object]
) -> pd.DataFrame:
    """Carica il month plan con reparti e codici di copertura."""

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(
        df,
        {"data", "reparto_id", "shift_code", "coverage_code"},
        "month_plan.csv",
    )

    df["data"] = df["data"].astype(str).str.strip()
    df["reparto_id"] = df["reparto_id"].astype(str).str.strip()
    df["shift_code"] = df["shift_code"].astype(str).str.strip()
    df["coverage_code"] = df["coverage_code"].astype(str).str.strip()

    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"month_plan.csv: formato data non valido: {exc}")

    allowed_departments = set(_resolve_allowed_departments(defaults))
    bad_depts = sorted(set(df["reparto_id"].unique()) - allowed_departments)
    if bad_depts:
        raise LoaderError(
            "month_plan.csv: reparti non ammessi rispetto alla config (defaults.departments): "
            f"{bad_depts}"
        )

    known_shifts = set(shifts_df["shift_id"].unique())
    bad_shifts = sorted(set(df["shift_code"].unique()) - known_shifts)
    if bad_shifts:
        raise LoaderError(
            "month_plan.csv: shift_code assenti dal catalogo shifts.csv: "
            f"{bad_shifts}"
        )

    if (df["coverage_code"] == "").any():
        bad = df.loc[df["coverage_code"] == "", ["data", "reparto_id", "shift_code"]]
        raise LoaderError(
            "month_plan.csv: coverage_code non può essere vuoto. Righe interessate:\n"
            f"{bad.head()}"
        )

    return df[["data", "data_dt", "reparto_id", "shift_code", "coverage_code"]]

def load_coverage_groups(path: str) -> pd.DataFrame:
    """Carica la tabella dei gruppi di copertura includendo il reparto di riferimento."""

    df = pd.read_csv(path, dtype=str).fillna("").copy()
    _ensure_cols(
        df,
        {"coverage_code", "shift_code", "reparto_id", "gruppo", "total_min", "ruoli_totale"},
        "coverage_groups.csv",
    )

    df = df.assign(
        coverage_code=df["coverage_code"].astype(str).str.strip().str.upper(),
        shift_code=df["shift_code"].astype(str).str.strip().str.upper(),
        reparto_id=df["reparto_id"].astype(str).str.strip().str.upper(),
        gruppo=df["gruppo"].astype(str).str.strip().str.upper(),
        ruoli_totale=df["ruoli_totale"].astype(str).str.strip(),
    )

    blank_reparto = df["reparto_id"] == ""
    if blank_reparto.any():
        rows = ", ".join(str(idx + 2) for idx in df.index[blank_reparto])
        raise ValueError(
            "coverage_groups.csv: reparto_id mancante alle righe "
            f"[{rows}]"
        )

    df["total_min"] = pd.to_numeric(df["total_min"], errors="raise").astype(int)

    def _split_roles(s: str) -> list[str]:
        return [x.strip().upper() for x in str(s).split("|") if x.strip()]

    df = df.assign(ruoli_totale_list=df["ruoli_totale"].apply(_split_roles))
    if df["ruoli_totale_list"].apply(len).eq(0).any():
        rows = df.index[df["ruoli_totale_list"].apply(len).eq(0)]
        detail = ", ".join(str(idx + 2) for idx in rows)
        raise ValueError(
            "coverage_groups.csv: ruoli_totale vuoto alle righe "
            f"[{detail}]"
        )

    dup_mask = df.duplicated(
        subset=["coverage_code", "shift_code", "reparto_id", "gruppo"], keep=False
    )
    if dup_mask.any():
        keys = (
            df.loc[dup_mask, ["coverage_code", "shift_code", "reparto_id", "gruppo"]]
            .drop_duplicates()
            .sort_values(["coverage_code", "shift_code", "reparto_id", "gruppo"])
        )
        raise ValueError(
            "coverage_groups.csv: duplicati su (coverage_code, shift_code, reparto_id, gruppo): "
            f"{keys.to_dict(orient='records')}"
        )

    return df.copy()


def load_coverage_roles(path: str) -> pd.DataFrame:
    """Carica la tabella dei requisiti minimi per ruolo includendo il reparto."""

    df = pd.read_csv(path, dtype=str).fillna("").copy()
    _ensure_cols(
        df,
        {"coverage_code", "shift_code", "reparto_id", "gruppo", "ruolo", "min_ruolo"},
        "coverage_roles.csv",
    )

    df = df.assign(
        coverage_code=df["coverage_code"].astype(str).str.strip().str.upper(),
        shift_code=df["shift_code"].astype(str).str.strip().str.upper(),
        reparto_id=df["reparto_id"].astype(str).str.strip().str.upper(),
        gruppo=df["gruppo"].astype(str).str.strip().str.upper(),
        ruolo=df["ruolo"].astype(str).str.strip().str.upper(),
    )

    blank_reparto = df["reparto_id"] == ""
    if blank_reparto.any():
        rows = ", ".join(str(idx + 2) for idx in df.index[blank_reparto])
        raise ValueError(
            "coverage_roles.csv: reparto_id mancante alle righe "
            f"[{rows}]"
        )

    df["min_ruolo"] = pd.to_numeric(df["min_ruolo"], errors="raise").astype(int)

    dup_mask = df.duplicated(
        subset=["coverage_code", "shift_code", "reparto_id", "gruppo", "ruolo"], keep=False
    )
    if dup_mask.any():
        keys = (
            df.loc[dup_mask, ["coverage_code", "shift_code", "reparto_id", "gruppo", "ruolo"]]
            .drop_duplicates()
            .sort_values(["coverage_code", "shift_code", "reparto_id", "gruppo", "ruolo"])
        )
        raise ValueError(
            "coverage_roles.csv: duplicati su (coverage_code, shift_code, reparto_id, gruppo, ruolo): "
            f"{keys.to_dict(orient='records')}"
        )

    return df.copy()


def validate_groups_roles(
    groups: pd.DataFrame, roles: pd.DataFrame, eligibility_df: pd.DataFrame
) -> None:
    """Convalida la coerenza tra gruppi, ruoli e idoneita dei turni con dettaglio per reparto."""

    r_join = roles.merge(
        groups[["coverage_code", "shift_code", "reparto_id", "gruppo", "ruoli_totale_list"]],
        on=["coverage_code", "shift_code", "reparto_id", "gruppo"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    missing_grp = r_join[r_join["_merge"] == "left_only"][
        ["coverage_code", "shift_code", "reparto_id", "gruppo", "ruolo"]
    ].drop_duplicates()
    if not missing_grp.empty:
        raise LoaderError(
            "coverage_roles.csv: (coverage_code,shift_code,reparto_id,gruppo) non definito in coverage_groups per:\n"
            f"{missing_grp}"
        )

    er = roles.merge(
        eligibility_df.rename(columns={"shift_id": "shift_code"}),
        on=["shift_code", "ruolo"],
        how="left",
        indicator=True,
    )
    bad = er[er["_merge"] == "left_only"][
        ["coverage_code", "shift_code", "reparto_id", "gruppo", "ruolo"]
    ].drop_duplicates()
    if not bad.empty:
        raise LoaderError(
            "coverage_roles.csv: (shift_code,ruolo) non idoneo secondo shift_role_eligibility:\n"
            f"{bad}"
        )

    rows = []
    for _, g in groups.iterrows():
        for ruolo in g["ruoli_totale_list"]:
            rows.append(
                (
                    g["coverage_code"],
                    g["shift_code"],
                    g["reparto_id"],
                    g["gruppo"],
                    ruolo,
                )
            )
    if rows:
        total_roles_df = pd.DataFrame(
            rows, columns=["coverage_code", "shift_code", "reparto_id", "gruppo", "ruolo"]
        )
        tr = total_roles_df.merge(
            eligibility_df.rename(columns={"shift_id": "shift_code"}),
            on=["shift_code", "ruolo"],
            how="left",
            indicator=True,
        )
        bad2 = tr[tr["_merge"] == "left_only"][
            ["coverage_code", "shift_code", "reparto_id", "gruppo", "ruolo"]
        ].drop_duplicates()
        if not bad2.empty:
            raise LoaderError(
                "coverage_groups.csv: ruoli_totale include ruoli non idonei per il turno:\n"
                f"{bad2}"
            )

    not_in_set = r_join[
        ~r_join.apply(lambda r: r["ruolo"] in set(r["ruoli_totale_list"] or []), axis=1)
    ]
    if not not_in_set.empty:
        raise LoaderError(
            "coverage_roles.csv: ruoli non inclusi in ruoli_totale_list del gruppo corrispondente:\n"
            f"{not_in_set[['coverage_code','shift_code','reparto_id','gruppo','ruolo']].drop_duplicates()}"
        )

    sums = (
        roles.groupby(["coverage_code", "shift_code", "reparto_id", "gruppo"], as_index=False)[
            "min_ruolo"
        ]
        .sum()
        .rename(columns={"min_ruolo": "sum_min_ruolo"})
    )
    chk = groups.merge(
        sums, on=["coverage_code", "shift_code", "reparto_id", "gruppo"], how="left"
    ).fillna({"sum_min_ruolo": 0})
    viol = chk[chk["total_min"] < chk["sum_min_ruolo"]]
    if not viol.empty:
        raise LoaderError(
            "Incoerenza: total_min < somma(min_ruolo) per:\n"
            f"{viol[['coverage_code','shift_code','reparto_id','gruppo','total_min','sum_min_ruolo']]}"
        )


def expand_requirements(
    month_plan: pd.DataFrame, groups: pd.DataFrame, roles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Espande month plan unendo gruppi e ruoli di copertura per reparto."""

    base_cols = ["data", "reparto_id", "shift_code", "coverage_code"]
    optional_cols = [c for c in ["data_dt"] if c in month_plan.columns]
    month_plan_base = month_plan.loc[:, base_cols + optional_cols].copy()
    month_plan_base = month_plan_base.assign(
        reparto_id=month_plan_base["reparto_id"].astype(str).str.strip().str.upper(),
        shift_code=month_plan_base["shift_code"].astype(str).str.strip().str.upper(),
        coverage_code=month_plan_base["coverage_code"].astype(str)
        .str.strip()
        .str.upper(),
    )

    gt = month_plan_base.merge(
        groups,
        on=["coverage_code", "shift_code", "reparto_id"],
        how="left",
        validate="many_to_many",
    )
    if gt["gruppo"].isna().any():
        miss = gt[gt["gruppo"].isna()].drop_duplicates(
            subset=["coverage_code", "shift_code", "reparto_id"]
        )[["coverage_code", "shift_code", "reparto_id"]]
        raise LoaderError(
            "month_plan contiene (coverage_code,shift_code,reparto_id) senza definizione in coverage_groups:\n"
            f"{miss}"
        )
    gt = gt.assign(
        ruoli_totale_set=gt["ruoli_totale_list"].apply(lambda xs: "|".join(xs))
    )
    ordered_cols = [
        "data",
        *optional_cols,
        "reparto_id",
        "shift_code",
        "coverage_code",
        "gruppo",
        "total_min",
        "ruoli_totale_set",
    ]
    gt = gt[ordered_cols].sort_values(
        ["data", "reparto_id", "shift_code", "coverage_code", "gruppo"]
    )

    gr = month_plan_base.merge(
        roles,
        on=["coverage_code", "shift_code", "reparto_id"],
        how="left",
        validate="many_to_many",
    )
    gr = gr.dropna(subset=["gruppo"], how="any")
    ordered_cols_roles = [
        "data",
        *optional_cols,
        "reparto_id",
        "shift_code",
        "coverage_code",
        "gruppo",
        "ruolo",
        "min_ruolo",
    ]
    gr = gr[ordered_cols_roles].sort_values(
        ["data", "reparto_id", "shift_code", "coverage_code", "gruppo", "ruolo"]
    )

    return gt.reset_index(drop=True).copy(), gr.reset_index(drop=True).copy()


def build_slot_requirements(
    slots_df: pd.DataFrame, coverage_roles_df: pd.DataFrame
) -> pd.DataFrame:
    """Costruisce la domanda per ruolo per ciascuno slot, distinguendo per reparto."""

    if slots_df.empty or coverage_roles_df.empty:
        return pd.DataFrame(columns=["slot_id", "reparto_id", "ruolo", "demand"])

    grouped = (
        coverage_roles_df.groupby(
            ["coverage_code", "shift_code", "reparto_id", "ruolo"], as_index=False
        )["min_ruolo"]
        .sum()
        .rename(columns={"min_ruolo": "demand"})
    )

    slots_merge = slots_df.copy()
    if not slots_merge.empty:
        slots_merge = slots_merge.assign(
            coverage_code=slots_merge["coverage_code"].astype(str).str.strip().str.upper(),
            shift_code=slots_merge["shift_code"].astype(str).str.strip().str.upper(),
            reparto_id=slots_merge["reparto_id"].astype(str).str.strip().str.upper(),
        )

    merged = slots_merge.merge(
        grouped,
        on=["coverage_code", "shift_code", "reparto_id"],
        how="left",
        validate="many_to_many",
    )

    missing_mask = merged["demand"].isna()
    if missing_mask.any():
        missing = (
            merged.loc[missing_mask, ["coverage_code", "shift_code", "reparto_id"]]
            .drop_duplicates()
            .sort_values(["coverage_code", "shift_code", "reparto_id"])
        )
        raise ValueError(
            "slots privi di requisiti per (coverage_code, shift_code, reparto_id): "
            f"{missing.to_dict(orient='records')}"
        )

    merged = merged.assign(demand=merged["demand"].astype(int))

    ordered = ["slot_id", "reparto_id", "ruolo", "demand"]
    extra_cols = [c for c in merged.columns if c not in ordered]
    out_cols = ordered + extra_cols

    out = merged[out_cols].sort_values(["slot_id", "reparto_id", "ruolo"])
    return out.reset_index(drop=True).copy()
