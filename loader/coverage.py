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
    """Carica la tabella dei gruppi di copertura."""

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(
        df,
        {"coverage_code", "shift_code", "gruppo", "total_min", "ruoli_totale"},
        "coverage_groups.csv",
    )
    df["total_min"] = pd.to_numeric(df["total_min"], errors="raise").astype(int)

    def _split_roles(s: str) -> list[str]:
        return [x.strip() for x in str(s).split("|") if x.strip()]

    df["ruoli_totale_list"] = df["ruoli_totale"].apply(_split_roles)
    if df.duplicated(subset=["coverage_code", "shift_code", "gruppo"]).any():
        dup = df[
            df.duplicated(subset=["coverage_code", "shift_code", "gruppo"], keep=False)
        ].sort_values(["coverage_code", "shift_code", "gruppo"])
        raise LoaderError(
            "coverage_groups.csv: duplicati su (coverage_code,shift_code,gruppo):\n"
            f"{dup}"
        )
    if df["ruoli_totale_list"].apply(len).eq(0).any():
        bad = df[df["ruoli_totale_list"].apply(len).eq(0)]
        raise LoaderError(
            f"coverage_groups.csv: ruoli_totale vuoto per righe:\n{bad}"
        )
    return df


def load_coverage_roles(path: str) -> pd.DataFrame:
    """Carica la tabella dei requisiti minimi per ruolo."""

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(
        df,
        {"coverage_code", "shift_code", "gruppo", "ruolo", "min_ruolo"},
        "coverage_roles.csv",
    )
    df["min_ruolo"] = pd.to_numeric(df["min_ruolo"], errors="raise").astype(int)
    if df.duplicated(subset=["coverage_code", "shift_code", "gruppo", "ruolo"]).any():
        dup = df[
            df.duplicated(
                subset=["coverage_code", "shift_code", "gruppo", "ruolo"], keep=False
            )
        ].sort_values(["coverage_code", "shift_code", "gruppo", "ruolo"])
        raise LoaderError(
            "coverage_roles.csv: duplicati su (coverage_code,shift_code,gruppo,ruolo):\n"
            f"{dup}"
        )
    return df


def validate_groups_roles(
    groups: pd.DataFrame, roles: pd.DataFrame, eligibility_df: pd.DataFrame
) -> None:
    """Convalida la coerenza tra gruppi, ruoli e idoneità dei turni."""

    r_join = roles.merge(
        groups[["coverage_code", "shift_code", "gruppo", "ruoli_totale_list"]],
        on=["coverage_code", "shift_code", "gruppo"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    missing_grp = r_join[r_join["_merge"] == "left_only"][[
        "coverage_code",
        "shift_code",
        "gruppo",
        "ruolo",
    ]].drop_duplicates()
    if not missing_grp.empty:
        raise LoaderError(
            "coverage_roles.csv: (coverage_code,shift_code,gruppo) non definito in coverage_groups per:\n"
            f"{missing_grp}"
        )

    er = roles.merge(
        eligibility_df.rename(columns={"shift_id": "shift_code"}),
        on=["shift_code", "ruolo"],
        how="left",
        indicator=True,
    )
    bad = er[er["_merge"] == "left_only"][[
        "coverage_code",
        "shift_code",
        "gruppo",
        "ruolo",
    ]].drop_duplicates()
    if not bad.empty:
        raise LoaderError(
            "coverage_roles.csv: (shift_code,ruolo) non idoneo secondo shift_role_eligibility:\n"
            f"{bad}"
        )

    rows = []
    for _, g in groups.iterrows():
        for ruolo in g["ruoli_totale_list"]:
            rows.append((g["coverage_code"], g["shift_code"], g["gruppo"], ruolo))
    if rows:
        total_roles_df = pd.DataFrame(
            rows, columns=["coverage_code", "shift_code", "gruppo", "ruolo"]
        )
        tr = total_roles_df.merge(
            eligibility_df.rename(columns={"shift_id": "shift_code"}),
            on=["shift_code", "ruolo"],
            how="left",
            indicator=True,
        )
        bad2 = tr[tr["_merge"] == "left_only"][[
            "coverage_code",
            "shift_code",
            "gruppo",
            "ruolo",
        ]].drop_duplicates()
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
            f"{not_in_set[['coverage_code','shift_code','gruppo','ruolo']].drop_duplicates()}"
        )

    sums = (
        roles.groupby(["coverage_code", "shift_code", "gruppo"], as_index=False)[
            "min_ruolo"
        ]
        .sum()
        .rename(columns={"min_ruolo": "sum_min_ruolo"})
    )
    chk = groups.merge(
        sums, on=["coverage_code", "shift_code", "gruppo"], how="left"
    ).fillna({"sum_min_ruolo": 0})
    viol = chk[chk["total_min"] < chk["sum_min_ruolo"]]
    if not viol.empty:
        raise LoaderError(
            "Incoerenza: total_min < somma(min_ruolo) per:\n"
            f"{viol[['coverage_code','shift_code','gruppo','total_min','sum_min_ruolo']]}"
        )


def expand_requirements(
    month_plan: pd.DataFrame, groups: pd.DataFrame, roles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Espande month plan unendo gruppi e ruoli di copertura."""

    base_cols = ["data", "reparto_id", "shift_code", "coverage_code"]
    optional_cols = [c for c in ["data_dt"] if c in month_plan.columns]
    month_plan_base = month_plan[base_cols + optional_cols]

    gt = month_plan_base.merge(
        groups,
        on=["coverage_code", "shift_code"],
        how="left",
        validate="many_to_many",
    )
    if gt["gruppo"].isna().any():
        miss = gt[gt["gruppo"].isna()].drop_duplicates(
            subset=["coverage_code", "shift_code"]
        )[["coverage_code", "shift_code"]]
        raise LoaderError(
            "month_plan contiene (coverage_code,shift_code) senza definizione in coverage_groups:\n"
            f"{miss}"
        )
    gt["ruoli_totale_set"] = gt["ruoli_totale_list"].apply(lambda xs: "|".join(xs))
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
        on=["coverage_code", "shift_code"],
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

    return gt.reset_index(drop=True), gr.reset_index(drop=True)


def build_slot_requirements(
    slots_df: pd.DataFrame, coverage_roles_df: pd.DataFrame
) -> pd.DataFrame:
    """Costruisce la domanda per ruolo per ciascuno slot."""

    if slots_df.empty or coverage_roles_df.empty:
        return pd.DataFrame(columns=["slot_id", "ruolo", "demand"])

    grouped = (
        coverage_roles_df.groupby(["coverage_code", "shift_code", "ruolo"], as_index=False)[
            "min_ruolo"
        ]
        .sum()
        .rename(columns={"min_ruolo": "demand"})
    )

    merged = slots_df.merge(
        grouped,
        on=["coverage_code", "shift_code"],
        how="left",
    )
    merged = merged.dropna(subset=["demand"])
    if merged.empty:
        return pd.DataFrame(columns=["slot_id", "ruolo", "demand"])

    merged["demand"] = merged["demand"].astype(int)
    out = merged[["slot_id", "ruolo", "demand"]].sort_values(
        ["slot_id", "ruolo"]
    )
    return out.reset_index(drop=True)
