from __future__ import annotations

import pandas as pd

from .utils import LoaderError, _ensure_cols


def load_month_plan(path: str, shifts_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"data", "turno", "codice"}, "month_plan.csv")
    df["data"] = df["data"].astype(str).str.strip()
    df["turno"] = df["turno"].astype(str).str.strip()
    df["codice"] = df["codice"].astype(str).str.strip()
    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"month_plan.csv: formato data non valido: {exc}")
    bad_turni = sorted(set(df["turno"].unique()) - {"M", "P", "N"})
    if bad_turni:
        raise LoaderError(
            f"month_plan.csv: turni non ammessi: {bad_turni}. Ammessi: ['M','P','N']"
        )
    missing_in_catalog = sorted(
        set(df["turno"].unique()) - set(shifts_df["shift_id"].unique())
    )
    if missing_in_catalog:
        raise LoaderError(
            "month_plan.csv: turni assenti dal catalogo shifts.csv: "
            f"{missing_in_catalog}"
        )
    return df


def load_coverage_groups(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(
        df,
        {"codice", "turno", "gruppo", "total_min", "ruoli_totale"},
        "coverage_groups.csv",
    )
    df["total_min"] = pd.to_numeric(df["total_min"], errors="raise").astype(int)

    def _split_roles(s: str) -> list[str]:
        return [x.strip() for x in str(s).split("|") if x.strip()]

    df["ruoli_totale_list"] = df["ruoli_totale"].apply(_split_roles)
    if df.duplicated(subset=["codice", "turno", "gruppo"]).any():
        dup = df[
            df.duplicated(subset=["codice", "turno", "gruppo"], keep=False)
        ].sort_values(["codice", "turno", "gruppo"])
        raise LoaderError(
            "coverage_groups.csv: duplicati su (codice,turno,gruppo):\n" f"{dup}"
        )
    if df["ruoli_totale_list"].apply(len).eq(0).any():
        bad = df[df["ruoli_totale_list"].apply(len).eq(0)]
        raise LoaderError(
            f"coverage_groups.csv: ruoli_totale vuoto per righe:\n{bad}"
        )
    return df


def load_coverage_roles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(
        df, {"codice", "turno", "gruppo", "ruolo", "min_ruolo"}, "coverage_roles.csv"
    )
    df["min_ruolo"] = pd.to_numeric(df["min_ruolo"], errors="raise").astype(int)
    if df.duplicated(subset=["codice", "turno", "gruppo", "ruolo"]).any():
        dup = df[
            df.duplicated(subset=["codice", "turno", "gruppo", "ruolo"], keep=False)
        ].sort_values(["codice", "turno", "gruppo", "ruolo"])
        raise LoaderError(
            "coverage_roles.csv: duplicati su (codice,turno,gruppo,ruolo):\n"
            f"{dup}"
        )
    return df


def validate_groups_roles(
    groups: pd.DataFrame, roles: pd.DataFrame, eligibility_df: pd.DataFrame
) -> None:
    r_join = roles.merge(
        groups[["codice", "turno", "gruppo", "ruoli_totale_list"]],
        on=["codice", "turno", "gruppo"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    missing_grp = r_join[r_join["_merge"] == "left_only"][[
        "codice",
        "turno",
        "gruppo",
        "ruolo",
    ]].drop_duplicates()
    if not missing_grp.empty:
        raise LoaderError(
            "coverage_roles.csv: (codice,turno,gruppo) non definito in coverage_groups per:\n"
            f"{missing_grp}"
        )

    er = roles.merge(
        eligibility_df.rename(columns={"shift_id": "turno"}),
        on=["turno", "ruolo"],
        how="left",
        indicator=True,
    )
    bad = er[er["_merge"] == "left_only"][["codice", "turno", "gruppo", "ruolo"]].drop_duplicates()
    if not bad.empty:
        raise LoaderError(
            "coverage_roles.csv: (turno,ruolo) non idoneo secondo shift_role_eligibility:\n"
            f"{bad}"
        )

    rows = []
    for _, g in groups.iterrows():
        for ruolo in g["ruoli_totale_list"]:
            rows.append((g["codice"], g["turno"], g["gruppo"], ruolo))
    if rows:
        total_roles_df = pd.DataFrame(rows, columns=["codice", "turno", "gruppo", "ruolo"])
        tr = total_roles_df.merge(
            eligibility_df.rename(columns={"shift_id": "turno"}),
            on=["turno", "ruolo"],
            how="left",
            indicator=True,
        )
        bad2 = tr[tr["_merge"] == "left_only"][["codice", "turno", "gruppo", "ruolo"]].drop_duplicates()
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
            f"{not_in_set[['codice','turno','gruppo','ruolo']].drop_duplicates()}"
        )

    sums = (
        roles.groupby(["codice", "turno", "gruppo"], as_index=False)["min_ruolo"]
        .sum()
        .rename(columns={"min_ruolo": "sum_min_ruolo"})
    )
    chk = groups.merge(sums, on=["codice", "turno", "gruppo"], how="left").fillna(
        {"sum_min_ruolo": 0}
    )
    viol = chk[chk["total_min"] < chk["sum_min_ruolo"]]
    if not viol.empty:
        raise LoaderError(
            "Incoerenza: total_min < somma(min_ruolo) per:\n"
            f"{viol[['codice','turno','gruppo','total_min','sum_min_ruolo']]}"
        )


def expand_requirements(
    month_plan: pd.DataFrame, groups: pd.DataFrame, roles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = ["data", "turno", "codice"]
    optional_cols = [c for c in ["data_dt"] if c in month_plan.columns]
    month_plan_base = month_plan[base_cols + optional_cols]

    gt = month_plan_base.merge(
        groups, on=["codice", "turno"], how="left", validate="many_to_many"
    )
    if gt["gruppo"].isna().any():
        miss = gt[gt["gruppo"].isna()].drop_duplicates(subset=["codice", "turno"])[[
            "codice",
            "turno",
        ]]
        raise LoaderError(
            "month_plan contiene (codice,turno) senza definizione in coverage_groups:\n"
            f"{miss}"
        )
    gt["ruoli_totale_set"] = gt["ruoli_totale_list"].apply(lambda xs: "|".join(xs))
    ordered_cols = [
        "data",
        *optional_cols,
        "turno",
        "codice",
        "gruppo",
        "total_min",
        "ruoli_totale_set",
    ]
    gt = gt[ordered_cols].sort_values(["data", "turno", "codice", "gruppo"])

    gr = month_plan_base.merge(
        roles, on=["codice", "turno"], how="left", validate="many_to_many"
    )
    gr = gr.dropna(subset=["gruppo"], how="any")
    ordered_cols_roles = [
        "data",
        *optional_cols,
        "turno",
        "codice",
        "gruppo",
        "ruolo",
        "min_ruolo",
    ]
    gr = gr[ordered_cols_roles].sort_values(
        ["data", "turno", "codice", "gruppo", "ruolo"]
    )
    return gt.reset_index(drop=True), gr.reset_index(drop=True)
