from __future__ import annotations

import os
from typing import Any

import pandas as pd

from .utils import (
    LoaderError,
    _ensure_cols,
    _resolve_allowed_departments,
    _resolve_allowed_roles,
)


def load_employees(path: str, defaults: dict[str, Any]) -> pd.DataFrame:
    """Carica e valida dati dipendenti con ore, limiti e contatori."""
    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(
        df,
        {
            "employee_id",
            "nome",
            "ruolo",
            "reparto",
            "ore_dovute_mese_h",
            "saldo_prog_iniziale_h",
        },
        "employees.csv",
    )

    if df["employee_id"].duplicated().any():
        dups = df[df["employee_id"].duplicated(keep=False)].sort_values("employee_id")
        raise LoaderError(
            f"employees.csv: employee_id duplicati:\n{dups[['employee_id','nome','ruolo']]}"
        )

    allowed_roles = _resolve_allowed_roles(defaults, fallback_roles=df["ruolo"].unique())

    bad_roles = sorted(set(df["ruolo"].unique()) - set(allowed_roles))
    if bad_roles:
        raise LoaderError(f"employees.csv: ruoli non ammessi rispetto alla config: {bad_roles}")

    allowed_departments = _resolve_allowed_departments(defaults)
    df["reparto"] = df["reparto"].astype(str).str.strip()
    if (df["reparto"] == "").any():
        bad = df.loc[df["reparto"] == "", ["employee_id", "nome", "ruolo"]]
        raise LoaderError(
            "employees.csv: la colonna 'reparto' è obbligatoria e non può essere vuota. "
            f"Righe interessate:\n{bad}"
        )

    bad_departments = sorted(set(df["reparto"].unique()) - set(allowed_departments))
    if bad_departments:
        raise LoaderError(
            "employees.csv: reparti non ammessi rispetto alla config (defaults.departments): "
            f"{bad_departments}"
        )

    def parse_hours_nonneg(x, field_name: str) -> float:
        """Converte valore in ore float non negativo con validazione."""
        s = str(x).strip()
        try:
            v = float(s)
        except ValueError:
            raise LoaderError(f"employees.csv: valore non numerico in '{field_name}': {x!r}")
        if v < 0:
            raise LoaderError(
                f"employees.csv: valore negativo non ammesso in '{field_name}': {v}"
            )
        return v

    def to_min_from_hours(v_hours: float) -> int:
        """Converte ore in minuti arrotondando."""
        return int(round(v_hours * 60.0))

    def parse_hours_allow_negative(x, field_name: str) -> float:
        """Converte valore in ore float ammettendo negativi."""
        s = str(x).strip()
        try:
            return float(s)
        except ValueError:
            raise LoaderError(f"employees.csv: valore non numerico in '{field_name}': {x!r}")

    contract_by_role = defaults.get("contract_hours_by_role_h", {}) or {}

    dovuto_min = []
    for _, r in df.iterrows():
        raw = str(r["ore_dovute_mese_h"]).strip()
        ruolo = str(r["ruolo"]).strip()
        if raw != "":
            hours = parse_hours_nonneg(raw, "ore_dovute_mese_h")
        else:
            if ruolo not in contract_by_role:
                raise LoaderError(
                    "employees.csv: ore_dovute_mese_h mancante e nessun default in config per ruolo "
                    f"'{ruolo}' (defaults.contract_hours_by_role_h)"
                )
            hours = parse_hours_nonneg(
                contract_by_role[ruolo], f"contract_hours_by_role_h[{ruolo}]"
            )
        dovuto_min.append(to_min_from_hours(hours))
    df["dovuto_min"] = dovuto_min

    df["saldo_init_min"] = df["saldo_prog_iniziale_h"].apply(
        lambda x: to_min_from_hours(
            parse_hours_allow_negative(x, "saldo_prog_iniziale_h")
        )
    )

    def get_hours_with_default(col_name: str, default_val) -> pd.Series:
        """Estrae colonna ore con fallback a default, convertendo in minuti."""
        if col_name in df.columns:
            ser = df[col_name].astype(str).str.strip()
            ser = ser.where(ser != "", other=str(default_val))
        else:
            ser = pd.Series([str(default_val)] * len(df))
        hours = ser.apply(lambda x: parse_hours_nonneg(x, col_name))
        return hours.apply(to_min_from_hours)

    df["max_week_min"] = get_hours_with_default(
        "max_week_hours_h", defaults.get("max_week_hours_h", 60)
    )
    df["max_month_extra_min"] = get_hours_with_default(
        "max_month_extra_h", defaults.get("max_month_extra_h", 40)
    )

    def get_int_with_default(col_name: str, default_val: int) -> pd.Series:
        """Estrae colonna intera con fallback a default, validando non-negativi."""
        if col_name in df.columns:
            ser = df[col_name].astype(str).str.strip()
            ser = ser.where(ser != "", other=str(default_val))
        else:
            ser = pd.Series([str(default_val)] * len(df))
        try:
            vals = ser.astype(float)
        except ValueError:
            raise LoaderError(f"employees.csv: valore non numerico in '{col_name}'")
        if (vals < 0).any():
            raise LoaderError(
                f"employees.csv: valore negativo non ammesso in '{col_name}'"
            )
        return vals.round().astype(int)

    df["max_nights_week"] = get_int_with_default(
        "max_nights_week", int(defaults.get("max_nights_week", 3))
    )
    df["max_nights_month"] = get_int_with_default(
        "max_nights_month", int(defaults.get("max_nights_month", 8))
    )

    df["saturday_count_ytd"] = get_int_with_default("saturday_count_ytd", 0)
    df["sunday_count_ytd"] = get_int_with_default("sunday_count_ytd", 0)
    df["holiday_count_ytd"] = get_int_with_default("holiday_count_ytd", 0)

    return df[
        [
            "employee_id",
            "nome",
            "ruolo",
            "reparto",
            "dovuto_min",
            "saldo_init_min",
            "max_week_min",
            "max_month_extra_min",
            "max_nights_week",
            "max_nights_month",
            "saturday_count_ytd",
            "sunday_count_ytd",
            "holiday_count_ytd",
        ]
    ]


def load_role_dept_pools(
    path: str, defaults: dict[str, Any], employees_df: pd.DataFrame
) -> pd.DataFrame:
    """Carica pool di reparti per ogni ruolo (file opzionale)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ruolo", "pool_id", "reparto"])

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"ruolo", "pool_id", "reparto"}, "role_dept_pools.csv")

    for col in ["ruolo", "pool_id", "reparto"]:
        df[col] = df[col].astype(str).str.strip()

    if (df[["ruolo", "pool_id", "reparto"]] == "").any().any():
        bad_rows = df.loc[(df[["ruolo", "pool_id", "reparto"]] == "").any(axis=1)]
        raise LoaderError(
            "role_dept_pools.csv: valori vuoti non ammessi nelle colonne ruolo/pool_id/reparto. "
            f"Righe interessate:\n{bad_rows}"
        )

    allowed_roles = set(
        _resolve_allowed_roles(defaults, fallback_roles=employees_df["ruolo"].unique())
    )
    allowed_departments = set(_resolve_allowed_departments(defaults))

    bad_roles = sorted(set(df["ruolo"].unique()) - allowed_roles)
    if bad_roles:
        raise LoaderError(
            "role_dept_pools.csv: ruoli non ammessi rispetto alla config: "
            f"{bad_roles}"
        )

    bad_departments = sorted(set(df["reparto"].unique()) - allowed_departments)
    if bad_departments:
        raise LoaderError(
            "role_dept_pools.csv: reparti non ammessi rispetto alla config: "
            f"{bad_departments}"
        )

    if df.duplicated(subset=["ruolo", "pool_id", "reparto"]).any():
        dup = df[
            df.duplicated(subset=["ruolo", "pool_id", "reparto"], keep=False)
        ].sort_values(["ruolo", "pool_id", "reparto"])
        raise LoaderError(
            "role_dept_pools.csv: duplicati non ammessi su (ruolo, pool_id, reparto):\n"
            f"{dup}"
        )

    return df.sort_values(["ruolo", "pool_id", "reparto"]).reset_index(drop=True)


def build_department_compatibility(
    defaults: dict[str, Any], pools_df: pd.DataFrame, employees_df: pd.DataFrame
) -> pd.DataFrame:
    """Costruisce matrice compatibilità tra reparti basata sui pool."""
    allowed_roles = _resolve_allowed_roles(
        defaults, fallback_roles=employees_df["ruolo"].unique()
    )
    allowed_departments = _resolve_allowed_departments(defaults)

    combos: list[tuple[str, str, str]] = []
    seen = set()

    def add(role: str, dept_home: str, dept_target: str) -> None:
        """Aggiunge combinazione ruolo-reparto se non già presente."""
        key = (role, dept_home, dept_target)
        if key not in seen:
            seen.add(key)
            combos.append(key)

    for role in allowed_roles:
        for dept in allowed_departments:
            add(role, dept, dept)

    if not pools_df.empty:
        pools_by_role = {role: grp for role, grp in pools_df.groupby("ruolo")}
        for role in allowed_roles:
            role_df = pools_by_role.get(role)
            if role_df is None:
                continue
            for _, pool_df in role_df.groupby("pool_id"):
                pool_departments = list(dict.fromkeys(pool_df["reparto"].tolist()))
                for dept_home in pool_departments:
                    for dept_target in pool_departments:
                        add(role, dept_home, dept_target)

    compat_df = pd.DataFrame(
        combos, columns=["ruolo", "reparto_home", "reparto_target"]
    )
    if not compat_df.empty:
        compat_df = compat_df.sort_values(
            ["ruolo", "reparto_home", "reparto_target"]
        ).reset_index(drop=True)
    return compat_df
