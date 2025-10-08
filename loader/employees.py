from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd

from .absences import get_absence_hours_from_config
from .utils import (
    LoaderError,
    _ensure_cols,
    _resolve_allowed_departments,
    _resolve_allowed_roles,
)


def load_employees(
    path: str,
    defaults: dict[str, Any],
    role_defaults: dict[str, Any],
    weeks_in_horizon: int,
    days_in_horizon: int,
) -> pd.DataFrame:
    """Carica e valida dati dipendenti con ore, limiti e contatori."""
    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(
        df,
        {
            "employee_id",
            "nome",
            "ruolo",
            "reparto_id",
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

    df["ruolo"] = df["ruolo"].astype(str).str.strip()
    # Esponiamo anche la colonna standardizzata `role` per compatibilita' con le
    # pipeline di pre-processing.
    df["role"] = df["ruolo"]

    allowed_roles = _resolve_allowed_roles(defaults, fallback_roles=df["ruolo"].unique())

    bad_roles = sorted(set(df["ruolo"].unique()) - set(allowed_roles))
    if bad_roles:
        raise LoaderError(f"employees.csv: ruoli non ammessi rispetto alla config: {bad_roles}")

    allowed_departments = _resolve_allowed_departments(defaults)
    df["reparto_id"] = df["reparto_id"].astype(str).str.strip()
    if (df["reparto_id"] == "").any():
        bad = df.loc[df["reparto_id"] == "", ["employee_id", "nome", "ruolo"]]
        raise LoaderError(
            "employees.csv: la colonna 'reparto_id' è obbligatoria e non può essere vuota. "
            f"Righe interessate:\n{bad}"
        )

    bad_departments = sorted(
        set(df["reparto_id"].unique()) - set(allowed_departments)
    )
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

    if weeks_in_horizon <= 0:
        raise LoaderError(
            "employees.csv: impossibile calcolare i limiti settimanali senza settimane nell'orizzonte"
        )
    if days_in_horizon <= 0:
        raise LoaderError(
            "employees.csv: impossibile calcolare i limiti settimanali senza giorni nell'orizzonte"
        )

    contract_hours = df["dovuto_min"].astype(float) / 60.0

    # Limite mensile massimo (hard constraint)
    month_cap_hours: list[float] = []
    month_col_present = "max_month_hours_h" in df.columns
    for idx, base_hours in enumerate(contract_hours):
        override = ""
        if month_col_present:
            override = str(df.iloc[idx]["max_month_hours_h"]).strip()
        if override:
            cap_hours = parse_hours_nonneg(override, "max_month_hours_h")
        else:
            cap_hours = base_hours * 1.25
        if cap_hours + 1e-9 < base_hours:
            raise LoaderError(
                "employees.csv: max_month_hours_h deve essere ≥ ore_dovute_mese_h per employee_id "
                f"{df.iloc[idx]['employee_id']}"
            )
        month_cap_hours.append(cap_hours)
    df["max_month_min"] = [to_min_from_hours(v) for v in month_cap_hours]

    # Limite settimanale massimo (hard constraint)
    week_cap_hours: list[float] = []
    week_col_present = "max_week_hours_h" in df.columns
    for idx, base_hours in enumerate(contract_hours):
        # Ripartiamo le ore contrattuali su una settimana "media" del mese
        # (ore_mese / giorni_orizzonte * 7). Il valore viene poi moltiplicato
        # per 1.5 così da consentire straordinari senza concentrare tutto in
        # un'unica settimana, applicando lo stesso cap anche alle settimane
        # parziali all'inizio o alla fine dell'orizzonte.
        weekly_theoretical = (
            base_hours / days_in_horizon * 7.0 if days_in_horizon else 0.0
        )
        override = ""
        if week_col_present:
            override = str(df.iloc[idx]["max_week_hours_h"]).strip()
        if override:
            week_hours = parse_hours_nonneg(override, "max_week_hours_h")
        else:
            week_hours = weekly_theoretical * 1.5
        if week_hours + 1e-9 < weekly_theoretical:
            raise LoaderError(
                "employees.csv: max_week_hours_h deve essere ≥ ore settimanali teoriche per employee_id "
                f"{df.iloc[idx]['employee_id']}"
            )
        week_cap_hours.append(week_hours)
    df["max_week_min"] = [to_min_from_hours(v) for v in week_cap_hours]

    def _coerce_bool(value: Any, label: str, allow_empty: bool = False) -> bool | None:
        """Converte un valore generico in bool, opzionalmente accettando vuoti."""
        if value is None:
            if allow_empty:
                return None
            raise LoaderError(f"{label} mancante")
        if isinstance(value, str):
            s = value.strip().lower()
            if s == "":
                if allow_empty:
                    return None
                raise LoaderError(f"{label} mancante")
            if s in {"true", "1", "yes", "y", "si", "s"}:
                return True
            if s in {"false", "0", "no", "n"}:
                return False
        elif isinstance(value, (int, float)):
            if value in {0, 1}:
                return bool(int(value))
        elif isinstance(value, bool):
            return value
        raise LoaderError(f"{label}: valore booleano non riconosciuto ({value!r})")

    def _coerce_nonneg_int(
        value: Any, label: str, allow_empty: bool = False
    ) -> int | None:
        """Converte un valore in intero non negativo, opzionalmente accettando vuoti."""
        if value is None:
            if allow_empty:
                return None
            raise LoaderError(f"{label} mancante")
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                if allow_empty:
                    return None
                raise LoaderError(f"{label} mancante")
            try:
                num = float(s)
            except ValueError as exc:
                raise LoaderError(f"{label}: valore non numerico ({value!r})") from exc
        else:
            try:
                num = float(value)
            except (TypeError, ValueError) as exc:
                raise LoaderError(f"{label}: valore non numerico ({value!r})") from exc
        if num < 0:
            raise LoaderError(f"{label}: valore negativo non ammesso ({num})")
        return int(round(num))

    night_defaults = defaults.get("night", {}) or {}
    global_can_work = _coerce_bool(
        night_defaults.get("can_work_night"),
        "config: defaults.night.can_work_night",
    )
    global_max_week = _coerce_nonneg_int(
        night_defaults.get("max_per_week"),
        "config: defaults.night.max_per_week",
    )
    global_max_month = _coerce_nonneg_int(
        night_defaults.get("max_per_month"),
        "config: defaults.night.max_per_month",
    )

    can_col_present = "can_work_night" in df.columns
    week_col_present = "max_nights_week" in df.columns
    month_col_present = "max_nights_month" in df.columns

    resolved_can: list[bool] = []
    resolved_week: list[int] = []
    resolved_month: list[int] = []

    for idx, row in df.iterrows():
        role = str(row["ruolo"]).strip()
        role_cfg = role_defaults.get(role, {}) or {}

        role_can = _coerce_bool(
            role_cfg.get("can_work_night"),
            f"config: roles.{role}.can_work_night",
            allow_empty=True,
        )
        if role_can is None:
            role_can = global_can_work

        role_night_cfg = role_cfg.get("night", {}) or {}
        role_week = _coerce_nonneg_int(
            role_night_cfg.get("max_per_week"),
            f"config: roles.{role}.night.max_per_week",
            allow_empty=True,
        )
        if role_week is None:
            role_week = global_max_week
        role_month = _coerce_nonneg_int(
            role_night_cfg.get("max_per_month"),
            f"config: roles.{role}.night.max_per_month",
            allow_empty=True,
        )
        if role_month is None:
            role_month = global_max_month

        emp_can = role_can
        if can_col_present:
            emp_can_override = _coerce_bool(
                row["can_work_night"],
                "employees.csv: can_work_night",
                allow_empty=True,
            )
            if emp_can_override is not None:
                emp_can = emp_can_override

        emp_week = role_week
        if week_col_present:
            emp_week_override = _coerce_nonneg_int(
                row["max_nights_week"],
                "employees.csv: max_nights_week",
                allow_empty=True,
            )
            if emp_week_override is not None:
                emp_week = emp_week_override

        emp_month = role_month
        if month_col_present:
            emp_month_override = _coerce_nonneg_int(
                row["max_nights_month"],
                "employees.csv: max_nights_month",
                allow_empty=True,
            )
            if emp_month_override is not None:
                emp_month = emp_month_override

        resolved_can.append(bool(emp_can))
        resolved_week.append(int(emp_week))
        resolved_month.append(int(emp_month))

    df["can_work_night"] = resolved_can
    df["max_nights_week"] = resolved_week
    df["max_nights_month"] = resolved_month

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

    df["saturday_count_ytd"] = get_int_with_default("saturday_count_ytd", 0)
    df["sunday_count_ytd"] = get_int_with_default("sunday_count_ytd", 0)
    df["holiday_count_ytd"] = get_int_with_default("holiday_count_ytd", 0)

    ordered_cols = [
        "employee_id",
        "nome",
        "ruolo",
        "role",
        "reparto_id",
        "ore_dovute_mese_h",
        "dovuto_min",
        "saldo_init_min",
        "max_month_min",
        "max_week_min",
        "can_work_night",
        "max_nights_week",
        "max_nights_month",
        "saturday_count_ytd",
        "sunday_count_ytd",
        "holiday_count_ytd",
    ]
    if "reparto_label" in df.columns:
        df["reparto_label"] = df["reparto_label"].astype(str).str.strip()
        ordered_cols.insert(ordered_cols.index("reparto_id") + 1, "reparto_label")

    optional_tail_cols = [
        "cross_max_shifts_week",
        "cross_max_shifts_month",
        "cross_penalty_weight",
    ]
    for col in optional_tail_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    return df[ordered_cols]


def load_role_dept_pools(
    path: str, defaults: dict[str, Any], employees_df: pd.DataFrame
) -> pd.DataFrame:
    """Carica pool di reparti per ogni ruolo (file opzionale)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ruolo", "pool_id", "reparto_id"])

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"ruolo", "pool_id", "reparto_id"}, "role_dept_pools.csv")

    for col in ["ruolo", "pool_id", "reparto_id"]:
        df[col] = df[col].astype(str).str.strip()

    if (df[["ruolo", "pool_id", "reparto_id"]] == "").any().any():
        bad_rows = df.loc[
            (df[["ruolo", "pool_id", "reparto_id"]] == "").any(axis=1)
        ]
        raise LoaderError(
            "role_dept_pools.csv: valori vuoti non ammessi nelle colonne ruolo/pool_id/reparto_id. "
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

    bad_departments = sorted(set(df["reparto_id"].unique()) - allowed_departments)
    if bad_departments:
        raise LoaderError(
            "role_dept_pools.csv: reparti non ammessi rispetto alla config: "
            f"{bad_departments}"
        )

    if df.duplicated(subset=["ruolo", "pool_id", "reparto_id"]).any():
        dup = df[
            df.duplicated(subset=["ruolo", "pool_id", "reparto_id"], keep=False)
        ].sort_values(["ruolo", "pool_id", "reparto_id"])
        raise LoaderError(
            "role_dept_pools.csv: duplicati non ammessi su (ruolo, pool_id, reparto_id):\n"
            f"{dup}"
        )

    return df.sort_values(["ruolo", "pool_id", "reparto_id"]).reset_index(drop=True)


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
                pool_departments = list(dict.fromkeys(pool_df["reparto_id"].tolist()))
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


def resolve_fulltime_baseline(config: dict, role: str | None) -> float:
    """Resolve baseline monthly full-time hours for the given role.

    Args:
        config: Configuration dictionary containing the ``payroll`` section.
        role: Role identifier used to look up role-specific baselines. Can be ``None``.

    Returns:
        The baseline full-time monthly hours as a float.

    Raises:
        ValueError: If the configuration is missing, malformed, or does not
            provide a baseline for the requested role/default. Also raised when the
            resolved baseline is not a positive number.
    """

    payroll_cfg = config.get("payroll")
    if payroll_cfg is None:
        payroll_cfg = {}
    elif not isinstance(payroll_cfg, dict):
        raise ValueError("config['payroll'] deve essere un dizionario valido")

    raw_by_role = payroll_cfg.get("fulltime_hours_mensili_by_role")
    if raw_by_role is None:
        raw_by_role = {}
    elif not isinstance(raw_by_role, dict):
        raise ValueError(
            "config['payroll']['fulltime_hours_mensili_by_role'] deve essere un dizionario"
        )

    defaults_cfg = config.get("defaults")
    contract_by_role_cfg: dict[str, object] = {}
    if isinstance(defaults_cfg, dict):
        contract_cfg = defaults_cfg.get("contract_hours_by_role_h")
        if isinstance(contract_cfg, dict):
            contract_by_role_cfg = {
                str(k).strip(): contract_cfg[k]
                for k in contract_cfg
                if str(k).strip()
            }

    role_key = role.strip() if isinstance(role, str) else None

    candidate = None
    candidate_source = None

    if role_key:
        lookup_key = role_key
        if lookup_key in raw_by_role:
            candidate = raw_by_role[lookup_key]
            candidate_source = f"payroll.fulltime_hours_mensili_by_role[{lookup_key}]"
        elif lookup_key in contract_by_role_cfg:
            candidate = contract_by_role_cfg[lookup_key]
            candidate_source = f"defaults.contract_hours_by_role_h[{lookup_key}]"

    if candidate is None:
        payroll_default = payroll_cfg.get("fulltime_hours_mensili_default")
        if payroll_default is not None:
            candidate = payroll_default
            candidate_source = "payroll.fulltime_hours_mensili_default"

    if candidate is None and not role_key and contract_by_role_cfg:
        first_key = next(iter(contract_by_role_cfg))
        candidate = contract_by_role_cfg[first_key]
        candidate_source = f"defaults.contract_hours_by_role_h[{first_key}]"

    if candidate is None:
        if role_key:
            raise ValueError(
                "Baseline full-time mancante per ruolo "
                f"'{role_key}': specificare payroll.fulltime_hours_mensili_by_role "
                "oppure defaults.contract_hours_by_role_h."
            )
        raise ValueError(
            "config['payroll'] deve specificare 'fulltime_hours_mensili_default' "
            "oppure defaults.contract_hours_by_role_h."
        )

    try:
        baseline = float(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valore non numerico per {candidate_source}: {candidate!r}") from exc

    if baseline <= 0:
        raise ValueError(f"Il baseline full-time deve essere positivo ({baseline})")

    return baseline


def enrich_employees_with_fte(employees: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return a copy of employees with FTE-related columns appended.

    The function expects an ``employees`` DataFrame containing at least the columns
    ``employee_id``, ``ore_dovute_mese_h`` and either ``role`` or ``ruolo``. It adds
    the columns ``fte`` and ``fte_weight`` at the end of the DataFrame without
    mutating the original input.

    Args:
        employees: Input DataFrame describing employees.
        config: Configuration dictionary containing contract hours and payroll
            information.

    Returns:
        A copy of ``employees`` with ``fte`` and ``fte_weight`` columns appended.

    Raises:
        ValueError: When required information is missing or invalid.
    """

    if "ore_dovute_mese_h" not in employees.columns:
        raise ValueError("La colonna 'ore_dovute_mese_h' è richiesta")
    if "employee_id" not in employees.columns:
        raise ValueError("La colonna 'employee_id' è richiesta")

    role_column = None
    for candidate in ("role", "ruolo"):
        if candidate in employees.columns:
            role_column = candidate
            break
    if role_column is None:
        raise ValueError("È necessaria la colonna 'role' o 'ruolo' nel DataFrame")

    df = employees.copy(deep=True)

    contract_by_role = config.get("contract_hours_by_role_h")
    if contract_by_role is None:
        contract_by_role = {}
    elif not isinstance(contract_by_role, dict):
        raise ValueError("config['contract_hours_by_role_h'] deve essere un dizionario")

    ore_raw = df["ore_dovute_mese_h"]
    ore_numeric = pd.to_numeric(ore_raw, errors="coerce")
    non_empty_mask = (~ore_raw.isna()) & (ore_raw.astype(str).str.strip() != "")
    invalid_mask = non_empty_mask & ore_numeric.isna()
    if invalid_mask.any():
        bad_ids = df.loc[invalid_mask, "employee_id"].tolist()
        raise ValueError(
            f"Valori non numerici in 'ore_dovute_mese_h' per employee_id: {bad_ids}"
        )

    fte_values: list[float] = []

    for idx, row in df.iterrows():
        role_value = row[role_column]
        if pd.isna(role_value):
            raise ValueError(
                f"Ruolo mancante per employee_id {row['employee_id']}: impossibile calcolare FTE"
            )
        role_str = str(role_value).strip()
        if not role_str:
            raise ValueError(
                f"Ruolo mancante per employee_id {row['employee_id']}: impossibile calcolare FTE"
            )

        due_hours = ore_numeric.loc[idx]
        if pd.isna(due_hours):
            if role_str not in contract_by_role:
                raise ValueError(
                    "ore_dovute_mese_h mancante per employee_id "
                    f"{row['employee_id']} e nessun default in config['contract_hours_by_role_h']"
                )
            try:
                due_hours = float(contract_by_role[role_str])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Valore non numerico in contract_hours_by_role_h per ruolo "
                    f"'{role_str}'"
                ) from exc
        if pd.isna(due_hours):
            raise ValueError(
                "Valore non determinabile di 'ore_dovute_mese_h' per employee_id "
                f"{row['employee_id']}"
            )

        if due_hours <= 0:
            warnings.warn(
                "ore_dovute_mese_h non positiva per employee_id "
                f"{row['employee_id']}: {due_hours}",
                stacklevel=2,
            )

        baseline = resolve_fulltime_baseline(config, role_str)
        fte = due_hours / baseline
        fte_values.append(fte)

    fte_series = pd.Series(fte_values, index=df.index, dtype=float)
    df["fte"] = fte_series
    df["fte_weight"] = fte_series

    return df


def get_absence_hours(config: dict) -> float:
    """Compatibilità retroattiva: delega a ``get_absence_hours_from_config``."""

    return get_absence_hours_from_config(config)
