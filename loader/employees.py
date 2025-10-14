from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import pandas as pd

from .absences import get_absence_hours_from_config
from .utils import (
    LoaderError,
    _resolve_allowed_departments,
    _resolve_allowed_roles,
)


logger = logging.getLogger(__name__)


def _contract_hours_by_role(defaults_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Normalizza la mappatura ruolo→ore contrattuali dai defaults."""

    if defaults_cfg is None:
        defaults_cfg = {}
    if not isinstance(defaults_cfg, dict):
        raise LoaderError("config['defaults'] deve essere un dizionario valido")

    contract_cfg = defaults_cfg.get("contract_hours_by_role_h")
    if contract_cfg is None:
        return {}
    if not isinstance(contract_cfg, dict):
        raise LoaderError(
            "config['defaults']['contract_hours_by_role_h'] deve essere un dizionario"
        )

    normalized: dict[str, Any] = {}
    for raw_key, value in contract_cfg.items():
        key = str(raw_key).strip()
        if key:
            normalized[key] = value

    return normalized


def _require_contract_hours(
    contract_by_role: dict[str, Any],
    role: str,
    *,
    error_cls: type[Exception] = LoaderError,
) -> Any:
    """Restituisce le ore contrattuali per ``role`` oppure solleva errore."""

    role_key = str(role).strip()
    display_role = role_key or str(role)
    if role_key not in contract_by_role:
        raise error_cls(
            f"contract_hours_by_role_h non definito per il ruolo {display_role}"
        )
    return contract_by_role[role_key]


def load_employees(
    path: str,
    defaults: dict[str, Any],
    role_defaults: dict[str, Any],
    weeks_in_horizon: int,
    days_in_horizon: int,
) -> pd.DataFrame:
    """Carica e valida dati dipendenti con ore, limiti e contatori."""
    df = pd.read_csv(path, dtype=str).fillna("")

    base_required = {
        "employee_id",
        "nome",
        "reparto_id",
        "ore_dovute_mese_h",
        "saldo_prog_iniziale_h",
    }
    missing_base = base_required - set(df.columns)
    if missing_base:
        raise LoaderError(
            "employees.csv: colonne mancanti: " + ", ".join(sorted(missing_base))
        )

    role_columns = [col for col in ("role", "ruolo") if col in df.columns]
    if not role_columns:
        raise LoaderError(
            "employees.csv: è richiesta la colonna 'role' (o il legacy 'ruolo')"
        )
    if len(role_columns) == 2:
        legacy = df["ruolo"].astype(str).str.strip()
        modern = df["role"].astype(str).str.strip()
        mismatch = legacy.ne(modern) & ~(legacy.eq("") & modern.eq(""))
        if mismatch.any():
            rows = ", ".join(str(i + 2) for i in mismatch[mismatch].index.tolist())
            raise LoaderError(
                "employees.csv: colonne 'role' e 'ruolo' presenti ma con valori diversi alle "
                f"righe [{rows}]"
            )
    role_source = role_columns[0]
    if role_source != "role":
        df = df.rename(columns={role_source: "role"})
    if "ruolo" in df.columns:
        df = df.drop(columns=["ruolo"])

    if df["employee_id"].duplicated().any():
        dups = df[df["employee_id"].duplicated(keep=False)].sort_values("employee_id")
        raise LoaderError(
            f"employees.csv: employee_id duplicati:\n{dups[['employee_id','nome','role']]}"
        )

    df["role"] = df["role"].astype(str).str.strip()

    allowed_roles = _resolve_allowed_roles(defaults, fallback_roles=df["role"].unique())

    bad_roles = sorted(set(df["role"].unique()) - set(allowed_roles))
    if bad_roles:
        raise LoaderError(f"employees.csv: ruoli non ammessi rispetto alla config: {bad_roles}")

    allowed_departments = _resolve_allowed_departments(defaults)
    df["reparto_id"] = df["reparto_id"].astype(str).str.strip()
    if (df["reparto_id"] == "").any():
        bad = df.loc[df["reparto_id"] == "", ["employee_id", "nome", "role"]]
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

    global_weekly_rest_min_days = defaults.get("weekly_rest_min_days", 1)
    if isinstance(global_weekly_rest_min_days, bool) or not isinstance(
        global_weekly_rest_min_days, int
    ):
        raise LoaderError(
            "config: defaults.weekly_rest_min_days deve essere un intero ≥ 0 "
            f"(trovato: {global_weekly_rest_min_days!r})"
        )
    if global_weekly_rest_min_days < 0:
        raise LoaderError(
            "config: defaults.weekly_rest_min_days deve essere un intero ≥ 0 "
            f"(trovato: {global_weekly_rest_min_days!r})"
        )

    def _parse_nonneg_int_config(value: Any, label: str, *, default: int = 0) -> int:
        """Converte un valore generico in intero ≥ 0 per configurazioni."""

        if value in (None, ""):
            return int(default)
        if isinstance(value, bool):
            raise LoaderError(f"{label} deve essere un intero ≥ 0 (trovato: {value!r})")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            if abs(value - round(value)) > 1e-9:
                raise LoaderError(
                    f"{label} deve essere un intero ≥ 0 (trovato: {value!r})"
                )
            parsed = int(round(value))
        elif isinstance(value, str):
            s = value.strip()
            if s == "":
                return int(default)
            try:
                parsed = int(s)
            except ValueError as exc:
                raise LoaderError(
                    f"{label} deve essere un intero ≥ 0 (trovato: {value!r})"
                ) from exc
        else:
            raise LoaderError(f"{label} deve essere un intero ≥ 0 (trovato: {value!r})")
        if parsed < 0:
            raise LoaderError(f"{label} deve essere un intero ≥ 0 (trovato: {value!r})")
        return parsed

    balance_defaults_cfg = defaults.get("balance", {}) or {}
    if balance_defaults_cfg and not isinstance(balance_defaults_cfg, dict):
        raise LoaderError("config: defaults.balance deve essere un dizionario")

    global_max_balance_delta = _parse_nonneg_int_config(
        balance_defaults_cfg.get("max_balance_delta_month_h", 0),
        "config: defaults.balance.max_balance_delta_month_h",
        default=0,
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

    absences_cfg = defaults.get("absences", {}) or {}
    if not isinstance(absences_cfg, dict):  # pragma: no cover - validato in config
        absences_cfg = {}
    absence_role_hours = {
        str(role).strip().casefold(): float(value)
        for role, value in (absences_cfg.get("full_day_hours_by_role_h") or {}).items()
        if str(role).strip()
    }
    absence_fallback = float(absences_cfg.get("fallback_contract_daily_avg_h", 0.0))

    contract_by_role = _contract_hours_by_role(defaults)

    dovuto_min = []
    for _, r in df.iterrows():
        raw = str(r["ore_dovute_mese_h"]).strip()
        role = str(r["role"]).strip()
        if raw != "":
            hours = parse_hours_nonneg(raw, "ore_dovute_mese_h")
        else:
            hours_default = _require_contract_hours(
                contract_by_role, role, error_cls=LoaderError
            )
            hours = parse_hours_nonneg(hours_default, f"contract_hours_by_role_h[{role}]")
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

    absence_override_col_present = "absence_full_day_hours_h" in df.columns
    employee_idx = df.columns.get_loc("employee_id")
    role_idx = df.columns.get_loc("role")
    absence_idx = df.columns.get_loc("absence_full_day_hours_h") if absence_override_col_present else None
    absence_overrides: list[float | None] = []
    absence_effective: list[float] = []

    for row in df.itertuples(index=False):
        employee_id = str(row[employee_idx]).strip()
        role_value = str(row[role_idx]).strip()
        role_lookup = role_value.casefold()

        override_value: float | None = None
        if absence_override_col_present:
            raw_value = str(row[absence_idx]).strip() if absence_idx is not None else ""
            if raw_value:
                try:
                    override_value = float(raw_value)
                except ValueError as exc:
                    raise LoaderError(
                        "Valore non valido per absence_full_day_hours_h per dipendente "
                        f"{employee_id}: {row[absence_idx]}"
                    ) from exc
                if override_value != override_value or override_value < 0:
                    raise LoaderError(
                        "Valore non valido per absence_full_day_hours_h per dipendente "
                        f"{employee_id}: {row[absence_idx]}"
                    )
                logger.info(
                    "employees.csv: absence_full_day_hours_h override per dipendente %s: %s",
                    employee_id,
                    override_value,
                )

        if override_value is not None:
            effective_hours = float(override_value)
        else:
            if role_lookup in absence_role_hours:
                effective_hours = float(absence_role_hours[role_lookup])
            else:
                effective_hours = float(absence_fallback)
                logger.warning(
                    "employees.csv: absence_full_day_hours_h non definito per ruolo %s (dipendente %s): uso fallback %s",
                    role_value or "<vuoto>",
                    employee_id,
                    effective_hours,
                )

        absence_overrides.append(override_value)
        absence_effective.append(float(effective_hours))

    if not absence_overrides:
        absence_overrides = [None] * len(df)
        absence_effective = [absence_fallback] * len(df)

    df["absence_full_day_hours_h"] = absence_overrides
    df["absence_full_day_hours_effective_h"] = absence_effective

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
        # per 1.4 (+40%) così da consentire straordinari senza concentrare tutto in
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
            week_hours = weekly_theoretical * 1.4
        if week_hours + 1e-9 < weekly_theoretical:
            raise LoaderError(
                "employees.csv: max_week_hours_h deve essere ≥ ore settimanali teoriche per employee_id "
                f"{df.iloc[idx]['employee_id']}"
            )
        week_cap_hours.append(week_hours)
    df["max_week_min"] = [to_min_from_hours(v) for v in week_cap_hours]

    balance_col_present = "max_balance_delta_month_h" in df.columns
    balance_deltas: list[int] = []

    for _, row in df.iterrows():
        employee_id = str(row["employee_id"]).strip()
        value = global_max_balance_delta
        if balance_col_present:
            original_value = row["max_balance_delta_month_h"]
            raw_value = str(original_value).strip()
            if raw_value == "":
                logger.warning(
                    "employees.csv: max_balance_delta_month_h vuoto per dipendente %s: uso default %s",
                    employee_id,
                    global_max_balance_delta,
                )
                logger.info(
                    "employees.csv: max_balance_delta_month_h default applicato per dipendente %s: %s",
                    employee_id,
                    global_max_balance_delta,
                )
            else:
                try:
                    parsed = int(raw_value)
                except ValueError as exc:
                    logger.warning(
                        "employees.csv: max_balance_delta_month_h non numerico per dipendente %s: %r",
                        employee_id,
                        original_value,
                    )
                    raise LoaderError(
                        "Valore non valido per max_balance_delta_month_h per dipendente "
                        f"{employee_id}: {original_value}"
                    ) from exc
                if parsed < 0:
                    logger.warning(
                        "employees.csv: max_balance_delta_month_h negativo per dipendente %s: %r",
                        employee_id,
                        original_value,
                    )
                    raise LoaderError(
                        "Valore non valido per max_balance_delta_month_h per dipendente "
                        f"{employee_id}: {original_value}"
                    )
                value = parsed
                logger.info(
                    "employees.csv: max_balance_delta_month_h override per dipendente %s: %s",
                    employee_id,
                    value,
                )
        balance_deltas.append(int(value))

    if not balance_col_present and balance_deltas:
        logger.info(
            "employees.csv: max_balance_delta_month_h default globale applicato a tutti i dipendenti: %s",
            global_max_balance_delta,
        )

    df["max_balance_delta_month_h"] = balance_deltas

    weekly_rest_values: list[int] = []
    weekly_rest_col_present = "weekly_rest_min_days" in df.columns

    for _, row in df.iterrows():
        employee_id = str(row["employee_id"]).strip()
        original_value = row["weekly_rest_min_days"] if weekly_rest_col_present else ""
        raw_value = str(original_value).strip() if weekly_rest_col_present else ""

        if raw_value == "":
            weekly_rest_values.append(global_weekly_rest_min_days)
            logger.info(
                "employees.csv: weekly_rest_min_days default applicato per dipendente %s: %s",
                employee_id,
                global_weekly_rest_min_days,
            )
            continue

        try:
            parsed = int(raw_value)
        except ValueError as exc:
            raise LoaderError(
                "Valore non valido per weekly_rest_min_days per dipendente "
                f"{employee_id}: {original_value}"
            ) from exc

        if parsed < 0:
            raise LoaderError(
                "Valore non valido per weekly_rest_min_days per dipendente "
                f"{employee_id}: {original_value}"
            )

        weekly_rest_values.append(parsed)
        logger.info(
            "employees.csv: weekly_rest_min_days override per dipendente %s: %s",
            employee_id,
            parsed,
        )

    if len(weekly_rest_values) != len(df):
        weekly_rest_values = [global_weekly_rest_min_days] * len(df)
    df["weekly_rest_min_days"] = weekly_rest_values

    rest11h_defaults_cfg = defaults.get("rest11h", {}) or {}
    try:
        global_rest11h_monthly = int(rest11h_defaults_cfg.get("max_monthly_exceptions", 0))
        global_rest11h_consecutive = int(
            rest11h_defaults_cfg.get("max_consecutive_exceptions", 0)
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - già validato in config
        raise LoaderError(
            "config: defaults.rest11h deve contenere interi non negativi"
        ) from exc

    if global_rest11h_monthly < 0 or global_rest11h_consecutive < 0:
        raise LoaderError(
            "config: defaults.rest11h deve contenere interi non negativi"
        )

    monthly_col_present = "rest11h_max_monthly_exceptions" in df.columns
    consecutive_col_present = "rest11h_max_consecutive_exceptions" in df.columns

    rest11h_monthly_values: list[int] = []
    rest11h_consecutive_values: list[int] = []

    for _, row in df.iterrows():
        employee_id = str(row["employee_id"]).strip()

        monthly_value = global_rest11h_monthly
        if monthly_col_present:
            original_value = row["rest11h_max_monthly_exceptions"]
            raw_value = str(original_value).strip()
            if raw_value == "":
                logger.warning(
                    "employees.csv: rest11h_max_monthly_exceptions vuoto per dipendente %s: "
                    "uso default %s",
                    employee_id,
                    global_rest11h_monthly,
                )
                logger.info(
                    "employees.csv: rest11h_max_monthly_exceptions default applicato per dipendente %s: %s",
                    employee_id,
                    global_rest11h_monthly,
                )
            else:
                try:
                    parsed_monthly = int(raw_value)
                except ValueError as exc:
                    logger.warning(
                        "employees.csv: rest11h_max_monthly_exceptions non numerico per dipendente %s: %r",
                        employee_id,
                        original_value,
                    )
                    raise LoaderError(
                        "Valore non valido per rest11h_max_monthly_exceptions per dipendente "
                        f"{employee_id}: {original_value}"
                    ) from exc
                if parsed_monthly < 0:
                    logger.warning(
                        "employees.csv: rest11h_max_monthly_exceptions negativo per dipendente %s: %r",
                        employee_id,
                        original_value,
                    )
                    raise LoaderError(
                        "Valore non valido per rest11h_max_monthly_exceptions per dipendente "
                        f"{employee_id}: {original_value}"
                    )
                monthly_value = parsed_monthly
                logger.info(
                    "employees.csv: rest11h_max_monthly_exceptions override per dipendente %s: %s",
                    employee_id,
                    monthly_value,
                )
        else:
            logger.info(
                "employees.csv: rest11h_max_monthly_exceptions default applicato per dipendente %s: %s",
                employee_id,
                global_rest11h_monthly,
            )

        consecutive_value = global_rest11h_consecutive
        if consecutive_col_present:
            original_value = row["rest11h_max_consecutive_exceptions"]
            raw_value = str(original_value).strip()
            if raw_value == "":
                logger.warning(
                    "employees.csv: rest11h_max_consecutive_exceptions vuoto per dipendente %s: "
                    "uso default %s",
                    employee_id,
                    global_rest11h_consecutive,
                )
                logger.info(
                    "employees.csv: rest11h_max_consecutive_exceptions default applicato per dipendente %s: %s",
                    employee_id,
                    global_rest11h_consecutive,
                )
            else:
                try:
                    parsed_consecutive = int(raw_value)
                except ValueError as exc:
                    logger.warning(
                        "employees.csv: rest11h_max_consecutive_exceptions non numerico per dipendente %s: %r",
                        employee_id,
                        original_value,
                    )
                    raise LoaderError(
                        "Valore non valido per rest11h_max_consecutive_exceptions per dipendente "
                        f"{employee_id}: {original_value}"
                    ) from exc
                if parsed_consecutive < 0:
                    logger.warning(
                        "employees.csv: rest11h_max_consecutive_exceptions negativo per dipendente %s: %r",
                        employee_id,
                        original_value,
                    )
                    raise LoaderError(
                        "Valore non valido per rest11h_max_consecutive_exceptions per dipendente "
                        f"{employee_id}: {original_value}"
                    )
                consecutive_value = parsed_consecutive
                logger.info(
                    "employees.csv: rest11h_max_consecutive_exceptions override per dipendente %s: %s",
                    employee_id,
                    consecutive_value,
                )
        else:
            logger.info(
                "employees.csv: rest11h_max_consecutive_exceptions default applicato per dipendente %s: %s",
                employee_id,
                global_rest11h_consecutive,
            )

        rest11h_monthly_values.append(int(monthly_value))
        rest11h_consecutive_values.append(int(consecutive_value))

    df["rest11h_max_monthly_exceptions"] = rest11h_monthly_values
    df["rest11h_max_consecutive_exceptions"] = rest11h_consecutive_values

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
        role = str(row["role"]).strip()
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
        "role",
        "absence_full_day_hours_h",
        "absence_full_day_hours_effective_h",
        "reparto_id",
        "ore_dovute_mese_h",
        "dovuto_min",
        "saldo_init_min",
        "max_balance_delta_month_h",
        "max_month_min",
        "max_week_min",
        "weekly_rest_min_days",
        "rest11h_max_monthly_exceptions",
        "rest11h_max_consecutive_exceptions",
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
        "cross_max_shifts_month",
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
        return pd.DataFrame(columns=["role", "pool_id", "reparto_id"])

    df = pd.read_csv(path, dtype=str).fillna("")

    base_required = {"pool_id", "reparto_id"}
    missing_base = base_required - set(df.columns)
    if missing_base:
        raise LoaderError(
            "role_dept_pools.csv: colonne mancanti: " + ", ".join(sorted(missing_base))
        )

    role_columns = [col for col in ("role", "ruolo") if col in df.columns]
    if not role_columns:
        raise LoaderError(
            "role_dept_pools.csv: è richiesta la colonna 'role' (o il legacy 'ruolo')"
        )
    if len(role_columns) == 2:
        legacy = df["ruolo"].astype(str).str.strip()
        modern = df["role"].astype(str).str.strip()
        mismatch = legacy.ne(modern) & ~(legacy.eq("") & modern.eq(""))
        if mismatch.any():
            rows = ", ".join(str(i + 2) for i in mismatch[mismatch].index.tolist())
            raise LoaderError(
                "role_dept_pools.csv: colonne 'role' e 'ruolo' con valori diversi alle righe "
                f"[{rows}]"
            )
    role_source = role_columns[0]
    if role_source != "role":
        df = df.rename(columns={role_source: "role"})
    if "ruolo" in df.columns:
        df = df.drop(columns=["ruolo"])

    for col in ["role", "pool_id", "reparto_id"]:
        df[col] = df[col].astype(str).str.strip()

    if (df[["role", "pool_id", "reparto_id"]] == "").any().any():
        bad_rows = df.loc[(df[["role", "pool_id", "reparto_id"]] == "").any(axis=1)]
        raise LoaderError(
            "role_dept_pools.csv: valori vuoti non ammessi nelle colonne role/pool_id/reparto_id. "
            f"Righe interessate:\n{bad_rows}"
        )

    allowed_roles = set(
        _resolve_allowed_roles(defaults, fallback_roles=employees_df["role"].unique())
    )
    allowed_departments = set(_resolve_allowed_departments(defaults))

    bad_roles = sorted(set(df["role"].unique()) - allowed_roles)
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

    if df.duplicated(subset=["role", "pool_id", "reparto_id"]).any():
        dup = df[
            df.duplicated(subset=["role", "pool_id", "reparto_id"], keep=False)
        ].sort_values(["role", "pool_id", "reparto_id"])
        raise LoaderError(
            "role_dept_pools.csv: duplicati non ammessi su (role, pool_id, reparto_id):\n"
            f"{dup}"
        )

    return df.sort_values(["role", "pool_id", "reparto_id"]).reset_index(drop=True)


def build_department_compatibility(
    defaults: dict[str, Any], pools_df: pd.DataFrame, employees_df: pd.DataFrame
) -> pd.DataFrame:
    """Costruisce matrice compatibilità tra reparti basata sui pool."""
    allowed_roles = _resolve_allowed_roles(
        defaults, fallback_roles=employees_df["role"].unique()
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
        pools_by_role = {role: grp for role, grp in pools_df.groupby("role")}
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
        combos, columns=["role", "reparto_home", "reparto_target"]
    )
    if not compat_df.empty:
        compat_df = compat_df.sort_values(
            ["role", "reparto_home", "reparto_target"]
        ).reset_index(drop=True)
    return compat_df


def resolve_fulltime_baseline(config: dict, role: str | None) -> float:
    """Restituisce le ore contrattuali mensili del ruolo dal config defaults."""

    contract_by_role = _contract_hours_by_role(config.get("defaults"))

    if role is None:
        raise LoaderError("contract_hours_by_role_h non definito per il ruolo None")

    candidate = _require_contract_hours(contract_by_role, role, error_cls=LoaderError)

    try:
        baseline = float(candidate)
    except (TypeError, ValueError) as exc:
        raise LoaderError(
            f"Valore non numerico per contract_hours_by_role_h[{role}]: {candidate!r}"
        ) from exc

    if baseline <= 0:
        raise LoaderError(f"Il baseline full-time deve essere positivo ({baseline})")

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

    contract_by_role = _contract_hours_by_role(config.get("defaults"))

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
            default_hours = _require_contract_hours(
                contract_by_role, role_str, error_cls=LoaderError
            )
            try:
                due_hours = float(default_hours)
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
