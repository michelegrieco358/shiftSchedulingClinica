from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd
import yaml

from .utils import LoaderError, _ensure_cols, _parse_date


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    try:
        _ = _parse_date(cfg["horizon"]["start_date"])
        _ = _parse_date(cfg["horizon"]["end_date"])
    except Exception as exc:  # pragma: no cover - mantiene messaggio originale
        raise LoaderError(
            f"config: horizon/start_date,end_date mancanti o non validi: {exc}"
        )

    defaults = cfg.get("defaults")
    if defaults is None:
        defaults = {}
        cfg["defaults"] = defaults
    if not isinstance(defaults, dict):
        raise LoaderError("config: defaults deve essere un dizionario valido")

    if "weekly_rest_min_h" in defaults:
        warnings.warn(
            "config: defaults.weekly_rest_min_h è deprecato e verrà ignorato; "
            "utilizzare defaults.weekly_rest_min_days",
            stacklevel=2,
        )
        defaults.pop("weekly_rest_min_h", None)

    weekly_rest_min_days = defaults.get("weekly_rest_min_days", 1)
    if weekly_rest_min_days is None:
        weekly_rest_min_days = 1

    if isinstance(weekly_rest_min_days, bool) or not isinstance(
        weekly_rest_min_days, int
    ):
        raise LoaderError(
            "config: defaults.weekly_rest_min_days deve essere un intero ≥ 0 "
            f"(trovato: {weekly_rest_min_days!r})"
        )
    if weekly_rest_min_days < 0:
        raise LoaderError(
            "config: defaults.weekly_rest_min_days deve essere un intero ≥ 0 "
            f"(trovato: {weekly_rest_min_days!r})"
        )

    defaults["weekly_rest_min_days"] = weekly_rest_min_days

    rest11h_cfg = defaults.get("rest11h")
    if rest11h_cfg is None:
        rest11h_cfg = {}
    if not isinstance(rest11h_cfg, dict):
        raise LoaderError("config: defaults.rest11h deve essere un dizionario")

    def _coerce_rest11h(value: Any, label: str) -> int:
        if value in (None, ""):
            return 0
        if isinstance(value, bool):
            raise LoaderError(f"{label} deve essere un intero ≥ 0 (trovato: {value!r})")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            s = value.strip()
            if s == "":
                return 0
            try:
                parsed = int(s)
            except ValueError as exc:
                raise LoaderError(
                    f"{label} deve essere un intero ≥ 0 (trovato: {value!r})"
                ) from exc
        else:
            raise LoaderError(f"{label} deve essere un intero ≥ 0 (trovato: {value!r})")
        if parsed < 0:
            raise LoaderError(
                f"{label} deve essere un intero ≥ 0 (trovato: {value!r})"
            )
        return parsed

    defaults["rest11h"] = {
        "max_monthly_exceptions": _coerce_rest11h(
            rest11h_cfg.get("max_monthly_exceptions"),
            "config: defaults.rest11h.max_monthly_exceptions",
        ),
        "max_consecutive_exceptions": _coerce_rest11h(
            rest11h_cfg.get("max_consecutive_exceptions"),
            "config: defaults.rest11h.max_consecutive_exceptions",
        ),
    }

    absences_cfg = defaults.get("absences")
    if absences_cfg is None:
        absences_cfg = {}
    if not isinstance(absences_cfg, dict):
        raise LoaderError("config: defaults.absences deve essere un dizionario")

    count_as_worked = absences_cfg.get("count_as_worked_hours", True)
    if count_as_worked is None:
        count_as_worked = True
    if not isinstance(count_as_worked, bool):
        raise LoaderError(
            "config: defaults.absences.count_as_worked_hours deve essere booleano (true/false)"
        )

    role_hours_cfg = absences_cfg.get("full_day_hours_by_role_h") or {}
    if not isinstance(role_hours_cfg, dict):
        raise LoaderError(
            "config: defaults.absences.full_day_hours_by_role_h deve essere un dizionario"
        )

    role_hours: dict[str, float] = {}
    for raw_role, raw_value in role_hours_cfg.items():
        role_key = str(raw_role).strip()
        if not role_key:
            continue
        try:
            parsed_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise LoaderError(
                "config: defaults.absences.full_day_hours_by_role_h deve contenere valori numerici ≥ 0"
            ) from exc
        if parsed_value != parsed_value or parsed_value < 0:
            raise LoaderError(
                "config: defaults.absences.full_day_hours_by_role_h deve contenere valori numerici ≥ 0"
            )
        role_hours[role_key] = parsed_value

    fallback_raw = absences_cfg.get("fallback_contract_daily_avg_h", 0.0)
    if fallback_raw in (None, ""):
        fallback_value = 0.0
    else:
        try:
            fallback_value = float(fallback_raw)
        except (TypeError, ValueError) as exc:
            raise LoaderError(
                "config: defaults.absences.fallback_contract_daily_avg_h deve essere un numero ≥ 0"
            ) from exc
    if fallback_value != fallback_value or fallback_value < 0:
        raise LoaderError(
            "config: defaults.absences.fallback_contract_daily_avg_h deve essere un numero ≥ 0"
        )

    defaults["absences"] = {
        "count_as_worked_hours": bool(count_as_worked),
        "full_day_hours_by_role_h": role_hours,
        "fallback_contract_daily_avg_h": float(fallback_value),
    }

    overstaffing_cfg = defaults.get("overstaffing")
    if overstaffing_cfg is None:
        overstaffing_cfg = {}
    if not isinstance(overstaffing_cfg, dict):
        raise LoaderError("config: defaults.overstaffing deve essere un dizionario")

    overstaff_enabled = overstaffing_cfg.get("enabled", True)
    if overstaff_enabled is None:
        overstaff_enabled = True
    if not isinstance(overstaff_enabled, bool):
        raise LoaderError(
            "config: defaults.overstaffing.enabled deve essere booleano (true/false)"
        )

    cap_default_raw = overstaffing_cfg.get("group_cap_default", 0)
    if cap_default_raw in (None, ""):
        cap_default_raw = 0
    try:
        cap_default = int(cap_default_raw)
    except (TypeError, ValueError) as exc:
        raise LoaderError(
            "config: defaults.overstaffing.group_cap_default deve essere un intero"
        ) from exc
    if cap_default < 0:
        raise LoaderError(
            "config: defaults.overstaffing.group_cap_default deve essere ≥ 0"
        )

    defaults["overstaffing"] = {
        "enabled": overstaff_enabled,
        "group_cap_default": cap_default,
    }

    def _normalize_fairness_section(section: Any, label: str) -> dict[str, float]:
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise LoaderError(f"{label} deve essere un dizionario")
        normalized = dict(section)
        for key in ("night_weight", "weekend_weight"):
            raw = normalized.get(key, 0.0)
            if raw in (None, ""):
                value = 0.0
            else:
                try:
                    value = float(raw)
                except (TypeError, ValueError) as exc:
                    raise LoaderError(
                        f"{label}.{key} deve essere un numero >= 0 (trovato: {raw!r})"
                    ) from exc
            if value < 0:
                raise LoaderError(
                    f"{label}.{key} deve essere un numero >= 0 (trovato: {raw!r})"
                )
            normalized[key] = float(value)
        return normalized

    cfg["fairness"] = _normalize_fairness_section(cfg.get("fairness"), "config: fairness")
    defaults["fairness"] = _normalize_fairness_section(
        defaults.get("fairness"), "config: defaults.fairness"
    )

    return cfg


def load_holidays(path: str) -> pd.DataFrame:
    """Carica le festività normalizzando le date e rimuovendo duplicati."""

    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "name"])

    df = pd.read_csv(path, dtype=str).fillna("")

    rename_map = {}
    if "data" in df.columns and "date" not in df.columns:
        rename_map["data"] = "date"
    if "descrizione" in df.columns and "name" not in df.columns:
        rename_map["descrizione"] = "name"
    if rename_map:
        df = df.rename(columns=rename_map)

    _ensure_cols(df, {"date", "name"}, "holidays.csv")

    df["date"] = df["date"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    parsed_dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    if parsed_dates.dt.tz is not None:
        parsed_dates = parsed_dates.dt.tz_convert(None)
    parsed_dates = parsed_dates.dt.normalize()

    df["date"] = parsed_dates
    df = df.dropna(subset=["date"])

    df = df[~df["date"].duplicated(keep="first")]

    return df[["date", "name"]].sort_values("date").reset_index(drop=True)
