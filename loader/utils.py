from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Set

import pandas as pd


TURNI_DOMANDA: Set[str] = {"M", "P", "N"}


class LoaderError(Exception):
    """Errore di caricamento dati."""


def _parse_date(s: str) -> date:
    """Converte stringa ISO (YYYY-MM-DD) in oggetto date."""
    return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()


def _ensure_cols(df: pd.DataFrame, required: Set[str], label: str) -> None:
    """Verifica che il DataFrame abbia tutte le colonne richieste."""
    missing = required - set(df.columns)
    if missing:
        raise LoaderError(f"{label}: colonne mancanti {sorted(missing)}")


def _resolve_allowed_roles(
    defaults: dict, fallback_roles: Iterable[str] | None = None
) -> list[str]:
    """Risolve e normalizza la lista dei ruoli ammessi dalla configurazione."""
    allowed_roles_cfg = defaults.get("allowed_roles", None)
    roles: list[str]
    if isinstance(allowed_roles_cfg, str):
        roles = [
            x.strip()
            for x in allowed_roles_cfg.replace(",", "|").split("|")
            if x.strip()
        ]
    elif isinstance(allowed_roles_cfg, (list, tuple, set)):
        roles = [str(x).strip() for x in allowed_roles_cfg if str(x).strip()]
    else:
        roles = []

    if not roles and fallback_roles is not None:
        roles = [str(x).strip() for x in fallback_roles if str(x).strip()]

    seen = set()
    deduped: list[str] = []
    for role in roles:
        if role not in seen:
            seen.add(role)
            deduped.append(role)
    return deduped


def _resolve_allowed_departments(defaults: dict) -> list[str]:
    """Risolve e normalizza la lista dei reparti ammessi dalla configurazione."""
    departments_cfg = defaults.get("departments", None)
    if isinstance(departments_cfg, str):
        departments = [
            x.strip()
            for x in departments_cfg.replace(",", "|").split("|")
            if x.strip()
        ]
    elif isinstance(departments_cfg, (list, tuple, set)):
        departments = [str(x).strip() for x in departments_cfg if str(x).strip()]
    else:
        departments = []

    if not departments:
        raise LoaderError(
            "config: defaults.departments deve essere una lista non vuota di reparti ammessi"
        )

    seen = set()
    deduped: list[str] = []
    for dept in departments:
        if dept not in seen:
            seen.add(dept)
            deduped.append(dept)
    return deduped


def _compute_horizon_window(calendar_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the inclusive horizon bounds derived from ``calendar_df``."""

    if calendar_df.empty:
        raise LoaderError("calendar_df non può essere vuoto per calcolare l'orizzonte")

    if "data_dt" in calendar_df.columns:
        dates = pd.to_datetime(calendar_df["data_dt"])
    elif "data" in calendar_df.columns:
        dates = pd.to_datetime(calendar_df["data"], errors="coerce")
    else:
        raise LoaderError(
            "calendar_df deve contenere la colonna 'data_dt' (o 'data') per determinare l'orizzonte"
        )

    if dates.isna().all():
        raise LoaderError("calendar_df contiene solo date non valide")

    horizon_mask = calendar_df.get("is_in_horizon")
    if horizon_mask is not None:
        horizon_mask = horizon_mask.astype(bool)
    else:
        horizon_mask = pd.Series(False, index=calendar_df.index)

    if horizon_mask.any():
        horizon_dates = dates.loc[horizon_mask]
        start = horizon_dates.min().normalize()
        end = horizon_dates.max().normalize() + pd.Timedelta(days=1)
    else:
        start = dates.min().normalize()
        end = dates.max().normalize() + pd.Timedelta(days=1)

    return pd.Timestamp(start), pd.Timestamp(end)
