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
