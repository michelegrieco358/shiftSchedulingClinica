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
