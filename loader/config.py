from __future__ import annotations

import os
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
    return cfg


def load_holidays(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["data", "data_dt", "descrizione"])

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"data", "descrizione"}, "holidays.csv")

    df["data"] = df["data"].astype(str).str.strip()
    df["descrizione"] = df["descrizione"].astype(str).str.strip()

    if (df["data"] == "").any():
        raise LoaderError("holidays.csv: la colonna 'data' non può contenere valori vuoti")

    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"holidays.csv: formato data non valido: {exc}")

    dup_dates = df[df["data"].duplicated(keep=False)]["data"].unique()
    if len(dup_dates) > 0:
        raise LoaderError(
            "holidays.csv: date duplicate non ammesse: " + ", ".join(sorted(dup_dates))
        )

    return df[["data", "data_dt", "descrizione"]].sort_values("data").reset_index(drop=True)
