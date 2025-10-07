"""Utility per la generazione delle coppie di turni incompatibili per riposo."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: set[str] = {
    "slot_id",
    "reparto_id",
    "start_dt",
    "end_dt",
}


def _ensure_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Verifica che il DataFrame contenga tutte le colonne richieste."""

    missing = set(columns) - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Colonne mancanti in shift_slots: {missing_str}")


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Converte una serie datetime tz-aware in naive su UTC."""

    if not pd.api.types.is_datetime64tz_dtype(series):
        raise TypeError("Le colonne start_dt e end_dt devono essere datetime tz-aware")
    return series.dt.tz_convert("UTC").dt.tz_localize(None)


def build_gap_pairs(
    shift_slots: pd.DataFrame,
    max_check_window_h: int = 15,
    *,
    add_debug: bool = False,
) -> pd.DataFrame:
    """Costruisce la tabella delle coppie di slot incompatibili per riposo."""

    if max_check_window_h <= 0:
        raise ValueError("max_check_window_h deve essere positivo")

    _ensure_required_columns(shift_slots, REQUIRED_COLUMNS)

    if shift_slots.empty:
        columns = ["reparto_id", "s1_id", "s2_id", "gap_hours"]
        if add_debug:
            columns += ["s1_end_dt", "s2_start_dt"]
        return pd.DataFrame(columns=columns)

    result_frames: list[pd.DataFrame] = []

    grouped = shift_slots.groupby("reparto_id", sort=False, dropna=False)

    for reparto_id, group in grouped:
        group_sorted = group.sort_values("start_dt").reset_index(drop=True)

        start_naive = _to_naive_utc(group_sorted["start_dt"])  # naive per searchsorted
        end_naive = _to_naive_utc(group_sorted["end_dt"])

        start_values = start_naive.to_numpy(dtype="datetime64[ns]")
        end_values = end_naive.to_numpy(dtype="datetime64[ns]")

        window_delta = np.timedelta64(max_check_window_h, "h")

        left_idx = np.searchsorted(start_values, end_values, side="right")
        right_idx = np.searchsorted(start_values, end_values + window_delta, side="right")

        counts = right_idx - left_idx
        total_pairs = int(counts.sum())
        if total_pairs == 0:
            continue

        s1_idx = np.repeat(np.arange(len(group_sorted), dtype=int), counts)

        s2_idx = np.empty(total_pairs, dtype=int)
        cursor = 0
        for l_idx, r_idx in zip(left_idx, right_idx):
            span = r_idx - l_idx
            if span <= 0:
                continue
            s2_idx[cursor : cursor + span] = np.arange(l_idx, r_idx)
            cursor += span

        s1_end = group_sorted["end_dt"].iloc[s1_idx].reset_index(drop=True)
        s2_start = group_sorted["start_dt"].iloc[s2_idx].reset_index(drop=True)

        gap_delta = s2_start - s1_end
        gap_hours = gap_delta.dt.total_seconds() / 3600.0

        valid_mask = gap_hours > 0
        valid_mask &= gap_hours <= float(max_check_window_h)

        if not valid_mask.any():
            continue

        pairs_df = pd.DataFrame(
            {
                "reparto_id": reparto_id,
                "s1_id": group_sorted["slot_id"].iloc[s1_idx].to_numpy(),
                "s2_id": group_sorted["slot_id"].iloc[s2_idx].to_numpy(),
                "gap_hours": gap_hours,
            }
        )

        if add_debug:
            pairs_df["s1_end_dt"] = s1_end.to_numpy()
            pairs_df["s2_start_dt"] = s2_start.to_numpy()

        result_frames.append(pairs_df.loc[valid_mask].reset_index(drop=True))

    if not result_frames:
        columns = ["reparto_id", "s1_id", "s2_id", "gap_hours"]
        if add_debug:
            columns += ["s1_end_dt", "s2_start_dt"]
        return pd.DataFrame(columns=columns)

    result_df = pd.concat(result_frames, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=["s1_id", "s2_id"])
    result_df = result_df.sort_values(["reparto_id", "s1_id", "s2_id"]).reset_index(drop=True)

    return result_df


__all__ = ["build_gap_pairs"]
