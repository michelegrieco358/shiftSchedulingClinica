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


def _normalise(series: pd.Series) -> pd.Series:
    """Normalizza stringhe con strip+upper senza mutare la serie originale."""

    return series.astype(str).str.strip().str.upper()


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Converte una serie datetime tz-aware in naive su UTC."""

    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise TypeError("Le colonne start_dt e end_dt devono essere datetime tz-aware")
    return series.dt.tz_convert("UTC").dt.tz_localize(None)


def _iso_year_week_of(ts: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Restituisce (iso_year, iso_week) da una Series datetime tz-aware."""

    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise TypeError("La serie deve contenere datetime tz-aware")

    iso = ts.dt.isocalendar()
    return iso["year"], iso["week"]


def build_gap_pairs(
    shift_slots: pd.DataFrame,
    max_check_window_h: int = 15,
    handover_minutes: int = 0,
    *,
    add_debug: bool = True,
) -> pd.DataFrame:
    """Costruisce la tabella delle coppie di slot incompatibili per riposo."""

    if max_check_window_h <= 0:
        raise ValueError("max_check_window_h deve essere positivo")
    if handover_minutes < 0:
        raise ValueError("handover_minutes deve essere >= 0")

    _ensure_required_columns(shift_slots, REQUIRED_COLUMNS)

    has_shift_code = "shift_code" in shift_slots.columns
    has_in_scope = "in_scope" in shift_slots.columns

    reparto_dtype = shift_slots["reparto_id"].dtype if "reparto_id" in shift_slots else "object"
    slot_dtype = shift_slots["slot_id"].dtype if "slot_id" in shift_slots else "int64"

    base_columns = {
        "reparto_id": pd.Series(dtype=reparto_dtype),
        "s1_id": pd.Series(dtype=slot_dtype),
        "s2_id": pd.Series(dtype=slot_dtype),
        "gap_hours": pd.Series(dtype="float64"),
    }

    if add_debug:
        tz_dtype = pd.DatetimeTZDtype(tz="Europe/Rome")
        debug_columns: dict[str, pd.Series] = {
            "s1_end_dt": pd.Series(dtype=tz_dtype),
            "s2_start_dt": pd.Series(dtype=tz_dtype),
            "s1_end_date": pd.Series(dtype="object"),
            "s2_start_date": pd.Series(dtype="object"),
            "s1_iso_year": pd.Series(dtype=pd.UInt32Dtype()),
            "s1_iso_week": pd.Series(dtype=pd.UInt32Dtype()),
            "s2_iso_year": pd.Series(dtype=pd.UInt32Dtype()),
            "s2_iso_week": pd.Series(dtype=pd.UInt32Dtype()),
        }
        if has_shift_code:
            shift_dtype = shift_slots["shift_code"].dtype
            debug_columns["s1_shift_code"] = pd.Series(dtype=shift_dtype)
            debug_columns["s2_shift_code"] = pd.Series(dtype=shift_dtype)
        if has_in_scope:
            scope_dtype = shift_slots["in_scope"].dtype
            debug_columns["s1_in_scope"] = pd.Series(dtype=scope_dtype)
            debug_columns["s2_in_scope"] = pd.Series(dtype=scope_dtype)
            debug_columns["pair_crosses_scope"] = pd.Series(dtype="bool")
        base_columns.update(debug_columns)

    if shift_slots.empty:
        return pd.DataFrame(base_columns)

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
        gap_hours = gap_delta.dt.total_seconds().to_numpy() / 3600.0

        handover_hours = handover_minutes / 60.0
        gap_hours_eff = np.maximum(gap_hours - handover_hours, 0.0)

        valid_mask = (gap_hours_eff > 0) & (gap_hours_eff <= float(max_check_window_h))

        if not np.any(valid_mask):
            continue

        s1_valid = s1_idx[valid_mask]
        s2_valid = s2_idx[valid_mask]

        gap_valid = gap_hours_eff[valid_mask]

        pairs_df = pd.DataFrame(
            {
                "reparto_id": reparto_id,
                "s1_id": group_sorted["slot_id"].iloc[s1_valid].to_numpy(),
                "s2_id": group_sorted["slot_id"].iloc[s2_valid].to_numpy(),
                "gap_hours": gap_valid,
            }
        )

        if add_debug:
            s1_end_valid = s1_end.iloc[valid_mask].reset_index(drop=True)
            s2_start_valid = s2_start.iloc[valid_mask].reset_index(drop=True)

            pairs_df["s1_end_dt"] = s1_end_valid
            pairs_df["s2_start_dt"] = s2_start_valid
            pairs_df["s1_end_date"] = s1_end_valid.dt.date
            pairs_df["s2_start_date"] = s2_start_valid.dt.date

            s1_iso_year, s1_iso_week = _iso_year_week_of(s1_end_valid)
            s2_iso_year, s2_iso_week = _iso_year_week_of(s2_start_valid)
            pairs_df["s1_iso_year"] = s1_iso_year.reset_index(drop=True)
            pairs_df["s1_iso_week"] = s1_iso_week.reset_index(drop=True)
            pairs_df["s2_iso_year"] = s2_iso_year.reset_index(drop=True)
            pairs_df["s2_iso_week"] = s2_iso_week.reset_index(drop=True)

            if has_shift_code:
                s1_shift = group_sorted["shift_code"].iloc[s1_valid].reset_index(drop=True)
                s2_shift = group_sorted["shift_code"].iloc[s2_valid].reset_index(drop=True)
                pairs_df["s1_shift_code"] = s1_shift
                pairs_df["s2_shift_code"] = s2_shift

            if has_in_scope:
                s1_scope = group_sorted["in_scope"].iloc[s1_valid].reset_index(drop=True)
                s2_scope = group_sorted["in_scope"].iloc[s2_valid].reset_index(drop=True)
                pairs_df["s1_in_scope"] = s1_scope
                pairs_df["s2_in_scope"] = s2_scope
                pairs_df["pair_crosses_scope"] = (~s1_scope.fillna(False)) & s2_scope.fillna(False)

        result_frames.append(pairs_df.reset_index(drop=True))

    if not result_frames:
        return pd.DataFrame(base_columns)

    result_df = pd.concat(result_frames, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=["reparto_id", "s1_id", "s2_id"])
    result_df = result_df.sort_values(["reparto_id", "s1_id", "s2_id"]).reset_index(drop=True)

    return result_df


def _validate_pools_unique(pools_df: pd.DataFrame) -> pd.DataFrame:
    """Valida che ogni reparto appaia in al più un pool e normalizza le chiavi."""

    required_cols = {"pool_id", "reparto_id"}
    missing = required_cols - set(pools_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"pools: colonne mancanti: {missing_str}")

    pools_norm = pools_df.copy()
    if pools_norm["pool_id"].isna().any() or pools_norm["reparto_id"].isna().any():
        raise ValueError("pools: pool_id e reparto_id non possono essere null")
    pools_norm["_pool_key"] = _normalise(pools_norm["pool_id"])
    pools_norm["_reparto_key"] = _normalise(pools_norm["reparto_id"])

    duplicates = pools_norm.duplicated(subset=["_reparto_key"], keep=False)
    if duplicates.any():
        duplicated_reparti = sorted(pools_norm.loc[duplicates, "reparto_id"].unique())
        raise ValueError(
            "pools: alcuni reparti sono associati a più pool: "
            + ", ".join(duplicated_reparti)
        )

    return pools_norm.drop_duplicates(subset=["_reparto_key"])


def _prepare_slots_for_pool_pairs(
    shift_slots: pd.DataFrame,
    pools_df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, bool]:
    """Prepara la tabella degli slot con chiavi normalizzate e dati di pool."""

    slots_df = shift_slots.copy()
    slots_df["start_dt"] = pd.to_datetime(slots_df["start_dt"], errors="coerce")
    slots_df["end_dt"] = pd.to_datetime(slots_df["end_dt"], errors="coerce")

    if slots_df["start_dt"].isna().any() or slots_df["end_dt"].isna().any():
        raise ValueError("shift_slots: valori non validi in start_dt o end_dt")

    if slots_df["reparto_id"].isna().any():
        raise ValueError("shift_slots: reparto_id non può essere nullo")

    if not (
        pd.api.types.is_datetime64tz_dtype(slots_df["start_dt"]) and
        pd.api.types.is_datetime64tz_dtype(slots_df["end_dt"])
    ):
        raise TypeError(
            "shift_slots: start_dt e end_dt devono essere datetime con timezone"
        )

    slots_df["_reparto_key"] = _normalise(slots_df["reparto_id"])

    pools_norm = _validate_pools_unique(pools_df)

    slots_df = slots_df.merge(
        pools_norm[["_reparto_key", "pool_id", "_pool_key"]],
        on="_reparto_key",
        how="left",
    )

    has_shift_code = "shift_code" in slots_df.columns
    has_in_scope = "in_scope" in slots_df.columns

    return slots_df, has_shift_code, has_in_scope


def _build_pool_pairs(
    group: pd.DataFrame,
    max_check_window_h: int,
    handover_minutes: int,
    *,
    add_debug: bool,
    has_shift_code: bool,
    has_in_scope: bool,
    pair_type: str,
) -> pd.DataFrame:
    """Calcola le coppie di slot all'interno di un gruppo di pool."""

    if group.empty:
        return pd.DataFrame()

    group_sorted = group.sort_values("start_dt").reset_index(drop=True)

    start_naive = _to_naive_utc(group_sorted["start_dt"])
    end_naive = _to_naive_utc(group_sorted["end_dt"])

    start_values = start_naive.to_numpy(dtype="datetime64[ns]")
    end_values = end_naive.to_numpy(dtype="datetime64[ns]")

    window_delta = np.timedelta64(max_check_window_h, "h")

    left_idx = np.searchsorted(start_values, end_values, side="right")
    right_idx = np.searchsorted(
        start_values, end_values + window_delta, side="right"
    )

    counts = right_idx - left_idx
    total_pairs = int(counts.sum())
    if total_pairs == 0:
        return pd.DataFrame()

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
    gap_hours = gap_delta.dt.total_seconds().to_numpy() / 3600.0
    handover_hours = handover_minutes / 60.0
    gap_hours_eff = np.maximum(gap_hours - handover_hours, 0.0)

    valid_mask = (gap_hours_eff > 0) & (gap_hours_eff <= float(max_check_window_h))
    if not np.any(valid_mask):
        return pd.DataFrame()

    s1_valid = s1_idx[valid_mask]
    s2_valid = s2_idx[valid_mask]
    gap_valid = gap_hours_eff[valid_mask]

    s1_reparto = group_sorted["reparto_id"].iloc[s1_valid].to_numpy()
    s2_reparto = group_sorted["reparto_id"].iloc[s2_valid].to_numpy()

    if pair_type == "CROSS":
        cross_mask = s1_reparto != s2_reparto
        if not np.any(cross_mask):
            return pd.DataFrame()
        s1_valid = s1_valid[cross_mask]
        s2_valid = s2_valid[cross_mask]
        gap_valid = gap_valid[cross_mask]
        s1_reparto = s1_reparto[cross_mask]
        s2_reparto = s2_reparto[cross_mask]

    pairs_df = pd.DataFrame(
        {
            "s1_id": group_sorted["slot_id"].iloc[s1_valid].to_numpy(),
            "s2_id": group_sorted["slot_id"].iloc[s2_valid].to_numpy(),
            "s1_reparto_id": s1_reparto,
            "s2_reparto_id": s2_reparto,
            "gap_hours": gap_valid,
            "pair_type": pair_type,
        }
    )

    if add_debug:
        s1_end_final = group_sorted["end_dt"].iloc[s1_valid].reset_index(drop=True)
        s2_start_final = group_sorted["start_dt"].iloc[s2_valid].reset_index(drop=True)
        pairs_df["s1_end_dt"] = s1_end_final
        pairs_df["s2_start_dt"] = s2_start_final
        if has_shift_code:
            s1_shift = group_sorted["shift_code"].iloc[s1_valid].reset_index(drop=True)
            s2_shift = group_sorted["shift_code"].iloc[s2_valid].reset_index(drop=True)
            pairs_df["s1_shift_code"] = s1_shift
            pairs_df["s2_shift_code"] = s2_shift
        if has_in_scope:
            s1_scope = group_sorted["in_scope"].iloc[s1_valid].reset_index(drop=True)
            s2_scope = group_sorted["in_scope"].iloc[s2_valid].reset_index(drop=True)
            pairs_df["s1_in_scope"] = s1_scope
            pairs_df["s2_in_scope"] = s2_scope
            pairs_df["pair_crosses_scope"] = (
                (~s1_scope.fillna(False)) & s2_scope.fillna(False)
            )

    return pairs_df.reset_index(drop=True)


def _empty_gap_pairs_frame(
    shift_slots: pd.DataFrame,
    *,
    add_debug: bool,
) -> pd.DataFrame:
    """Costruisce un DataFrame vuoto con gli stessi dtypes attesi."""

    slot_dtype = shift_slots["slot_id"].dtype if "slot_id" in shift_slots.columns else "object"
    reparto_dtype = (
        shift_slots["reparto_id"].dtype if "reparto_id" in shift_slots.columns else "object"
    )

    data: dict[str, pd.Series] = {
        "s1_id": pd.Series(dtype=slot_dtype),
        "s2_id": pd.Series(dtype=slot_dtype),
        "s1_reparto_id": pd.Series(dtype=reparto_dtype),
        "s2_reparto_id": pd.Series(dtype=reparto_dtype),
        "gap_hours": pd.Series(dtype="float64"),
        "pair_type": pd.Series(dtype="object"),
    }

    if add_debug:
        if "start_dt" in shift_slots.columns and pd.api.types.is_datetime64tz_dtype(
            shift_slots["start_dt"]
        ):
            tz_dtype = shift_slots["start_dt"].dtype
        else:
            tz_dtype = pd.DatetimeTZDtype(tz="Europe/Rome")
        data.update(
            {
                "s1_end_dt": pd.Series(dtype=tz_dtype),
                "s2_start_dt": pd.Series(dtype=tz_dtype),
            }
        )
        if "shift_code" in shift_slots.columns:
            data["s1_shift_code"] = pd.Series(dtype=shift_slots["shift_code"].dtype)
            data["s2_shift_code"] = pd.Series(dtype=shift_slots["shift_code"].dtype)
        if "in_scope" in shift_slots.columns:
            data["s1_in_scope"] = pd.Series(dtype=shift_slots["in_scope"].dtype)
            data["s2_in_scope"] = pd.Series(dtype=shift_slots["in_scope"].dtype)
            data["pair_crosses_scope"] = pd.Series(dtype="bool")

    return pd.DataFrame(data)


def build_gap_pairs_pool(
    shift_slots: pd.DataFrame,
    pools: pd.DataFrame,
    max_check_window_h: int = 15,
    handover_minutes: int = 0,
    *,
    add_debug: bool = True,
) -> pd.DataFrame:
    """Costruisce coppie di slot intra reparto e cross-pool entro la stessa finestra.

    Le coppie rispettano ``end(s1) < start(s2)`` e ``start(s2)`` entro la
    finestra definita da ``max_check_window_h``. I gap vengono ridotti di
    ``handover_minutes`` (espressi in minuti) e troncati a zero prima di
    applicare il filtro ``gap_hours > 0``.

    Sono considerate:

    * coppie ``INTRA``: slot appartenenti allo stesso reparto;
    * coppie ``CROSS``: slot di reparti diversi ma appartenenti allo stesso pool.

    Quando ``add_debug`` è vero vengono aggiunte colonne di supporto
    (timestamp, scope e codici turno se disponibili).
    """

    if max_check_window_h <= 0:
        raise ValueError("max_check_window_h deve essere positivo")
    if handover_minutes < 0:
        raise ValueError("handover_minutes deve essere >= 0")

    _ensure_required_columns(shift_slots, REQUIRED_COLUMNS)

    if shift_slots.empty:
        return _empty_gap_pairs_frame(shift_slots, add_debug=add_debug)

    slots_with_pool, has_shift_code, has_in_scope = _prepare_slots_for_pool_pairs(
        shift_slots, pools
    )

    result_frames: list[pd.DataFrame] = []

    intra_pairs = build_gap_pairs(
        shift_slots,
        max_check_window_h=max_check_window_h,
        handover_minutes=handover_minutes,
        add_debug=add_debug,
    )
    if not intra_pairs.empty:
        intra_pairs = intra_pairs.rename(columns={"reparto_id": "s1_reparto_id"})
        intra_pairs["s2_reparto_id"] = intra_pairs["s1_reparto_id"]
        intra_pairs["pair_type"] = "INTRA"
        result_frames.append(intra_pairs)

    pool_slots = slots_with_pool.loc[slots_with_pool["_pool_key"].notna()].copy()
    if not pool_slots.empty:
        for _, group in pool_slots.groupby("_pool_key", sort=False):
            pool_pairs = _build_pool_pairs(
                group,
                max_check_window_h,
                handover_minutes,
                add_debug=add_debug,
                has_shift_code=has_shift_code,
                has_in_scope=has_in_scope,
                pair_type="CROSS",
            )
            if not pool_pairs.empty:
                result_frames.append(pool_pairs)

    if not result_frames:
        return _empty_gap_pairs_frame(shift_slots, add_debug=add_debug)

    result_df = pd.concat(result_frames, ignore_index=True, sort=False)

    result_df = result_df.drop_duplicates(
        subset=["s1_id", "s2_id", "pair_type"], keep="first"
    )
    result_df = result_df.sort_values(["s1_id", "s2_id", "pair_type"]).reset_index(
        drop=True
    )

    expected_cols = [
        "s1_id",
        "s2_id",
        "s1_reparto_id",
        "s2_reparto_id",
        "gap_hours",
        "pair_type",
    ]
    if add_debug:
        debug_cols: list[str] = ["s1_end_dt", "s2_start_dt"]
        if has_shift_code:
            debug_cols.extend(["s1_shift_code", "s2_shift_code"])
        if has_in_scope:
            debug_cols.extend(["s1_in_scope", "s2_in_scope", "pair_crosses_scope"])
        expected_cols.extend(debug_cols)

    for col in expected_cols:
        if col not in result_df.columns:
            result_df[col] = pd.Series(dtype="object")

    return result_df.loc[:, expected_cols]


__all__ = ["build_gap_pairs", "_iso_year_week_of", "build_gap_pairs_pool"]
