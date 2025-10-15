from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _ensure_date(value: date | pd.Timestamp | str) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    converted = pd.to_datetime(value)
    if pd.isna(converted):
        raise ValueError("Invalid date value provided")
    return converted.date()


def compute_month_progressive_balance(
    employees_df: pd.DataFrame,
    history_df: pd.DataFrame | None,
    plan_df: pd.DataFrame | None,
    month_start: date,
    month_end: date,
    horizon_start: date,
    horizon_end: date,
) -> pd.DataFrame:
    """Compute the progressive month balance per employee.

    Notes
    -----
    * ``start_balance`` is always the balance at the end of the previous
      month, regardless of the horizon start day.
    * When the planning horizon ends at the end of the month the
      pro-rata computation is skipped and the full ``hours_due_month`` is
      used instead.
    """

    required_columns = {"employee_id", "hours_due_month", "start_balance"}
    missing_columns = required_columns.difference(employees_df.columns)
    if missing_columns:
        raise ValueError(
            f"employees_df is missing required columns: {sorted(missing_columns)}"
        )

    month_start_d = _ensure_date(month_start)
    month_end_d = _ensure_date(month_end)
    horizon_start_d = _ensure_date(horizon_start)
    horizon_end_d = _ensure_date(horizon_end)

    if horizon_end_d < month_start_d or horizon_start_d > month_end_d:
        raise ValueError("Horizon is outside the provided month range")
    if horizon_start_d > horizon_end_d:
        raise ValueError("horizon_start must be on or before horizon_end")

    base = (
        employees_df.loc[:, ["employee_id", "hours_due_month", "start_balance"]]
        .copy()
    )
    base["employee_id"] = base["employee_id"].astype(str)
    base["hours_due_month"] = pd.to_numeric(
        base["hours_due_month"], errors="coerce"
    ).fillna(0.0)
    base["start_balance"] = pd.to_numeric(
        base["start_balance"], errors="coerce"
    ).fillna(0.0)

    def _aggregate_hours(
        df: pd.DataFrame | None,
        value_column: str,
        output_column: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame({"employee_id": [], output_column: []})
        work = df.copy()
        work["employee_id"] = work["employee_id"].astype(str)
        if "date" not in work.columns:
            raise ValueError("Expected 'date' column in dataframe")
        if value_column not in work.columns:
            raise ValueError(f"Expected '{value_column}' column in dataframe")
        work_dates = pd.to_datetime(work["date"], errors="coerce").dt.date
        mask = work_dates.ge(start) & work_dates.le(end)
        if not mask.any():
            return pd.DataFrame({"employee_id": [], output_column: []})
        work = work.loc[mask]
        work[value_column] = pd.to_numeric(work[value_column], errors="coerce").fillna(0.0)
        aggregated = work.groupby("employee_id", as_index=False)[value_column].sum()
        aggregated = aggregated.rename(columns={value_column: output_column})
        return aggregated

    hist_end = horizon_start_d - timedelta(days=1)
    if horizon_start_d <= month_start_d:
        hist_hours = pd.DataFrame({"employee_id": [], "hours_hist_to_hstart": []})
    else:
        hist_hours = _aggregate_hours(
            history_df,
            "hours_effective",
            "hours_hist_to_hstart",
            month_start_d,
            hist_end,
        )

    plan_hours = _aggregate_hours(
        plan_df,
        "hours_planned",
        "hours_plan_horizon",
        horizon_start_d,
        horizon_end_d,
    )

    result = base.merge(hist_hours, on="employee_id", how="left").merge(
        plan_hours, on="employee_id", how="left"
    )

    result[["hours_hist_to_hstart", "hours_plan_horizon"]] = result[
        ["hours_hist_to_hstart", "hours_plan_horizon"]
    ].fillna(0.0)
    result["hours_eff_to_date"] = (
        result["hours_hist_to_hstart"] + result["hours_plan_horizon"]
    )

    if horizon_end_d == month_end_d:
        result["hours_due_period"] = result["hours_due_month"]
    else:
        days_in_month = (month_end_d - month_start_d).days + 1
        days_to_date = (horizon_end_d - month_start_d).days + 1
        ratio = days_to_date / days_in_month
        result["hours_due_period"] = result["hours_due_month"] * ratio

    result["end_balance"] = (
        result["start_balance"]
        + (result["hours_eff_to_date"] - result["hours_due_period"])
    )

    columns = [
        "employee_id",
        "hours_hist_to_hstart",
        "hours_plan_horizon",
        "hours_eff_to_date",
        "hours_due_period",
        "start_balance",
        "end_balance",
    ]

    return result[columns].sort_values("employee_id").reset_index(drop=True)


def compute_adaptive_coefficients(
    balances: pd.Series | pd.DataFrame,
    p_low: float = 0.10,
    p_high: float = 0.90,
    abs_min: float = -60.0,
    abs_max: float = 60.0,
    min_range: float = 10.0,
) -> pd.DataFrame:
    """Return adaptive coefficients for under/over-hour penalties.

    Parameters
    ----------
    balances:
        A series indexed by ``employee_id`` containing the starting balance
        (in hours, measured at the end of the previous month). A DataFrame
        with columns ``employee_id`` and ``start_balance`` is also accepted.
    p_low, p_high:
        Percentiles used to derive a robust operating range before clamping.
    abs_min, abs_max:
        Safety clamps applied after percentile extraction.
    min_range:
        Minimum desired range width in hours. When the percentile window is
        narrower the range is expanded symmetrically around the median.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``employee_id`` with the columns ``c_under`` and
        ``c_over`` containing values in the interval ``[1, 2]``.
    """

    if isinstance(balances, pd.DataFrame):
        required_cols = {"employee_id", "start_balance"}
        missing = required_cols.difference(balances.columns)
        if missing:
            raise ValueError(
                "balances dataframe is missing required columns: "
                f"{sorted(missing)}"
            )
        index = balances["employee_id"].astype(str).str.strip()
        values = pd.to_numeric(balances["start_balance"], errors="coerce")
        series = pd.Series(values.to_numpy(), index=index)
    else:
        series = pd.Series(dtype=float) if balances is None else balances.copy()
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        if series.index.name != "employee_id":
            series.index = series.index.astype(str)
        series.index = series.index.map(lambda x: str(x).strip())
        series = pd.to_numeric(series, errors="coerce")

    if series.empty:
        return pd.DataFrame(columns=["c_under", "c_over"], index=series.index)

    values = series.fillna(0.0).to_numpy(dtype=float)

    if np.allclose(values, values[0]):
        coeffs = pd.DataFrame(1.5, index=series.index, columns=["c_under", "c_over"])
        return coeffs

    q_low = float(np.nanquantile(values, p_low))
    q_high = float(np.nanquantile(values, p_high))

    if q_high - q_low < min_range:
        median = float(np.nanmedian(values))
        half_range = max(min_range / 2.0, 0.0)
        q_low = median - half_range
        q_high = median + half_range

    q_low = max(q_low, abs_min)
    q_high = min(q_high, abs_max)
    if q_high <= q_low:
        q_high = q_low + 1.0

    clamped = np.clip(values, q_low, q_high)
    scale = q_high - q_low
    scaled = (clamped - q_low) / scale
    scaled = np.clip(scaled, 0.0, 1.0)

    c_under = 2.0 - scaled
    c_over = 1.0 + scaled

    coeffs = pd.DataFrame(
        {"c_under": c_under, "c_over": c_over}, index=series.index
    )
    return coeffs


def _normalize_roles(series: pd.Series) -> pd.Series:
    cleaned = (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return cleaned[(cleaned != "") & (cleaned != "NAN")]


def _build_absence_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if "employee_id" not in df.columns:
        return None

    work = df.copy()
    date_col = next((col for col in ("slot_date", "date", "data") if col in work.columns), None)
    if date_col is None:
        return None

    work["employee_id"] = work["employee_id"].astype(str).str.strip()
    work["slot_date"] = pd.to_datetime(work[date_col]).dt.date

    mask = pd.Series(True, index=work.index)
    if "kind" in work.columns:
        kind_series = work["kind"].astype(str).str.strip().str.lower()
        mask &= kind_series.isin({"full_day", "full-day", "full"})
    elif "tipo" in work.columns:
        tipo_series = work["tipo"].astype(str).str.strip().str.lower()
        mask &= tipo_series.isin({"full_day", "full-day"})
    elif "is_absent" in work.columns:
        abs_series = work["is_absent"].astype(str).str.strip().str.lower()
        mask &= abs_series.isin({"1", "true", "t", "yes", "y", "si"})

    filtered = work.loc[mask, ["employee_id", "slot_date"]].drop_duplicates()
    if filtered.empty:
        return None
    return filtered.reset_index(drop=True)


def _pick_frame(store: dict, *keys: str) -> pd.DataFrame | None:
    """Return the first non-None dataframe found in ``store`` for ``keys``."""

    for key in keys:
        value = store.get(key)
        if value is not None:
            return value
    return None


def _parse_horizon_start(cfg: dict) -> date | None:
    horizon = cfg.get("horizon") if isinstance(cfg, dict) else None
    if not isinstance(horizon, dict):
        return None
    start_raw = horizon.get("start_date")
    if not start_raw:
        return None
    try:
        return pd.to_datetime(start_raw).date()
    except Exception:
        return None


def _resolve_rest_threshold(cfg: dict) -> float:
    rest_cfg = cfg.get("rest_rules") if isinstance(cfg, dict) else None
    if not isinstance(rest_cfg, dict):
        rest_cfg = {}
    value = rest_cfg.get("min_between_shifts_h", 11)
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 11.0
    return max(threshold, 0.0)


def _build_month_to_date_summary(
    cfg: dict,
    employees: pd.DataFrame,
    history: pd.DataFrame | None,
    leaves_days: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "employee_id",
        "window_start_date",
        "window_end_date",
        "hours_worked_h",
        "absence_hours_h",
        "hours_with_leaves_h",
        "night_shifts_count",
        "rest11_exceptions_count",
    ]

    horizon_start = _parse_horizon_start(cfg)
    if horizon_start is None or horizon_start.day == 1:
        return pd.DataFrame(columns=columns)

    month_start = horizon_start.replace(day=1)
    window_end = horizon_start - timedelta(days=1)
    if window_end < month_start:
        return pd.DataFrame(columns=columns)

    if employees is None or employees.empty:
        return pd.DataFrame(columns=columns)

    employee_ids = (
        employees["employee_id"].astype(str).str.strip().replace({"": pd.NA}).dropna().unique()
    )
    employee_ids = sorted(employee_ids.tolist())
    if not employee_ids:
        return pd.DataFrame(columns=columns)

    night_codes = set()
    shift_types = cfg.get("shift_types") if isinstance(cfg, dict) else None
    if isinstance(shift_types, dict):
        raw_codes = shift_types.get("night_codes", [])
        night_codes = {
            str(code).strip().upper()
            for code in raw_codes
            if isinstance(code, (str, int, float)) and str(code).strip()
        }

    history_window = pd.DataFrame()
    if history is not None and not history.empty:
        if {"employee_id", "shift_duration_min"}.issubset(history.columns):
            hist = history.copy()
            if "data_dt" in hist.columns:
                hist_dates = pd.to_datetime(hist["data_dt"], errors="coerce").dt.date
            elif "data" in hist.columns:
                hist_dates = pd.to_datetime(hist["data"], errors="coerce").dt.date
            else:
                hist_dates = pd.Series(pd.NaT, index=hist.index)

            hist["_day"] = hist_dates
            mask = hist["_day"].ge(month_start) & hist["_day"].lt(horizon_start)
            history_window = hist.loc[mask].copy()

    if history_window.empty:
        history_window = pd.DataFrame(columns=["employee_id", "shift_duration_min", "turno"])

    if "employee_id" not in history_window.columns:
        history_window["employee_id"] = pd.Series(index=history_window.index, dtype=str)
    history_window["employee_id"] = history_window["employee_id"].astype(str).str.strip()

    if "shift_duration_min" not in history_window.columns:
        history_window["shift_duration_min"] = 0.0
    history_window["shift_duration_min"] = pd.to_numeric(
        history_window["shift_duration_min"], errors="coerce"
    ).fillna(0.0)

    if "turno" not in history_window.columns:
        history_window["turno"] = pd.Series(index=history_window.index, dtype=str)
    history_window["turno"] = (
        history_window["turno"].astype(str).str.strip().str.upper()
    )

    worked_minutes = (
        history_window.groupby("employee_id")["shift_duration_min"].sum()
        if not history_window.empty
        else pd.Series(dtype=float)
    )
    worked_hours = (worked_minutes / 60.0).to_dict()

    night_counts = {}
    if night_codes and not history_window.empty and "turno" in history_window.columns:
        night_counts = (
            history_window.loc[history_window["turno"].isin(night_codes)]
            .groupby("employee_id")["turno"]
            .count()
            .to_dict()
        )

    rest_threshold = _resolve_rest_threshold(cfg)
    rest_counts: dict[str, int] = {}
    if rest_threshold > 0 and not history_window.empty:
        required_cols = {"shift_start_dt", "shift_end_dt"}
        if required_cols.issubset(history_window.columns):
            working = history_window.loc[history_window["shift_duration_min"] > 0].copy()
            working = working.dropna(subset=list(required_cols))
            if not working.empty:
                working["shift_start_dt"] = pd.to_datetime(
                    working["shift_start_dt"], errors="coerce"
                )
                working["shift_end_dt"] = pd.to_datetime(
                    working["shift_end_dt"], errors="coerce"
                )
                working = working.dropna(subset=list(required_cols))
                working = working.sort_values(["employee_id", "shift_start_dt"])
                for emp_id, grp in working.groupby("employee_id"):
                    prev_end = None
                    count = 0
                    for row in grp.itertuples():
                        start = row.shift_start_dt
                        end = row.shift_end_dt
                        if pd.isna(start) or pd.isna(end):
                            continue
                        if prev_end is not None:
                            rest_hours = (start - prev_end).total_seconds() / 3600.0
                            if rest_hours < rest_threshold:
                                count += 1
                        prev_end = end
                    rest_counts[emp_id] = count

    absence_hours = {}
    if leaves_days is not None and not leaves_days.empty:
        if "employee_id" in leaves_days.columns and "absence_hours_h" in leaves_days.columns:
            leaves = leaves_days.copy()
            if "data_dt" in leaves.columns:
                leave_dates = pd.to_datetime(leaves["data_dt"], errors="coerce").dt.date
            elif "data" in leaves.columns:
                leave_dates = pd.to_datetime(leaves["data"], errors="coerce").dt.date
            else:
                leave_dates = pd.Series(pd.NaT, index=leaves.index)

            leaves["_day"] = leave_dates
            mask = leaves["_day"].ge(month_start) & leaves["_day"].lt(horizon_start)
            filtered = leaves.loc[mask].copy()
            if not filtered.empty:
                if "is_absent" in filtered.columns:
                    is_abs = filtered["is_absent"]
                    if str(is_abs.dtype) == "boolean":
                        mask_abs = is_abs.fillna(False)
                    else:
                        mask_abs = (
                            is_abs.astype(str)
                            .str.strip()
                            .str.lower()
                            .isin({"true", "1", "t", "yes", "y"})
                        )
                    filtered = filtered[mask_abs]
                filtered["absence_hours_h"] = pd.to_numeric(
                    filtered["absence_hours_h"], errors="coerce"
                ).fillna(0.0)
                absence_hours = (
                    filtered.groupby("employee_id")["absence_hours_h"].sum().to_dict()
                )

    summary = pd.DataFrame({"employee_id": employee_ids})
    summary["window_start_date"] = pd.to_datetime(month_start)
    summary["window_end_date"] = pd.to_datetime(window_end)
    summary["hours_worked_h"] = summary["employee_id"].map(worked_hours).fillna(0.0)
    summary["absence_hours_h"] = summary["employee_id"].map(absence_hours).fillna(0.0)
    summary["hours_with_leaves_h"] = summary["hours_worked_h"] + summary["absence_hours_h"]
    summary["night_shifts_count"] = (
        summary["employee_id"].map(night_counts).fillna(0).astype(int)
    )
    summary["rest11_exceptions_count"] = (
        summary["employee_id"].map(rest_counts).fillna(0).astype(int)
    )

    return summary[columns]


def build_all(dfs: dict, cfg: dict) -> dict:
    """Crea dizionari e strutture dati derivate dai DataFrame principali."""
    bundle = {}

    # Estraggo i DataFrame principali
    df_employees = _pick_frame(dfs, "employees", "employees_df")
    df_slots = _pick_frame(dfs, "shift_slots", "shift_slots_df")
    df_elig = _pick_frame(dfs, "shift_role_eligibility", "shift_role_eligibility_df")
    df_pools = _pick_frame(dfs, "role_dept_pools", "role_dept_pools_df")
    df_locks = _pick_frame(dfs, "locks", "locks_df")
    df_abs = _pick_frame(dfs, "absences", "leaves_days_df", "leaves_df")
    df_availability = _pick_frame(dfs, "availability", "availability_df")
    df_role_requirements = _pick_frame(dfs, "groups_role_min_expanded")
    df_history = _pick_frame(dfs, "history", "history_df")
    df_leaves_days = _pick_frame(dfs, "leaves_days", "leaves_days_df")
    df_preassignments = _pick_frame(dfs, "preassignments", "preassignments_df")

    if df_employees is None or df_slots is None:
        raise ValueError("Mancano DataFrame essenziali: employees o shift_slots")
    df_slots = df_slots.copy()
    if "is_night" not in df_slots.columns:
        df_slots["is_night"] = False
    if "can_work_night" not in df_employees.columns:
        df_employees = df_employees.copy()
        df_employees["can_work_night"] = True
    if "pool_id" not in df_employees.columns:
        df_employees = df_employees.copy()
        df_employees["pool_id"] = ""

    if "date" not in df_slots.columns:
        if "data" in df_slots.columns:
            df_slots["date"] = pd.to_datetime(df_slots["data"]).dt.date
        elif "start_dt" in df_slots.columns:
            df_slots["date"] = pd.to_datetime(df_slots["start_dt"]).dt.date
        else:
            raise ValueError("shift_slots: impossibile derivare la colonna 'date'")
    else:
        df_slots["date"] = pd.to_datetime(df_slots["date"]).dt.date

    # (1) Indici contigui -----------------------------------------------
    # Dipendenti
    employee_ids = sorted(df_employees["employee_id"].unique())
    eid_of = {eid: i for i, eid in enumerate(employee_ids)}
    emp_of = {i: eid for eid, i in eid_of.items()}
    df_employees["employee_id2"] = df_employees["employee_id"].map(eid_of)

    # Slot
    slot_ids = sorted(df_slots["slot_id"].unique())
    sid_of = {sid: i for i, sid in enumerate(slot_ids)}
    slot_of = {i: sid for sid, i in sid_of.items()}
    df_slots["slot_id2"] = df_slots["slot_id"].map(sid_of)

    # Giorni (dal calendario esteso, non dagli slot)
    df_cal = dfs.get("calendar_df") 
    if df_cal is None:
        raise ValueError("Manca il DataFrame di calendario (calendar_df).")

    day_values = sorted(pd.to_datetime(df_cal["data"]).dt.date.unique())

    did_of = {d: i for i, d in enumerate(day_values)}
    date_of = {i: d for d, i in did_of.items()}

    # Aggiungi al bundle
    bundle.update({
        "eid_of": eid_of,
        "emp_of": emp_of,
        "sid_of": sid_of,
        "slot_of": slot_of,
        "did_of": did_of,
        "date_of": date_of,
        "num_employees": len(eid_of),
        "num_slots": len(sid_of),
        "num_days": len(did_of),
    })

    # (2) Mappe slot (date, reparti, turni, ecc.)

    # Mappe slot: date, reparto, shift, durata (dopo override già nel DF)
    slot_date = df_slots.set_index("slot_id")["date"].to_dict()
    slot_reparto = df_slots.set_index("slot_id")["reparto_id"].to_dict()
    slot_shiftcode = df_slots.set_index("slot_id")["shift_code"].to_dict()
    slot_duration_min = df_slots.set_index("slot_id")["duration_min"].astype(int).to_dict()

    # Versione indicizzata (usa slot_id2 -> day_id)
    slot_date2 = {sid_of[sid]: did_of[pd.to_datetime(dt).date()]
                for sid, dt in slot_date.items()}

    bundle.update({
        "slot_date": slot_date,
        "slot_reparto": slot_reparto,
        "slot_shiftcode": slot_shiftcode,
        "slot_duration_min": slot_duration_min,
        "slot_date2": slot_date2,
    })

    # Giorni toccati da ciascuno slot (utile per notti/SN)
    slot_days_touched = {}
    for _, row in df_slots.iterrows():
        sid = row["slot_id"]
        start_source = row.get("start_datetime", row.get("start_dt"))
        end_source = row.get("end_datetime", row.get("end_dt"))
        if pd.isna(start_source) or pd.isna(end_source):
            raise ValueError(
                f"shift_slots: valori start/end mancanti per slot_id {sid}"
            )
        start = pd.to_datetime(start_source)
        end = pd.to_datetime(end_source)
        days = pd.date_range(start.normalize(), end.normalize()).date
        slot_days_touched[sid] = [did_of[d] for d in days if d in did_of]

    bundle["slot_days_touched"] = slot_days_touched

    preassignment_pairs: list[tuple[int, int, str]] = []
    if df_preassignments is not None and not df_preassignments.empty:
        work = df_preassignments.copy()
        if "employee_id" in work.columns and "state_code" in work.columns:
            work["employee_id"] = work["employee_id"].astype(str).str.strip()
            work["state_code"] = work["state_code"].astype(str).str.strip().str.upper()
            if "date" in work.columns:
                dates = pd.to_datetime(work["date"], errors="coerce")
            elif "data_dt" in work.columns:
                dates = pd.to_datetime(work["data_dt"], errors="coerce")
            else:
                dates = pd.Series([pd.NaT] * len(work))
            work = work.assign(_date=dates.dt.date)
            work = work.loc[work["_date"].notna()]
            work = work.loc[
                work["state_code"].ne("")
                & work["employee_id"].ne("")
            ]
            work = work.loc[
                work["employee_id"].isin(df_employees["employee_id"].astype(str).str.strip())
            ]

            seen_keys: set[tuple[int, int]] = set()
            for employee_id, day_value, state in work.loc[
                :, ["employee_id", "_date", "state_code"]
            ].itertuples(index=False):
                emp_idx = eid_of.get(employee_id)
                day_idx = did_of.get(day_value)
                if emp_idx is None or day_idx is None:
                    continue
                key = (emp_idx, day_idx)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                preassignment_pairs.append((emp_idx, day_idx, state))

    bundle["preassignment_pairs"] = preassignment_pairs


    # (3) Idoneita e coverage

    role_column = "role" if "role" in df_employees.columns else "ruolo"
    if role_column not in df_employees.columns:
        raise ValueError("preprocessing: colonna ruolo mancante in df_employees")

    slot_base = df_slots.loc[
        :, ["slot_id", "slot_id2", "reparto_id", "shift_code", "coverage_code", "date", "is_night"]
    ].copy()
    slot_base = slot_base.rename(columns={"reparto_id": "slot_reparto_id"})
    slot_base["slot_reparto_id"] = (
        slot_base["slot_reparto_id"].astype(str).str.strip().str.upper()
    )
    slot_base["shift_code"] = slot_base["shift_code"].astype(str).str.strip().str.upper()
    slot_base["coverage_code"] = (
        slot_base["coverage_code"].astype(str).str.strip().str.upper()
    )
    slot_base["slot_date"] = pd.to_datetime(slot_base["date"]).dt.date
    slot_base["is_night"] = slot_base["is_night"].fillna(False).astype(bool)

    allowed_roles = None
    if df_elig is not None and not df_elig.empty:
        allowed_roles = (
            df_elig.loc[df_elig["allowed"] == True, ["shift_code", "role"]]
            .assign(
                shift_code=lambda df: df["shift_code"].astype(str).str.strip().str.upper(),
                role=lambda df: df["role"].astype(str).str.strip().str.upper(),
            )
            .drop_duplicates()
        )

    slot_role: pd.DataFrame
    if df_role_requirements is not None and not df_role_requirements.empty:
        role_req = df_role_requirements.loc[
            :, ["data", "reparto_id", "shift_code", "coverage_code", "role"]
        ].copy()
        role_req["slot_date"] = pd.to_datetime(role_req["data"]).dt.date
        role_req["reparto_id"] = role_req["reparto_id"].astype(str).str.strip().str.upper()
        role_req["shift_code"] = role_req["shift_code"].astype(str).str.strip().str.upper()
        role_req["coverage_code"] = role_req["coverage_code"].astype(str).str.strip().str.upper()
        role_req["role"] = role_req["role"].astype(str).str.strip().str.upper()
        role_req = role_req.drop_duplicates(
            subset=["slot_date", "reparto_id", "shift_code", "coverage_code", "role"]
        )

        slot_role = slot_base.merge(
            role_req,
            left_on=["slot_date", "slot_reparto_id", "shift_code", "coverage_code"],
            right_on=["slot_date", "reparto_id", "shift_code", "coverage_code"],
            how="inner",
        ).drop(columns=["reparto_id"])

        if allowed_roles is not None and not allowed_roles.empty:
            slot_role = slot_role.merge(
                allowed_roles, on=["shift_code", "role"], how="inner"
            )
    else:
        if allowed_roles is not None and not allowed_roles.empty:
            slot_role = slot_base.merge(allowed_roles, on="shift_code", how="inner")
        else:
            roles = _normalize_roles(df_employees[role_column])
            slot_role = slot_base.assign(key=1).merge(
                pd.DataFrame({"role": roles.unique(), "key": 1}),
                on="key",
                how="inner",
            ).drop(columns="key")

    slot_role["role"] = slot_role["role"].astype(str).str.strip().str.upper()
    slot_role = slot_role.drop_duplicates(
        subset=["slot_id", "slot_id2", "role"]
    )

    emp_cols = ["employee_id", "employee_id2", role_column, "reparto_id", "pool_id"]
    if "can_work_night" in df_employees.columns:
        emp_cols.append("can_work_night")

    emp_base = df_employees.loc[:, emp_cols].copy()
    emp_base = emp_base.rename(columns={role_column: "role", "reparto_id": "employee_reparto_id"})
    emp_base["role"] = emp_base["role"].astype(str).str.strip().str.upper()
    emp_base["employee_reparto_id"] = (
        emp_base["employee_reparto_id"].astype(str).str.strip().str.upper()
    )
    emp_base["pool_id"] = emp_base["pool_id"].fillna("").astype(str).str.strip()

    # 1) dipendenti del reparto dello slot
    in_reparto_candidates = slot_role.merge(
        emp_base,
        left_on=["role", "slot_reparto_id"],
        right_on=["role", "employee_reparto_id"],
        how="inner",
    )

    # 2) dipendenti abilitati tramite pool
    cross_candidates = pd.DataFrame(columns=in_reparto_candidates.columns)
    if df_pools is not None and not df_pools.empty:
        pool_tbl = df_pools.loc[:, ["role", "pool_id", "reparto_id"]].copy()
        pool_tbl["role"] = pool_tbl["role"].astype(str).str.strip().str.upper()
        pool_tbl["pool_id"] = pool_tbl["pool_id"].fillna("").astype(str).str.strip()
        pool_tbl["reparto_id"] = pool_tbl["reparto_id"].astype(str).str.strip().str.upper()
        pool_tbl = pool_tbl.drop_duplicates()

        cross_slot = slot_role.merge(
            pool_tbl,
            left_on=["role", "slot_reparto_id"],
            right_on=["role", "reparto_id"],
            how="inner",
        ).drop(columns=["reparto_id"])

        cross_candidates = cross_slot.merge(
            emp_base.loc[emp_base["pool_id"] != ""],
            on=["role", "pool_id"],
            how="inner",
        )
        cross_candidates = cross_candidates[
            cross_candidates["slot_reparto_id"] != cross_candidates["employee_reparto_id"]
        ]

    candidates = pd.concat([in_reparto_candidates, cross_candidates], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["employee_id2", "slot_id2"])

    if df_locks is not None and not df_locks.empty:
        forbidden_tbl = df_locks.loc[
            df_locks["lock"].astype(int) == -1, ["employee_id", "slot_id"]
        ].drop_duplicates()
        candidates = candidates.merge(
            forbidden_tbl,
            on=["employee_id", "slot_id"],
            how="left",
            indicator="_forbid",
        )
        candidates = candidates[candidates["_forbid"] != "both"].drop(columns="_forbid")

    absence_tbl = _build_absence_index(df_abs)
    if absence_tbl is not None and not absence_tbl.empty:
        candidates = candidates.merge(
            absence_tbl,
            on=["employee_id", "slot_date"],
            how="left",
            indicator="_abs",
        )
        candidates = candidates[candidates["_abs"] != "both"].drop(columns="_abs")

    if df_availability is not None and not df_availability.empty:
        availability_tbl = (
            df_availability.loc[:, ["employee_id", "turno", "data"]]
            .assign(
                employee_id=lambda df: df["employee_id"].astype(str).str.strip(),
                turno=lambda df: df["turno"].astype(str).str.strip().str.upper(),
                slot_date=lambda df: pd.to_datetime(df["data"]).dt.date,
            )
            .drop(columns="data")
            .drop_duplicates()
        )
        candidates = candidates.merge(
            availability_tbl,
            left_on=["employee_id", "slot_date", "shift_code"],
            right_on=["employee_id", "slot_date", "turno"],
            how="left",
            indicator="_availability",
        )
        candidates = candidates[candidates["_availability"] != "both"]
        drop_cols = [col for col in ("_availability", "turno") if col in candidates.columns]
        if drop_cols:
            candidates = candidates.drop(columns=drop_cols)

    if "is_night" in candidates.columns:
        night_mask = (
            candidates["is_night"].astype("boolean", copy=False).fillna(False).astype(bool)
        )
        if "can_work_night" in candidates.columns:
            can_night = candidates["can_work_night"].fillna(True).astype(bool)
            candidates = candidates[~(night_mask & ~can_night)]
        else:
            candidates = candidates[~night_mask]

    eligible_sids = {eid_of[eid]: [] for eid in df_employees["employee_id"].unique()}
    eligible_eids = {sid_of[sid]: [] for sid in df_slots["slot_id"].unique()}

    if not candidates.empty:
        for e2, slot_ids in candidates.groupby("employee_id2")["slot_id2"]:
            eligible_sids[e2] = slot_ids.tolist()
        for sid2, employee_ids in candidates.groupby("slot_id2")["employee_id2"]:
            eligible_eids[sid2] = employee_ids.tolist()

    bundle.update(
        {
            "eligible_sids": eligible_sids,
            "eligible_eids": eligible_eids,
        }
    )

    bundle["history_month_to_date"] = _build_month_to_date_summary(
        cfg, df_employees, df_history, df_leaves_days
    )

    # (4) Altri set utili (assenze, festivi, ecc.)

    return bundle
