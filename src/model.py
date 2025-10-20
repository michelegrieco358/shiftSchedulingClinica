from __future__ import annotations

from bisect import bisect_left, bisect_right
from calendar import monthrange
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import math
from typing import Any, Dict, Iterable, List, Mapping

DUE_HOUR_OBJECTIVE_SCALE = 1000
CONSECUTIVE_NIGHT_OBJECTIVE_SCALE = 1000
SINGLE_NIGHT_RECOVERY_OBJECTIVE_SCALE = 1000
REST11_OBJECTIVE_SCALE = 1000
WEEKLY_REST_OBJECTIVE_SCALE = 1000
CROSS_ASSIGNMENT_OBJECTIVE_SCALE = 1000
NIGHT_FAIRNESS_OBJECTIVE_SCALE = 10
NIGHT_FAIRNESS_WEIGHT_SCALE = 100
WEEKEND_FAIRNESS_OBJECTIVE_SCALE = 10
WEEKEND_FAIRNESS_WEIGHT_SCALE = 100
PREASSIGNMENT_OBJECTIVE_SCALE = 1000

import pandas as pd

from .preprocessing import compute_adaptive_coefficients

try:  # pragma: no cover - optional dependency guard
    from ortools.sat.python import cp_model as _cp_model
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for tests without ortools
    class _MissingCpModelModule:
        _IMPORT_ERROR = exc

        class CpModel:  # type: ignore[empty-body]
            pass

        class IntVar:  # type: ignore[empty-body]
            pass

    cp_model = _MissingCpModelModule()
else:  # pragma: no cover - executed when ortools is available
    cp_model = _cp_model
    setattr(cp_model, "_IMPORT_ERROR", None)


@dataclass(frozen=True)
class ModelContext:
    """Raccoglie i DataFrame e gli indici necessari al modello CP-SAT."""

    cfg: dict
    employees: pd.DataFrame
    slots: pd.DataFrame
    coverage_roles: pd.DataFrame
    coverage_totals: pd.DataFrame
    slot_requirements: pd.DataFrame
    availability: pd.DataFrame
    leaves: pd.DataFrame
    history: pd.DataFrame
    locks_must: pd.DataFrame
    locks_forbid: pd.DataFrame
    gap_pairs: pd.DataFrame
    calendars: pd.DataFrame
    preassignments: pd.DataFrame
    bundle: Mapping[str, object]

    def employees_for_slot(self, slot_id: int) -> Iterable[str]:
        """Restituisce gli employee_id candidati per lo slot indicato."""
        sid_map: Dict[int, int] = self.bundle["sid_of"]
        eligible_by_slot: Dict[int, Iterable[int]] = self.bundle["eligible_eids"]
        emp_map: Dict[int, str] = self.bundle["emp_of"]
        slot_idx = sid_map[slot_id]
        return (emp_map[eid_idx] for eid_idx in eligible_by_slot[slot_idx])



@dataclass(frozen=True)
class ObjectiveTermContribution:
    component: str
    var: cp_model.IntVar
    coeff: int
    is_complement: bool = False
    label: str | None = None
    unit_scale: float | None = None


@dataclass
class ModelArtifacts:
    """Colleziona riferimenti alle variabili e agli indici usati in solver."""
    model: cp_model.CpModel
    assign_vars: Dict[tuple[int, int], cp_model.IntVar]
    state_vars: Dict[tuple[int, int, str], cp_model.IntVar]
    employee_index: Mapping[str, int]
    slot_index: Mapping[int, int]
    day_index: Mapping[object, int]
    state_codes: tuple[str, ...]
    monthly_hour_balance: Dict[tuple[int, str], cp_model.IntVar]
    monthly_hour_under_slack: Dict[tuple[int, str], cp_model.IntVar]
    monthly_hour_over_slack: Dict[tuple[int, str], cp_model.IntVar]
    monthly_hour_under_effective: Dict[tuple[int, str], cp_model.IntVar]
    monthly_hour_over_effective: Dict[tuple[int, str], cp_model.IntVar]
    final_hour_balance: Dict[tuple[int, str], cp_model.IntVar]
    final_hour_balance_abs: Dict[tuple[int, str], cp_model.IntVar]
    night_assignment_by_day: Dict[tuple[int, int], cp_model.BoolVar]
    consecutive_night_penalties: Dict[tuple[int, int], cp_model.IntVar]
    consecutive_night_penalty_totals: Dict[int, cp_model.IntVar]
    consecutive_night_penalty_weight: float
    single_night_recovery_penalties: Dict[tuple[int, int], cp_model.BoolVar]
    single_night_recovery_penalty_weight: float
    rest11_penalty_weight: float
    rest_violation_pairs: Dict[tuple[int, int, int], cp_model.BoolVar]
    rest_violation_by_day: Dict[tuple[int, int], cp_model.BoolVar]
    rest_violation_month_totals: Dict[tuple[int, str], cp_model.IntVar]
    rest_violation_streaks: Dict[tuple[int, int], cp_model.IntVar]
    rest_day_flags: Dict[tuple[int, int], cp_model.BoolVar]
    weekly_rest_violations: Dict[tuple[int, int], cp_model.BoolVar]
    weekly_rest_penalty_weight: float
    objective_terms: tuple[ObjectiveTermContribution, ...] = tuple()


def build_model(context: ModelContext) -> ModelArtifacts:
    """Istanzia il modello CP-SAT e crea le variabili base di assegnazione."""
    import_error = getattr(cp_model, "_IMPORT_ERROR", None)
    if import_error is not None:  # pragma: no cover - guard for environments without ortools
        raise RuntimeError(
            "ortools è richiesto per costruire il modello CP-SAT. "
            "Installare il pacchetto 'ortools' per usare il solver."
        ) from import_error

    model = cp_model.CpModel()

    bundle = context.bundle
    eid_of: Mapping[str, int] = bundle["eid_of"]
    sid_of: Mapping[int, int] = bundle["sid_of"]
    did_of: Mapping[object, int] = bundle["did_of"]
    eligible_eids: Mapping[int, Iterable[int]] = bundle["eligible_eids"]

    assign_vars: Dict[tuple[int, int], cp_model.IntVar] = {}
    state_vars: Dict[tuple[int, int, str], cp_model.IntVar] = {}

    raw_states = context.cfg.get("state_codes") if isinstance(context.cfg, Mapping) else None
    if raw_states:
        state_codes = tuple(
            str(code).strip().upper()
            for code in raw_states
            if str(code).strip()
        )
    else:
        state_codes = ("M", "P", "N", "SN", "R", "F")

    num_employees = int(bundle.get("num_employees", len(eid_of)))
    num_days = int(bundle.get("num_days", len(did_of)))

    for slot_id in context.slots["slot_id"]:
        slot_idx = sid_of[slot_id]
        for emp_idx in eligible_eids[slot_idx]:
            var_name = f"x_e{emp_idx}_s{slot_idx}"
            assign_vars[(emp_idx, slot_idx)] = model.NewBoolVar(var_name)

    for emp_idx in range(num_employees):
        for day_idx in range(num_days):
            for state in state_codes:
                var_name = f"st_e{emp_idx}_d{day_idx}_{state}"
                state_vars[(emp_idx, day_idx, state)] = model.NewBoolVar(var_name)

    for emp_idx in range(num_employees):
        for day_idx in range(num_days):
            vars_for_day = [state_vars[(emp_idx, day_idx, state)] for state in state_codes]
            model.Add(sum(vars_for_day) == 1)

    absence_state = _resolve_absence_state_code(context.cfg, state_codes)
    absence_pairs: set[tuple[int, int]] = set()
    if absence_state is not None:
        absence_pairs = _collect_absence_pairs(context.leaves, eid_of, did_of)
        for emp_idx, day_idx in absence_pairs:
            var = state_vars.get((emp_idx, day_idx, absence_state))
            if var is not None:
                model.Add(var == 1)

    slot_date2 = bundle.get("slot_date2")
    if slot_date2 is None:
        slot_date2 = {}
        if not context.slots.empty and "date" in context.slots.columns:
            for row in context.slots.loc[:, ["slot_id", "date"]].itertuples(index=False):
                slot_idx = sid_of.get(getattr(row, "slot_id"))
                if slot_idx is None:
                    continue
                day_raw = getattr(row, "date")
                day = pd.to_datetime(day_raw, errors="coerce")
                if pd.isna(day):
                    continue
                day_idx = did_of.get(day.date())
                if day_idx is not None:
                    slot_date2[slot_idx] = day_idx

    slot_shiftcode_map: Dict[int, str] = {}
    if not context.slots.empty and "shift_code" in context.slots.columns:
        shift_series = (
            context.slots["shift_code"].astype(str).str.strip().str.upper()
        )
        for slot_id, shift_code in zip(context.slots["slot_id"], shift_series, strict=False):
            slot_idx = sid_of.get(slot_id)
            if slot_idx is not None:
                slot_shiftcode_map[slot_idx] = shift_code

    slots_by_day: Dict[int, list[int]] = {}
    slots_by_day_state: Dict[tuple[int, str], list[int]] = {}
    for slot_idx, day_idx in slot_date2.items():
        slots_by_day.setdefault(day_idx, []).append(slot_idx)
        shift_code = slot_shiftcode_map.get(slot_idx)
        if shift_code:
            slots_by_day_state.setdefault((day_idx, shift_code), []).append(slot_idx)

    shift_states = {state for state in ("M", "P", "N") if state in state_codes}
    for emp_idx in range(num_employees):
        for day_idx in range(num_days):
            for state in shift_states:
                state_var = state_vars[(emp_idx, day_idx, state)]
                slot_indices = slots_by_day_state.get((day_idx, state), [])
                assign_list = [
                    assign_vars[(emp_idx, slot_idx)]
                    for slot_idx in slot_indices
                    if (emp_idx, slot_idx) in assign_vars
                ]
                if not assign_list:
                    model.Add(state_var == 0)
                    continue
                model.Add(sum(assign_list) >= state_var)
                for var in assign_list:
                    model.Add(var <= state_var)

    sn_code = "SN" if "SN" in state_codes else None
    night_codes = _resolve_night_codes(
        context.cfg if isinstance(context.cfg, Mapping) else {}
    )
    day_codes = _resolve_day_codes(
        context.cfg if isinstance(context.cfg, Mapping) else {}
    )
    prev_day_index = _build_previous_day_index(bundle)
    next_day_index = _build_next_day_index(bundle)
    history_prev_night = _collect_prev_night_pairs(context.history, eid_of, did_of)

    restricted_after_night_list: list[str] = [
        state for state in ("M", "F", "R") if state in state_codes
    ]
    for code in day_codes:
        normalized = str(code).strip().upper()
        if (
            normalized
            and normalized in state_codes
            and normalized not in night_codes
            and normalized != sn_code
            and normalized not in restricted_after_night_list
        ):
            restricted_after_night_list.append(normalized)
    restricted_after_night = tuple(restricted_after_night_list)

    if sn_code is not None:
        for emp_idx in range(num_employees):
            for day_idx in range(num_days):
                sn_var = state_vars[(emp_idx, day_idx, sn_code)]

                day_assignments = [
                    assign_vars[(emp_idx, slot_idx)]
                    for slot_idx in slots_by_day.get(day_idx, [])
                    if (emp_idx, slot_idx) in assign_vars
                ]
                for var in day_assignments:
                    model.Add(var <= 1 - sn_var)

                prev_idx = prev_day_index.get(day_idx)
                if prev_idx is not None:
                    prev_n_var = state_vars.get((emp_idx, prev_idx, "N"))
                    if prev_n_var is None:
                        model.Add(sn_var == 0)
                    else:
                        model.Add(sn_var <= prev_n_var)
                else:
                    if (emp_idx, day_idx) not in history_prev_night:
                        model.Add(sn_var == 0)

    if restricted_after_night:
        for emp_idx in range(num_employees):
            for day_idx in range(num_days):
                prev_idx = prev_day_index.get(day_idx)

                if prev_idx is not None:
                    prev_n_var = state_vars.get((emp_idx, prev_idx, "N"))
                    if prev_n_var is None:
                        continue
                    for state in restricted_after_night:
                        state_var = state_vars[(emp_idx, day_idx, state)]
                        model.Add(state_var + prev_n_var <= 1)
                elif (emp_idx, day_idx) in history_prev_night:
                    for state in restricted_after_night:
                        state_var = state_vars[(emp_idx, day_idx, state)]
                        model.Add(state_var == 0)

    _apply_lock_constraints(context, model, assign_vars, bundle)
    _add_cross_assignment_limits(context, model, assign_vars, bundle)

    hour_info = _add_hour_constraints(context, model, assign_vars, bundle)
    night_info = _add_night_constraints(context, model, assign_vars, bundle)
    pattern_info = _add_night_pattern_constraints(
        context,
        model,
        state_vars,
        state_codes,
        bundle,
        prev_day_index,
        next_day_index,
        history_prev_night,
        absence_state,
        absence_pairs,
    )
    rest_info = _add_rest_constraints(context, model, assign_vars, bundle)
    rest_day_info = _add_rest_day_windows(context, model, state_vars, bundle, state_codes)
    rest11_weight = _resolve_rest11_penalty_weight(
        context.cfg if isinstance(context.cfg, Mapping) else None
    )
    weekly_rest_weight = _resolve_weekly_rest_penalty_weight(
        context.cfg if isinstance(context.cfg, Mapping) else None
    )

    objective_terms: List[cp_model.LinearExpr] = []
    objective_metadata: list[ObjectiveTermContribution] = []

    terms, details = _build_due_hour_objective_terms(context, bundle, hour_info)
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    terms, details = _build_final_balance_objective_terms(context, bundle, hour_info)
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    terms, details = _build_consecutive_night_objective_terms(night_info)
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    single_night_info = (
        pattern_info.get("single_night_recovery", {})
        if isinstance(pattern_info, Mapping)
        else {}
    )
    terms, details = _build_single_night_recovery_objective_terms(single_night_info)
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    terms, details = _build_night_fairness_objective_terms(
        model, context, bundle, night_info
    )
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    terms, details = _build_weekend_fairness_objective_terms(
        model, context, bundle, assign_vars
    )
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    terms, details = _build_rest11_objective_terms(
        rest_info["pair_violations"], rest11_weight
    )
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    terms, details = _build_weekly_rest_objective_terms(
        rest_day_info["weekly_violations"], weekly_rest_weight
    )
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    terms, details = _build_cross_assignment_objective_terms(
        context, bundle, assign_vars
    )
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    terms, details = _build_preassignment_objective_terms(
        context, bundle, state_vars
    )
    objective_terms.extend(terms)
    objective_metadata.extend(details)

    if objective_terms:
        model.Minimize(sum(objective_terms))

    single_night_penalties = single_night_info.get("violations", {}) if isinstance(single_night_info, Mapping) else {}
    single_night_weight = float(single_night_info.get("weight", 0.0)) if isinstance(single_night_info, Mapping) else 0.0

    return ModelArtifacts(
        model=model,
        assign_vars=assign_vars,
        state_vars=state_vars,
        employee_index=eid_of,
        slot_index=sid_of,
        day_index=did_of,
        state_codes=state_codes,
        monthly_hour_balance=hour_info["monthly_balance"],
        monthly_hour_under_slack=hour_info["monthly_under"],
        monthly_hour_over_slack=hour_info["monthly_over"],
        monthly_hour_under_effective=hour_info["monthly_under_effective"],
        monthly_hour_over_effective=hour_info["monthly_over_effective"],
        final_hour_balance=hour_info["final_balance"],
        final_hour_balance_abs=hour_info["final_balance_abs"],
        night_assignment_by_day=night_info["night_by_day"],
        consecutive_night_penalties=night_info["extra_penalties"],
        consecutive_night_penalty_totals=night_info["extra_totals"],
        consecutive_night_penalty_weight=float(night_info.get("penalty_weight", 0.0)),
        single_night_recovery_penalties=single_night_penalties,
        single_night_recovery_penalty_weight=single_night_weight,
        rest11_penalty_weight=rest11_weight,
        rest_violation_pairs=rest_info["pair_violations"],
        rest_violation_by_day=rest_info["day_flags"],
        rest_violation_month_totals=rest_info["monthly_totals"],
        rest_violation_streaks=rest_info["streak_vars"],
        rest_day_flags=rest_day_info["rest_day_flags"],
        weekly_rest_violations=rest_day_info["weekly_violations"],
        weekly_rest_penalty_weight=weekly_rest_weight,
        objective_terms=tuple(objective_metadata),
    )


# --- Coperture: per RUOLO e per GRUPPO ---------------------------------------
# --- Vincoli di copertura: per RUOLO e per GRUPPO -----------------------------


def _norm_upper(value) -> str:
    """Normalizza stringhe applicando strip e upper."""
    return str(value).strip().upper()


def _parse_role_set(raw) -> set[str]:
    """Converte un valore generico in un set di ruoli uppercase."""
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {_norm_upper(item) for item in raw if str(item).strip()}
    text = str(raw)
    if "|" in text:
        tokens = text.split("|")
    elif "," in text:
        tokens = text.split(",")
    else:
        tokens = [text]
    return {_norm_upper(token) for token in tokens if token.strip()}


def _resolve_absence_state_code(cfg: Mapping[str, Any] | None, state_codes: Iterable[str]) -> str | None:
    """Determina il codice di stato da utilizzare per le assenze."""

    codes = list(state_codes)
    if not codes:
        return None

    candidates: list[str] = []
    if isinstance(cfg, Mapping):
        direct = cfg.get("absence_state_code")
        if isinstance(direct, str):
            candidates.append(direct)
        states_cfg = cfg.get("states")
        if isinstance(states_cfg, Mapping):
            nested = states_cfg.get("absence_state_code")
            if isinstance(nested, str):
                candidates.append(nested)
            nested_list = states_cfg.get("absence_state_codes")
            if isinstance(nested_list, Iterable) and not isinstance(nested_list, (str, bytes)):
                for item in nested_list:
                    text = str(item).strip()
                    if text:
                        candidates.append(text)
        direct_list = cfg.get("absence_state_codes")
        if isinstance(direct_list, Iterable) and not isinstance(direct_list, (str, bytes)):
            for item in direct_list:
                text = str(item).strip()
                if text:
                    candidates.append(text)

    normalized = [str(item).strip().upper() for item in candidates if str(item).strip()]
    for candidate in normalized:
        if candidate in codes:
            return candidate

    return "F" if "F" in codes else None


def _collect_absence_pairs(
    leaves: pd.DataFrame | None,
    eid_of: Mapping[str, int],
    did_of: Mapping[object, int],
) -> set[tuple[int, int]]:
    """Ricava le coppie (employee_idx, day_idx) assenti da ``leaves``."""

    if leaves is None or leaves.empty:
        return set()
    if "employee_id" not in leaves.columns:
        return set()

    work = leaves.copy()
    work["employee_id"] = work["employee_id"].astype(str).str.strip()

    date_series = None
    for col in ("date", "data", "giorno", "day", "slot_date"):
        if col in work.columns:
            date_series = pd.to_datetime(work[col], errors="coerce").dt.date
            break
    if date_series is None:
        for col in ("date_dt", "data_dt", "giorno_dt", "day_dt"):
            if col in work.columns:
                date_series = pd.to_datetime(work[col], errors="coerce").dt.date
                break

    if date_series is not None:
        work = work.assign(_date=date_series)
    elif {"date_from", "date_to"}.issubset(work.columns):
        records: list[tuple[str, object]] = []
        for row in work.itertuples(index=False):
            start = pd.to_datetime(getattr(row, "date_from"), errors="coerce")
            end = pd.to_datetime(getattr(row, "date_to"), errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            start = start.normalize()
            end = end.normalize()
            if end < start:
                start, end = end, start
            for day in pd.date_range(start, end, freq="D"):
                records.append((getattr(row, "employee_id"), day.date()))
        if not records:
            return set()
        work = pd.DataFrame(records, columns=["employee_id", "_date"])
    else:
        return set()

    if "_date" not in work.columns:
        return set()

    mask = work["_date"].notna()
    if "is_absent" in work.columns:
        mask &= work["is_absent"].astype(bool)
    if "is_leave_day" in work.columns:
        mask &= work["is_leave_day"].astype(bool)
    if "is_in_horizon" in work.columns:
        mask &= work["is_in_horizon"].astype(bool)

    filtered = work.loc[mask, ["employee_id", "_date"]].drop_duplicates()

    result: set[tuple[int, int]] = set()
    for employee_id, day in filtered.itertuples(index=False):
        emp_idx = eid_of.get(employee_id)
        day_idx = did_of.get(day)
        if emp_idx is not None and day_idx is not None:
            result.add((emp_idx, day_idx))
    return result


def _build_previous_day_index(bundle: Mapping[str, object]) -> Dict[int, int | None]:
    """Restituisce la mappa day_idx -> day_idx del giorno precedente (se esiste)."""

    date_of: Mapping[int, object] = bundle.get("date_of", {})  # type: ignore[assignment]
    did_of: Mapping[object, int] = bundle.get("did_of", {})  # type: ignore[assignment]

    prev_map: Dict[int, int | None] = {}
    for day_idx, raw_date in date_of.items():
        day = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(day):
            prev_map[day_idx] = None
            continue
        prev_date = (day - timedelta(days=1)).date()
        prev_map[day_idx] = did_of.get(prev_date)
    return prev_map


def _build_next_day_index(bundle: Mapping[str, object]) -> Dict[int, int | None]:
    """Restituisce la mappa day_idx -> day_idx del giorno successivo (se esiste)."""

    date_of: Mapping[int, object] = bundle.get("date_of", {})  # type: ignore[assignment]
    did_of: Mapping[object, int] = bundle.get("did_of", {})  # type: ignore[assignment]

    next_map: Dict[int, int | None] = {}
    for day_idx, raw_date in date_of.items():
        day = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(day):
            next_map[day_idx] = None
            continue
        next_date = (day + timedelta(days=1)).date()
        next_map[day_idx] = did_of.get(next_date)
    return next_map


def _collect_prev_night_pairs(
    history: pd.DataFrame | None,
    eid_of: Mapping[str, int],
    did_of: Mapping[object, int],
) -> set[tuple[int, int]]:
    """Restituisce le coppie (emp_idx, day_idx) con notte il giorno precedente (da history)."""

    if history is None or history.empty:
        return set()
    required = {"employee_id", "turno", "data"}
    if not required.issubset(history.columns):
        return set()

    work = history.loc[:, ["employee_id", "turno", "data"]].copy()
    work["employee_id"] = work["employee_id"].astype(str).str.strip()
    work["turno"] = work["turno"].astype(str).str.strip().str.upper()
    work["data"] = pd.to_datetime(work["data"], errors="coerce").dt.date

    mask = work["turno"].eq("N") & work["data"].notna()
    if not mask.any():
        return set()

    result: set[tuple[int, int]] = set()
    for employee_id, day in work.loc[mask, ["employee_id", "data"]].itertuples(index=False):
        emp_idx = eid_of.get(employee_id)
        if emp_idx is None:
            continue
        next_day = day + timedelta(days=1)
        day_idx = did_of.get(next_day)
        if day_idx is not None:
            result.add((emp_idx, day_idx))
    return result


def _ensure_state_indicator(
    model: cp_model.CpModel,
    state_vars: Dict[tuple[int, int, str], cp_model.IntVar],
    emp_idx: int,
    day_idx: int,
    codes: Iterable[str],
    cache: dict[tuple[int, int, tuple[str, ...]], cp_model.IntVar],
    prefix: str,
) -> cp_model.IntVar | None:
    """Restituisce una BoolVar che rappresenta l'appartenenza del giorno a ``codes``."""

    normalized_codes = tuple(
        sorted({str(code).strip().upper() for code in codes if str(code).strip()})
    )
    if not normalized_codes:
        return None

    available_codes = tuple(
        code for code in normalized_codes if (emp_idx, day_idx, code) in state_vars
    )
    if not available_codes:
        return None

    key = (emp_idx, day_idx, available_codes)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if len(available_codes) == 1:
        var = state_vars[(emp_idx, day_idx, available_codes[0])]
        cache[key] = var
        return var

    var = model.NewBoolVar(f"{prefix}_e{emp_idx}_d{day_idx}")
    model.Add(
        var
        == sum(state_vars[(emp_idx, day_idx, code)] for code in available_codes)
    )
    cache[key] = var
    return var


def _parse_date_value(value: Any) -> date | None:
    """Converte un generico valore data in ``datetime.date``."""

    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        dt = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    if isinstance(dt, pd.Timestamp):
        if dt.tz is not None:
            dt = dt.tz_convert(None)
        return dt.normalize().date()
    try:
        return pd.Timestamp(dt).normalize().date()
    except Exception:
        return None


def _absences_count_as_worked(cfg: Mapping[str, Any] | None) -> bool:
    """Restituisce True se le assenze valgono come ore lavorate."""

    if not isinstance(cfg, Mapping):
        return True
    defaults = cfg.get("defaults")
    if not isinstance(defaults, Mapping):
        return True
    abs_cfg = defaults.get("absences")
    if not isinstance(abs_cfg, Mapping):
        return True
    value = abs_cfg.get("count_as_worked_hours")
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return True
        return text in {"true", "1", "yes", "y", "si"}
    if value is None:
        return True
    try:
        return bool(value)
    except Exception:
        return True


def _extract_calendar_maps(
    context: ModelContext, bundle: Mapping[str, object]
) -> tuple[
    dict[date, str],
    dict[date, str],
    dict[int, str],
    dict[int, str],
    date | None,
    date | None,
    set[str],
]:
    """Deriva mappe giorno->settimana/mese e le date dell'orizzonte."""

    calendars = context.calendars
    if calendars is None or calendars.empty or "data" not in calendars.columns:
        return {}, {}, {}, {}, None, None, set()

    work = calendars.copy()
    data_dt = pd.to_datetime(work["data"], errors="coerce")
    work = work.assign(_data_dt=data_dt).dropna(subset=["_data_dt"])
    work["_data_dt"] = work["_data_dt"].dt.tz_localize(None)
    work["_data_dt"] = work["_data_dt"].dt.normalize()
    work["data"] = work["_data_dt"].dt.date
    work = work.drop_duplicates(subset=["data"], keep="last")

    horizon_cfg = context.cfg.get("horizon") if isinstance(context.cfg, Mapping) else None
    horizon_start = _parse_date_value(horizon_cfg.get("start_date")) if isinstance(horizon_cfg, Mapping) else None
    horizon_end = _parse_date_value(horizon_cfg.get("end_date")) if isinstance(horizon_cfg, Mapping) else None

    if "is_in_horizon" in work.columns:
        horizon_mask = work["is_in_horizon"].astype(bool)
    elif horizon_start is not None and horizon_end is not None:
        horizon_mask = work["data"].apply(lambda d: horizon_start <= d <= horizon_end)
    elif horizon_start is not None:
        horizon_mask = work["data"].apply(lambda d: d >= horizon_start)
    else:
        horizon_mask = pd.Series(True, index=work.index)

    horizon_days = work.loc[horizon_mask, "data"]
    if horizon_start is None and not horizon_days.empty:
        horizon_start = horizon_days.min()
    if horizon_end is None and not horizon_days.empty:
        horizon_end = horizon_days.max()
    if horizon_start is None and not work.empty:
        horizon_start = work["data"].min()
    if horizon_end is None and not work.empty:
        horizon_end = work["data"].max()

    if "week_start_date" in work.columns:
        week_start_norm = work["week_start_date"].apply(_parse_date_value)
    else:
        week_start_norm = (work["_data_dt"] - pd.to_timedelta(work["_data_dt"].dt.dayofweek, unit="D")).dt.date
    work["week_start_norm"] = week_start_norm

    if "week_id" in work.columns:
        week_id_series = work["week_id"].astype(str).str.strip()
    else:
        week_id_series = pd.Series("", index=work.index)
    fallback_week_id = work["week_start_norm"].apply(lambda d: d.isoformat() if isinstance(d, date) else "")
    week_id_series = week_id_series.where(week_id_series != "", fallback_week_id)
    work["week_id_norm"] = week_id_series

    work["month_id_norm"] = work["data"].apply(lambda d: f"{d:%Y-%m}" if isinstance(d, date) else "")

    week_by_date: dict[date, str] = {}
    month_by_date: dict[date, str] = {}
    for row in work.itertuples(index=False):
        day = getattr(row, "data")
        week_id = getattr(row, "week_id_norm", "")
        month_id = getattr(row, "month_id_norm", "")
        if isinstance(day, date):
            if week_id:
                week_by_date[day] = str(week_id)
            if month_id:
                month_by_date[day] = str(month_id)

    did_of: Mapping[object, int] = bundle.get("did_of", {})  # type: ignore[assignment]
    day_week_map: dict[int, str] = {}
    day_month_map: dict[int, str] = {}
    for raw_day, idx in did_of.items():
        day = _parse_date_value(raw_day)
        if day is None:
            continue
        week_id = week_by_date.get(day)
        month_id = month_by_date.get(day)
        if week_id is not None:
            day_week_map[idx] = week_id
        if month_id is not None:
            day_month_map[idx] = month_id

    horizon_week_ids = {week_by_date.get(day) for day in horizon_days if day in week_by_date}
    horizon_week_ids = {str(week_id) for week_id in horizon_week_ids if week_id}

    return week_by_date, month_by_date, day_week_map, day_month_map, horizon_start, horizon_end, horizon_week_ids


def _pick_float(row: pd.Series, columns: Iterable[str]) -> float | None:
    for col in columns:
        if col in row.index:
            value = row[col]
            if pd.isna(value):
                continue
            text = str(value).strip()
            if not text or text.upper() == "NAN":
                continue
            try:
                return float(value)
            except Exception:
                try:
                    return float(text.replace(",", "."))
                except Exception:
                    continue
    return None


def _pick_float_from_mapping(
    data: Mapping[str, Any] | None, columns: Iterable[str]
) -> float | None:
    if not isinstance(data, Mapping):
        return None
    for col in columns:
        if col not in data:
            continue
        value = data[col]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.upper() == "NAN":
            continue
        try:
            return float(value)
        except Exception:
            try:
                return float(text.replace(",", "."))
            except Exception:
                continue
    return None


def _resolve_night_fairness_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0

    def _extract(mapping: Mapping[str, Any] | None) -> float | None:
        return _pick_float_from_mapping(
            mapping,
            (
                "night_penalty_weight",
                "night_fairness_weight",
                "night_weight",
                "alpha_nights",
            ),
        )

    fairness_cfg = cfg.get("fairness") if isinstance(cfg.get("fairness"), Mapping) else None
    value = _extract(fairness_cfg)
    if value is None:
        defaults = cfg.get("defaults")
        if isinstance(defaults, Mapping):
            value = _extract(defaults.get("fairness") if isinstance(defaults.get("fairness"), Mapping) else None)
    if value is None:
        value = _extract(cfg)
    if value is None:
        return 0.0
    return max(float(value), 0.0)


def _resolve_weekend_fairness_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0

    fairness_cfg = cfg.get("fairness") if isinstance(cfg.get("fairness"), Mapping) else None
    candidates = [
        "weekend_penalty_weight",
        "weekend_fairness_weight",
        "weekend_weight",
        "weekend_gamma",
        "gamma_weekend",
    ]

    value = _pick_float_from_mapping(fairness_cfg, candidates)
    if value is None:
        defaults = cfg.get("defaults")
        if isinstance(defaults, Mapping):
            value = _pick_float_from_mapping(
                defaults.get("fairness") if isinstance(defaults.get("fairness"), Mapping) else None,
                candidates,
            )
    if value is None:
        value = _pick_float_from_mapping(
            cfg,
            [
                "weekend_penalty_weight",
                "weekend_fairness_weight",
                "weekend_gamma",
            ],
        )

    if value is None:
        return 0.0

    return max(float(value), 0.0)


def _extract_employee_hour_params(
    context: ModelContext, bundle: Mapping[str, object]
) -> dict[int, dict[str, int | None]]:
    employees = context.employees
    if employees is None or employees.empty or "employee_id" not in employees.columns:
        return {}

    eid_of: Mapping[str, int] = bundle["eid_of"]
    role_col = next((col for col in ("role", "ruolo") if col in employees.columns), None)
    roles = (
        employees[role_col].astype(str).str.strip().str.upper()
        if role_col is not None
        else pd.Series("", index=employees.index)
    )

    defaults = context.cfg.get("defaults") if isinstance(context.cfg, Mapping) else None
    contract_cfg = (
        defaults.get("contract_hours_by_role_h")
        if isinstance(defaults, Mapping)
        else None
    )
    contract_by_role = {}
    if isinstance(contract_cfg, Mapping):
        contract_by_role = {
            str(role).strip().upper(): float(value)
            for role, value in contract_cfg.items()
            if str(role).strip() and pd.notna(value)
        }

    params: dict[int, dict[str, int | None]] = {}
    for row_idx, row in employees.iterrows():
        emp_id = str(row["employee_id"]).strip()
        if not emp_id:
            continue
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue
        role_u = roles.iloc[row_idx] if row_idx in roles.index else ""

        due_hours = _pick_float(row, [
            "ore_dovute_mese_h",
            "ore_dovute_h",
            "contract_hours_h",
            "contract_hours_month_h",
        ])
        if due_hours is None and role_u and role_u in contract_by_role:
            due_hours = contract_by_role[role_u]

        max_week_hours = _pick_float(row, [
            "max_week_hours_h",
            "max_week_hour_h",
            "week_hours_cap_h",
        ])
        max_month_hours = _pick_float(row, [
            "max_month_hours_h",
            "max_month_hour_h",
            "month_hours_cap_h",
        ])

        max_week_minutes: int | None = None
        if max_week_hours is not None:
            try:
                max_week_minutes = int(round(max_week_hours * 60))
            except (TypeError, ValueError):
                max_week_minutes = None
        else:
            max_week_minutes_raw = _pick_float(
                row,
                [
                    "max_week_minutes",
                    "max_week_min",
                ],
            )
            if max_week_minutes_raw is not None:
                try:
                    max_week_minutes = int(round(max_week_minutes_raw))
                except (TypeError, ValueError):
                    max_week_minutes = None

        max_month_minutes: int | None = None
        if max_month_hours is not None:
            try:
                max_month_minutes = int(round(max_month_hours * 60))
            except (TypeError, ValueError):
                max_month_minutes = None
        else:
            max_month_minutes_raw = _pick_float(
                row,
                [
                    "max_month_minutes",
                    "max_month_min",
                ],
            )
            if max_month_minutes_raw is not None:
                try:
                    max_month_minutes = int(round(max_month_minutes_raw))
                except (TypeError, ValueError):
                    max_month_minutes = None

        max_balance_delta = _pick_float(
            row,
            [
                "max_balance_delta_month_h",
                "balance_delta_month_h",
                "max_month_balance_delta_h",
            ],
        )
        if max_balance_delta is not None:
            try:
                max_balance_delta_minutes = int(round(max_balance_delta * 60))
            except (TypeError, ValueError):
                max_balance_delta_minutes = None
            else:
                if max_balance_delta_minutes <= 0:
                    max_balance_delta_minutes = None
        else:
            max_balance_delta_minutes = None

        params[emp_idx] = {
            "due_minutes": int(round(due_hours * 60)) if due_hours is not None else None,
            "max_week_minutes": max_week_minutes,
            "max_month_minutes": max_month_minutes,
            "max_balance_delta_minutes": max_balance_delta_minutes,
        }

    return params


def _resolve_night_codes(cfg: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(cfg, Mapping):
        return set()
    shift_types = cfg.get("shift_types")
    codes: set[str] = set()
    if isinstance(shift_types, Mapping):
        raw_codes = shift_types.get("night_codes", [])
        if isinstance(raw_codes, str):
            raw_iterable = [raw_codes]
        elif isinstance(raw_codes, Iterable):
            raw_iterable = list(raw_codes)
        else:
            raw_iterable = [raw_codes]
        for code in raw_iterable:
            code_str = str(code).strip().upper()
            if code_str:
                codes.add(code_str)
    if not codes:
        codes.add("N")
    return codes


def _resolve_day_codes(cfg: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(cfg, Mapping):
        return {"M", "P"}
    shift_types = cfg.get("shift_types")
    codes: set[str] = set()
    if isinstance(shift_types, Mapping):
        raw_codes = shift_types.get("day_codes", [])
        if isinstance(raw_codes, str):
            raw_iterable = [raw_codes]
        elif isinstance(raw_codes, Iterable):
            raw_iterable = list(raw_codes)
        else:
            raw_iterable = [raw_codes]
        for code in raw_iterable:
            code_str = str(code).strip().upper()
            if code_str:
                codes.add(code_str)
    if not codes:
        codes.update({"M", "P"})
    return codes


def _extract_employee_night_params(
    context: ModelContext, bundle: Mapping[str, object]
) -> dict[int, dict[str, int | None]]:
    employees = context.employees
    if employees is None or employees.empty or "employee_id" not in employees.columns:
        return {}

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    if not eid_of:
        return {}

    try:
        role_map = _build_employee_role_map(employees, bundle)
    except ValueError:
        role_map = {}

    cfg = context.cfg if isinstance(context.cfg, Mapping) else {}
    defaults = cfg.get("defaults") if isinstance(cfg, Mapping) else None
    default_night_cfg = (
        defaults.get("night") if isinstance(defaults, Mapping) else None
    )
    global_night_cfg = cfg.get("night") if isinstance(cfg, Mapping) else None

    roles_cfg = cfg.get("roles") if isinstance(cfg, Mapping) else None
    role_limits: dict[str, dict[str, float | None]] = {}
    if isinstance(roles_cfg, Mapping):
        for role_name, role_cfg in roles_cfg.items():
            if not isinstance(role_cfg, Mapping):
                continue
            role_u = str(role_name).strip().upper()
            role_night_cfg = role_cfg.get("night") if isinstance(role_cfg.get("night"), Mapping) else None
            week_limit = _pick_float_from_mapping(
                role_night_cfg,
                (
                    "max_per_week",
                    "max_week",
                    "max_week_nights",
                    "max_nights_week",
                ),
            )
            month_limit = _pick_float_from_mapping(
                role_night_cfg,
                (
                    "max_per_month",
                    "max_month",
                    "max_month_nights",
                    "max_nights_month",
                ),
            )
            consecutive_limit = _pick_float_from_mapping(
                role_night_cfg,
                (
                    "max_consecutive_nights",
                    "max_consecutive_night",
                    "max_nights_consecutive",
                    "max_consecutive",
                ),
            )
            if week_limit is None:
                week_limit = _pick_float_from_mapping(
                    role_cfg,
                    (
                        "max_nights_week",
                        "max_week_nights",
                        "max_per_week",
                    ),
                )
            if month_limit is None:
                month_limit = _pick_float_from_mapping(
                    role_cfg,
                    (
                        "max_nights_month",
                        "max_month_nights",
                        "max_per_month",
                    ),
                )
            if consecutive_limit is None:
                consecutive_limit = _pick_float_from_mapping(
                    role_cfg,
                    (
                        "max_consecutive_nights",
                        "max_consecutive_night",
                        "max_nights_consecutive",
                        "max_consecutive",
                    ),
                )
            role_limits[role_u] = {
                "week": week_limit,
                "month": month_limit,
                "consecutive": consecutive_limit,
            }

    params: dict[int, dict[str, int | None]] = {}
    for _, row in employees.iterrows():
        emp_id = str(row.get("employee_id", "")).strip()
        if not emp_id:
            continue
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue

        week_limit = _pick_float(row, [
            "max_nights_week",
            "max_week_nights",
            "max_night_week",
            "max_nights_per_week",
        ])
        month_limit = _pick_float(row, [
            "max_nights_month",
            "max_month_nights",
            "max_night_month",
            "max_nights_per_month",
        ])
        consecutive_limit = _pick_float(row, [
            "max_consecutive_nights",
            "max_consecutive_night",
            "max_nights_consecutive",
            "max_consecutive",
        ])

        role_u = role_map.get(emp_idx, "")
        role_cfg = role_limits.get(role_u)
        if week_limit is None and role_cfg is not None:
            week_limit = role_cfg.get("week")
        if month_limit is None and role_cfg is not None:
            month_limit = role_cfg.get("month")
        if consecutive_limit is None and role_cfg is not None:
            consecutive_limit = role_cfg.get("consecutive")

        if week_limit is None:
            week_limit = _pick_float_from_mapping(
                default_night_cfg,
                (
                    "max_per_week",
                    "max_week",
                    "max_week_nights",
                    "max_nights_week",
                ),
            )
        if month_limit is None:
            month_limit = _pick_float_from_mapping(
                default_night_cfg,
                (
                    "max_per_month",
                    "max_month",
                    "max_month_nights",
                    "max_nights_month",
                ),
            )
        if consecutive_limit is None:
            consecutive_limit = _pick_float_from_mapping(
                default_night_cfg,
                (
                    "max_consecutive_nights",
                    "max_consecutive_night",
                    "max_nights_consecutive",
                    "max_consecutive",
                ),
            )
        if week_limit is None:
            week_limit = _pick_float_from_mapping(
                global_night_cfg,
                (
                    "max_per_week",
                    "max_week",
                    "max_week_nights",
                    "max_nights_week",
                ),
            )
        if month_limit is None:
            month_limit = _pick_float_from_mapping(
                global_night_cfg,
                (
                    "max_per_month",
                    "max_month",
                    "max_month_nights",
                    "max_nights_month",
                ),
            )
        if consecutive_limit is None:
            consecutive_limit = _pick_float_from_mapping(
                global_night_cfg,
                (
                    "max_consecutive_nights",
                    "max_consecutive_night",
                    "max_nights_consecutive",
                    "max_consecutive",
                ),
            )
        week_limit_int = None
        if week_limit is not None and pd.notna(week_limit):
            week_limit_int = max(int(round(float(week_limit))), 0)

        month_limit_int = None
        if month_limit is not None and pd.notna(month_limit):
            month_limit_int = max(int(round(float(month_limit))), 0)

        consecutive_limit_int = None
        if consecutive_limit is not None and pd.notna(consecutive_limit):
            consecutive_limit_int = max(int(round(float(consecutive_limit))), 0)

        params[emp_idx] = {
            "max_week_nights": week_limit_int,
            "max_month_nights": month_limit_int,
            "max_consecutive_nights": consecutive_limit_int,
        }

    return params


def _build_slot_duration_minutes(
    context: ModelContext, bundle: Mapping[str, object]
) -> dict[int, int]:
    sid_of: Mapping[int, int] = bundle.get("sid_of", {})  # type: ignore[assignment]
    duration_map: dict[int, int] = {}

    raw_durations = bundle.get("slot_duration_min")
    if isinstance(raw_durations, Mapping):
        for slot_id, minutes in raw_durations.items():
            try:
                slot_int = int(slot_id)
            except Exception:
                continue
            slot_idx = sid_of.get(slot_int)
            if slot_idx is None:
                continue
            try:
                value = int(round(float(minutes)))
            except Exception:
                value = 0
            duration_map[slot_idx] = max(value, 0)

    if len(duration_map) < len(sid_of):
        if {
            "slot_id",
            "duration_min",
        }.issubset(context.slots.columns):
            for row in context.slots.loc[:, ["slot_id", "duration_min"]].itertuples(index=False):
                slot_idx = sid_of.get(int(getattr(row, "slot_id")))
                if slot_idx is None or slot_idx in duration_map:
                    continue
                try:
                    value = int(round(float(getattr(row, "duration_min"))))
                except Exception:
                    value = 0
                duration_map[slot_idx] = max(value, 0)
        else:
            for slot_id in context.slots.get("slot_id", []):
                slot_idx = sid_of.get(int(slot_id))
                if slot_idx is not None and slot_idx not in duration_map:
                    duration_map[slot_idx] = 0

    return duration_map


def _build_month_history_minutes(
    bundle: Mapping[str, object],
    eid_of: Mapping[str, int],
    count_leaves: bool,
) -> dict[tuple[int, str], int]:
    summary = bundle.get("history_month_to_date")
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return {}

    value_col = "hours_with_leaves_h" if count_leaves and "hours_with_leaves_h" in summary.columns else "hours_worked_h"
    if value_col not in summary.columns:
        return {}

    result: dict[tuple[int, str], int] = {}
    for row in summary.itertuples(index=False):
        emp_id = str(getattr(row, "employee_id", "")).strip()
        if not emp_id:
            continue
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue
        month_start = _parse_date_value(getattr(row, "window_start_date", None))
        if month_start is None:
            continue
        month_id = f"{month_start:%Y-%m}"
        try:
            hours_value = float(getattr(row, value_col))
        except Exception:
            hours_value = 0.0
        minutes = int(round(hours_value * 60))
        result[(emp_idx, month_id)] = result.get((emp_idx, month_id), 0) + max(minutes, 0)

    return result


def _build_month_history_ranges(bundle: Mapping[str, object]) -> dict[str, tuple[date, date]]:
    summary = bundle.get("history_month_to_date")
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return {}

    if not {"window_start_date", "window_end_date"}.issubset(summary.columns):
        return {}

    starts = pd.to_datetime(summary["window_start_date"], errors="coerce").dt.date
    ends = pd.to_datetime(summary["window_end_date"], errors="coerce").dt.date

    result: dict[str, tuple[date, date]] = {}
    for start_date, end_date in zip(starts, ends, strict=False):
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        try:
            month_id = f"{start_date:%Y-%m}"
        except Exception:
            continue
        existing = result.get(month_id)
        if existing is None:
            result[month_id] = (start_date, end_date)
        else:
            result[month_id] = (min(existing[0], start_date), max(existing[1], end_date))

    return result


def _month_bounds_from_id(month_id: str) -> tuple[date, date] | None:
    parts = month_id.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        year = int(parts[0])
        month = int(parts[1])
        last_day = monthrange(year, month)[1]
    except Exception:
        return None
    return date(year, month, 1), date(year, month, last_day)


def _should_enforce_month_due_hours(
    month_id: str,
    horizon_start: date | None,
    horizon_end: date | None,
    history_ranges: Mapping[str, tuple[date, date]],
) -> bool:
    """Return True when the model has full-month coverage for due-hour slack.

    The monthly balance becomes meaningful only if we can account for every day
    of the month either inside the planning horizon or through historical
    records.  When that coverage is incomplete we keep the constraint dormant
    so the solver is not forced to make up for time that falls outside the
    available data window.
    """
    bounds = _month_bounds_from_id(month_id)
    if bounds is None:
        return False

    month_start, month_end = bounds

    if horizon_end is None or horizon_end < month_end:
        return False

    if horizon_start is not None and horizon_start <= month_start:
        return True

    history_range = history_ranges.get(month_id)
    if history_range is None:
        return False

    history_start, history_end = history_range
    if history_start > month_start:
        return False

    if horizon_start is None or horizon_start > month_end:
        return False

    required_history_end = min(month_end, horizon_start - timedelta(days=1))
    return history_end >= required_history_end


def _build_month_history_nights(
    bundle: Mapping[str, object],
    eid_of: Mapping[str, int],
) -> dict[tuple[int, str], int]:
    summary = bundle.get("history_month_to_date")
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return {}
    if "night_shifts_count" not in summary.columns:
        return {}

    result: dict[tuple[int, str], int] = {}
    for row in summary.itertuples(index=False):
        emp_id = str(getattr(row, "employee_id", "")).strip()
        if not emp_id:
            continue
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue
        month_start = _parse_date_value(getattr(row, "window_start_date", None))
        if month_start is None:
            continue
        month_id = f"{month_start:%Y-%m}"
        try:
            count_value = int(round(float(getattr(row, "night_shifts_count", 0))))
        except Exception:
            count_value = 0
        result[(emp_idx, month_id)] = result.get((emp_idx, month_id), 0) + max(count_value, 0)

    return result


def _history_night_dates_by_employee(
    history: pd.DataFrame | None,
    night_codes: set[str],
    eid_of: Mapping[str, int],
    before_day: date | None = None,
) -> dict[int, set[date]]:
    if history is None or history.empty:
        return {}

    if "employee_id" not in history.columns:
        return {}

    shift_col = next(
        (
            col
            for col in ("turno", "shift_code", "shift", "codice")
            if col in history.columns
        ),
        None,
    )
    if shift_col is None:
        return {}

    date_col = next(
        (
            col
            for col in ("data", "date", "giorno", "day")
            if col in history.columns
        ),
        None,
    )
    if date_col is None:
        return {}

    work = history.loc[:, ["employee_id", shift_col, date_col]].copy()
    work["employee_id"] = work["employee_id"].astype(str).str.strip()
    work[shift_col] = work[shift_col].astype(str).str.strip().str.upper()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.date

    mask = work[shift_col].isin({code.upper() for code in night_codes}) & work[date_col].notna()
    if before_day is not None:
        mask &= work[date_col] < before_day

    result: dict[int, set[date]] = defaultdict(set)
    for employee_id, day in work.loc[mask, ["employee_id", date_col]].itertuples(index=False):
        emp_idx = eid_of.get(employee_id)
        if emp_idx is None:
            continue
        result[emp_idx].add(day)
    return result


def _compute_initial_night_streaks(
    history: pd.DataFrame | None,
    eid_of: Mapping[str, int],
    night_codes: set[str],
    first_day: date | None,
) -> dict[int, int]:
    if first_day is None:
        return {}

    night_dates = _history_night_dates_by_employee(history, night_codes, eid_of, before_day=first_day)
    if not night_dates:
        return {}

    result: dict[int, int] = {}
    for emp_idx, dates in night_dates.items():
        if not dates:
            continue
        cursor = first_day - timedelta(days=1)
        streak = 0
        while cursor in dates:
            streak += 1
            cursor -= timedelta(days=1)
        if streak:
            result[emp_idx] = streak
    return result


def _compute_history_minutes_by_week(
    context: ModelContext,
    week_by_date: Mapping[date, str],
    horizon_start: date | None,
    horizon_week_ids: set[str],
    count_leaves: bool,
    eid_of: Mapping[str, int],
) -> dict[tuple[int, str], int]:
    if not week_by_date or not horizon_week_ids:
        return {}

    result: dict[tuple[int, str], int] = {}

    def accumulate(df: pd.DataFrame | None, duration_col: str) -> None:
        if df is None or df.empty or duration_col not in df.columns or "employee_id" not in df.columns:
            return
        work = df.copy()

        day_series = None
        for col in ("data_dt", "date_dt", "giorno_dt", "day_dt"):
            if col in work.columns:
                day_series = pd.to_datetime(work[col], errors="coerce").dt.date
                break
        if day_series is None:
            for col in ("data", "date", "giorno", "day"):
                if col in work.columns:
                    day_series = pd.to_datetime(work[col], errors="coerce").dt.date
                    break
        if day_series is None:
            return

        work = work.assign(_day=day_series).dropna(subset=["_day"])
        if horizon_start is not None:
            work = work[work["_day"] < horizon_start]
        if work.empty:
            return

        work["week_id"] = work["_day"].map(week_by_date)
        work = work[work["week_id"].isin(horizon_week_ids)]
        if work.empty:
            return

        minutes = pd.to_numeric(work[duration_col], errors="coerce").fillna(0.0)
        work = work.assign(_minutes=minutes)

        grouped = work.groupby(["employee_id", "week_id"])["_minutes"].sum()
        for (emp_id, week_id), total in grouped.items():
            emp_idx = eid_of.get(str(emp_id).strip())
            if emp_idx is None:
                continue
            result[(emp_idx, str(week_id))] = result.get((emp_idx, str(week_id)), 0) + int(round(float(total)))

    accumulate(context.history, "shift_duration_min")
    if count_leaves:
        accumulate(context.leaves, "shift_duration_min")

    return result


def _compute_history_nights_by_week(
    context: ModelContext,
    week_by_date: Mapping[date, str],
    horizon_start: date | None,
    horizon_week_ids: set[str],
    night_codes: set[str],
    eid_of: Mapping[str, int],
) -> dict[tuple[int, str], int]:
    if not week_by_date or not horizon_week_ids or not night_codes:
        return {}

    history = context.history
    if history is None or history.empty:
        return {}
    if "employee_id" not in history.columns or "turno" not in history.columns:
        return {}

    work = history.copy()

    day_series = None
    for col in ("data_dt", "date_dt", "giorno_dt", "day_dt"):
        if col in work.columns:
            day_series = pd.to_datetime(work[col], errors="coerce").dt.date
            break
    if day_series is None:
        for col in ("data", "date", "giorno", "day"):
            if col in work.columns:
                day_series = pd.to_datetime(work[col], errors="coerce").dt.date
                break
    if day_series is None:
        return {}

    work = work.assign(_day=day_series).dropna(subset=["_day"])
    if horizon_start is not None:
        work = work[work["_day"] < horizon_start]
    if work.empty:
        return {}

    work["week_id"] = work["_day"].map(week_by_date)
    work = work[work["week_id"].isin(horizon_week_ids)]
    if work.empty:
        return {}

    turni = work["turno"].astype(str).str.strip().str.upper()
    work = work[turni.isin(night_codes)]
    if work.empty:
        return {}

    result: dict[tuple[int, str], int] = {}
    counts = work.groupby(["employee_id", "week_id"]).size()
    for (emp_id, week_id), value in counts.items():
        emp_idx = eid_of.get(str(emp_id).strip())
        if emp_idx is None:
            continue
        result[(emp_idx, str(week_id))] = result.get((emp_idx, str(week_id)), 0) + int(value)

    return result


def _apply_lock_constraints(
    context: ModelContext,
    model: cp_model.CpModel,
    assign_vars: Mapping[tuple[int, int], cp_model.IntVar],
    bundle: Mapping[str, object],
) -> None:
    """Apply hard must/forbid lock constraints on assignment variables."""

    locks_must = getattr(context, "locks_must", pd.DataFrame())
    locks_forbid = getattr(context, "locks_forbid", pd.DataFrame())

    if (locks_must is None or locks_must.empty) and (
        locks_forbid is None or locks_forbid.empty
    ):
        return

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    sid_of: Mapping[int, int] = bundle.get("sid_of", {})  # type: ignore[assignment]

    if locks_must is not None and not locks_must.empty:
        must_df = locks_must.loc[:, ["employee_id", "slot_id"]].copy()
        must_df["employee_id"] = must_df["employee_id"].astype(str).str.strip()
        must_df["slot_id"] = pd.to_numeric(must_df["slot_id"], errors="coerce")
        must_df = must_df.dropna(subset=["employee_id", "slot_id"])
        must_df["slot_id"] = must_df["slot_id"].astype(int)
        must_df = must_df.drop_duplicates(ignore_index=True)

        missing_pairs: list[tuple[str, int]] = []
        for employee_id, slot_id in must_df.itertuples(index=False):
            slot_idx = sid_of.get(slot_id)
            if slot_idx is None:
                # Il lock non cade nell'orizzonte attuale, viene ignorato.
                continue

            emp_idx = eid_of.get(employee_id)
            if emp_idx is None:
                missing_pairs.append((employee_id, slot_id))
                continue

            var = assign_vars.get((emp_idx, slot_idx))
            if var is None:
                missing_pairs.append((employee_id, slot_id))
                continue

            model.Add(var == 1)

        if missing_pairs:
            preview = ", ".join(f"({emp}, {slot})" for emp, slot in missing_pairs[:5])
            raise ValueError(
                "locks_must contiene coppie non ammesse tra i candidati: "
                f"{preview}"
            )

    if locks_forbid is not None and not locks_forbid.empty:
        forbid_df = locks_forbid.loc[:, ["employee_id", "slot_id"]].copy()
        forbid_df["employee_id"] = forbid_df["employee_id"].astype(str).str.strip()
        forbid_df["slot_id"] = pd.to_numeric(forbid_df["slot_id"], errors="coerce")
        forbid_df = forbid_df.dropna(subset=["employee_id", "slot_id"])
        forbid_df["slot_id"] = forbid_df["slot_id"].astype(int)
        forbid_df = forbid_df.drop_duplicates(ignore_index=True)

        for employee_id, slot_id in forbid_df.itertuples(index=False):
            slot_idx = sid_of.get(slot_id)
            if slot_idx is None:
                # Fuori orizzonte: il blocco non ha effetto nel modello corrente.
                continue

            emp_idx = eid_of.get(employee_id)
            if emp_idx is None:
                continue

            var = assign_vars.get((emp_idx, slot_idx))
            if var is not None:
                model.Add(var == 0)


def _add_hour_constraints(
    context: ModelContext,
    model: cp_model.CpModel,
    assign_vars: Dict[tuple[int, int], cp_model.IntVar],
    bundle: Mapping[str, object],
) -> dict[str, dict[tuple[int, str], cp_model.IntVar]]:
    empty_info = {
        "monthly_balance": {},
        "monthly_under": {},
        "monthly_over": {},
        "monthly_under_effective": {},
        "monthly_over_effective": {},
        "final_balance": {},
        "final_balance_abs": {},
    }

    if not assign_vars:
        return empty_info

    (
        week_by_date,
        month_by_date,
        day_week_map,
        day_month_map,
        horizon_start,
        horizon_end,
        horizon_week_ids,
    ) = _extract_calendar_maps(context, bundle)

    slot_duration_map = _build_slot_duration_minutes(context, bundle)
    if not slot_duration_map:
        return empty_info

    slot_date2: Mapping[int, int] = bundle.get("slot_date2", {})  # type: ignore[assignment]
    week_slots: dict[str, list[int]] = defaultdict(list)
    month_slots: dict[str, list[int]] = defaultdict(list)
    month_capacity: dict[str, int] = defaultdict(int)

    for slot_idx, day_idx in slot_date2.items():
        duration = int(slot_duration_map.get(slot_idx, 0))
        week_id = day_week_map.get(day_idx)
        if week_id is not None:
            week_slots[week_id].append(slot_idx)
        day_month = day_month_map.get(day_idx)
        if day_month is not None:
            month_slots[day_month].append(slot_idx)
            month_capacity[day_month] += max(duration, 0)

    employee_params = _extract_employee_hour_params(context, bundle)
    if not employee_params:
        return empty_info

    eid_of: Mapping[str, int] = bundle["eid_of"]
    count_leaves = _absences_count_as_worked(context.cfg)
    month_history_minutes = _build_month_history_minutes(bundle, eid_of, count_leaves)
    month_history_ranges = _build_month_history_ranges(bundle)
    week_history_minutes = _compute_history_minutes_by_week(
        context,
        week_by_date,
        horizon_start,
        horizon_week_ids,
        count_leaves,
        eid_of,
    )

    for _, month_id in month_history_minutes.keys():
        if month_id not in month_slots:
            month_slots[month_id] = []
            month_capacity.setdefault(month_id, 0)

    monthly_balance_vars: dict[tuple[int, str], cp_model.IntVar] = {}
    monthly_under_vars: dict[tuple[int, str], cp_model.IntVar] = {}
    monthly_over_vars: dict[tuple[int, str], cp_model.IntVar] = {}
    monthly_under_effective: dict[tuple[int, str], cp_model.IntVar] = {}
    monthly_over_effective: dict[tuple[int, str], cp_model.IntVar] = {}
    monthly_balance_bounds: dict[tuple[int, str], int] = {}

    deadband_minutes = _resolve_due_hours_deadband(context.cfg)

    for month_id, slots_in_month in month_slots.items():
        capacity = month_capacity.get(month_id, 0)
        for emp_idx, params in employee_params.items():
            due_minutes = params.get("due_minutes")
            month_limit = params.get("max_month_minutes")
            history_minutes = month_history_minutes.get((emp_idx, month_id), 0)

            if due_minutes is None and month_limit is None and history_minutes == 0 and not slots_in_month:
                continue

            terms = [
                assign_vars[(emp_idx, slot_idx)] * int(slot_duration_map.get(slot_idx, 0))
                for slot_idx in slots_in_month
                if (emp_idx, slot_idx) in assign_vars and slot_duration_map.get(slot_idx, 0) > 0
            ]
            expr = sum(terms) if terms else 0

            has_full_month_coverage = False
            if due_minutes is not None:
                has_full_month_coverage = _should_enforce_month_due_hours(
                    month_id,
                    horizon_start,
                    horizon_end,
                    month_history_ranges,
                )

            if has_full_month_coverage and due_minutes is not None:
                # We only build the due-hour slack (and the related balance
                # variable) when the planning horizon plus the historical
                # summary covers the entire calendar month.  Otherwise the
                # equality below would force the solver to compensate for time
                # that lies outside the available data window.  In those cases
                # the hard balance-delta limit is skipped as well because it
                # relies on the same balance variable.
                bound = max(capacity + abs(history_minutes) + abs(due_minutes), 1)
                balance = model.NewIntVar(
                    -bound,
                    bound,
                    f"month_balance_e{emp_idx}_m{month_id}",
                )
                under_var = model.NewIntVar(
                    0,
                    bound,
                    f"month_under_e{emp_idx}_m{month_id}",
                )
                over_var = model.NewIntVar(
                    0,
                    bound,
                    f"month_over_e{emp_idx}_m{month_id}",
                )
                model.Add(expr + history_minutes + under_var == due_minutes + over_var)
                model.Add(balance == expr + history_minutes - due_minutes)
                model.Add(balance == over_var - under_var)
                monthly_balance_vars[(emp_idx, month_id)] = balance
                monthly_balance_bounds[(emp_idx, month_id)] = bound
                monthly_under_vars[(emp_idx, month_id)] = under_var
                monthly_over_vars[(emp_idx, month_id)] = over_var

                balance_delta = params.get("max_balance_delta_minutes")
                if isinstance(balance_delta, int) and balance_delta > 0:
                    model.Add(balance <= balance_delta)
                    model.Add(balance >= -balance_delta)

                if deadband_minutes > 0:
                    under_eff = model.NewIntVar(
                        0,
                        bound,
                        f"month_under_eff_e{emp_idx}_m{month_id}",
                    )
                    over_eff = model.NewIntVar(
                        0,
                        bound,
                        f"month_over_eff_e{emp_idx}_m{month_id}",
                    )
                    under_shift = model.NewIntVar(
                        -bound,
                        bound,
                        f"month_under_shift_e{emp_idx}_m{month_id}",
                    )
                    over_shift = model.NewIntVar(
                        -bound,
                        bound,
                        f"month_over_shift_e{emp_idx}_m{month_id}",
                    )
                    model.Add(under_shift == under_var - deadband_minutes)
                    model.Add(over_shift == over_var - deadband_minutes)
                    model.AddMaxEquality(under_eff, [under_shift, 0])
                    model.AddMaxEquality(over_eff, [over_shift, 0])
                else:
                    under_eff = under_var
                    over_eff = over_var

                monthly_under_effective[(emp_idx, month_id)] = under_eff
                monthly_over_effective[(emp_idx, month_id)] = over_eff

            if month_limit is not None:
                model.Add(expr + history_minutes <= month_limit)

    final_balance_vars: dict[tuple[int, str], cp_model.IntVar] = {}
    final_balance_abs_vars: dict[tuple[int, str], cp_model.IntVar] = {}

    if monthly_balance_vars:
        emp_of: Mapping[int, str] = bundle.get("emp_of", {})  # type: ignore[assignment]
        start_balances = _extract_start_balance_series(context)
        balance_by_emp: dict[int, list[tuple[str, cp_model.IntVar]]] = defaultdict(list)
        for (emp_idx, month_id), balance in monthly_balance_vars.items():
            balance_by_emp.setdefault(emp_idx, []).append((month_id, balance))

        for emp_idx, entries in balance_by_emp.items():
            if not entries:
                continue
            emp_id = emp_of.get(emp_idx)
            if emp_id is None:
                continue
            start_hours = float(start_balances.get(emp_id, 0.0))
            start_minutes = int(round(start_hours * 60.0))

            bound_sum = abs(start_minutes)
            for month_id, _ in entries:
                bound_sum += int(monthly_balance_bounds.get((emp_idx, month_id), 0))

            if bound_sum <= 0:
                bound_sum = 1

            final_balance = model.NewIntVar(
                -bound_sum,
                bound_sum,
                f"final_balance_e{emp_idx}",
            )
            month_terms = [balance for _, balance in entries]
            model.Add(final_balance == start_minutes + sum(month_terms))

            abs_bound = bound_sum
            final_abs = model.NewIntVar(
                0,
                abs_bound,
                f"final_balance_abs_e{emp_idx}",
            )
            model.AddAbsEquality(final_abs, final_balance)

            final_balance_vars[(emp_idx, "final")] = final_balance
            final_balance_abs_vars[(emp_idx, "final")] = final_abs

    for week_id, slots_in_week in week_slots.items():
        for emp_idx, params in employee_params.items():
            limit = params.get("max_week_minutes")
            if limit is None:
                continue
            history_minutes = week_history_minutes.get((emp_idx, week_id), 0)
            terms = [
                assign_vars[(emp_idx, slot_idx)] * int(slot_duration_map.get(slot_idx, 0))
                for slot_idx in slots_in_week
                if (emp_idx, slot_idx) in assign_vars and slot_duration_map.get(slot_idx, 0) > 0
            ]
            expr = sum(terms) if terms else 0
            model.Add(expr + history_minutes <= limit)

    return {
        "monthly_balance": monthly_balance_vars,
        "monthly_under": monthly_under_vars,
        "monthly_over": monthly_over_vars,
        "monthly_under_effective": monthly_under_effective,
        "monthly_over_effective": monthly_over_effective,
        "final_balance": final_balance_vars,
        "final_balance_abs": final_balance_abs_vars,
    }


def _resolve_due_hours_penalty_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0
    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), Mapping) else None
    if defaults is None:
        return 0.0
    balance_cfg = defaults.get("balance") if isinstance(defaults.get("balance"), Mapping) else None
    if balance_cfg is None:
        return 0.0
    raw = balance_cfg.get("due_hours_penalty_weight")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


def _resolve_due_hour_penalty_weights(
    cfg: Mapping[str, Any] | None,
) -> tuple[float, float]:
    base = _resolve_due_hours_penalty_weight(cfg)
    under_weight = base
    over_weight = base

    if isinstance(cfg, Mapping):
        defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), Mapping) else None
        balance_cfg = (
            defaults.get("balance")
            if isinstance(defaults.get("balance"), Mapping)
            else None
        )
        if isinstance(balance_cfg, Mapping):
            under_raw = balance_cfg.get("due_hours_under_penalty_weight")
            over_raw = balance_cfg.get("due_hours_over_penalty_weight")
            if under_raw is not None:
                try:
                    under_weight = max(float(under_raw), 0.0)
                except (TypeError, ValueError):
                    under_weight = base
            if over_raw is not None:
                try:
                    over_weight = max(float(over_raw), 0.0)
                except (TypeError, ValueError):
                    over_weight = base

    return under_weight, over_weight


def _resolve_due_hours_deadband(cfg: Mapping[str, Any] | None) -> int:
    if not isinstance(cfg, Mapping):
        return 0
    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), Mapping) else None
    if defaults is None:
        return 0
    balance_cfg = defaults.get("balance") if isinstance(defaults.get("balance"), Mapping) else None
    if balance_cfg is None:
        return 0

    raw = None
    for key in ("due_hours_deadband_h", "due_hours_deadband_hours", "due_hours_deadband"):
        if key in balance_cfg:
            raw = balance_cfg.get(key)
            break
    if raw is None:
        return 0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0
    if value <= 0:
        return 0
    return int(round(value * 60.0))


def _resolve_final_balance_penalty_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0
    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), Mapping) else None
    if defaults is None:
        return 0.0
    balance_cfg = defaults.get("balance") if isinstance(defaults.get("balance"), Mapping) else None
    if balance_cfg is None:
        return 0.0

    raw = balance_cfg.get("final_balance_penalty_weight")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


def _extract_start_balance_series(context: ModelContext) -> pd.Series:
    employees = context.employees
    if employees is None or employees.empty:
        return pd.Series(dtype=float)
    if "employee_id" not in employees.columns:
        return pd.Series(dtype=float)

    emp_ids = employees["employee_id"].astype(str).str.strip()
    candidates = [
        "start_balance",
        "start_balance_h",
        "saldo_iniziale_h",
        "saldo_iniziale_ore",
        "saldo_iniziale",
    ]

    values = None
    for column in candidates:
        if column in employees.columns:
            values = pd.to_numeric(employees[column], errors="coerce")
            break

    if values is None:
        values = pd.Series(0.0, index=employees.index)

    series = pd.Series(values.fillna(0.0).to_numpy(), index=emp_ids)
    series.index.name = "employee_id"
    return series


def _resolve_cross_penalty_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0
    cross_cfg = cfg.get("cross") if isinstance(cfg.get("cross"), Mapping) else None
    if cross_cfg is None:
        return 0.0
    raw = cross_cfg.get("penalty_weight")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


def _resolve_role_daily_minutes(cfg: Mapping[str, Any] | None) -> tuple[dict[str, float], float]:
    role_minutes: dict[str, float] = {}
    fallback_minutes = 0.0

    if isinstance(cfg, Mapping):
        defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), Mapping) else None
        abs_cfg = defaults.get("absences") if isinstance(defaults.get("absences"), Mapping) else None

        if isinstance(abs_cfg, Mapping):
            role_cfg = abs_cfg.get("full_day_hours_by_role_h")
            if isinstance(role_cfg, Mapping):
                for role, value in role_cfg.items():
                    try:
                        hours = float(value)
                    except (TypeError, ValueError):
                        continue
                    if hours > 0:
                        role_minutes[str(role).strip().upper()] = hours * 60.0

            fallback_raw = abs_cfg.get("fallback_contract_daily_avg_h")
            if fallback_raw is not None:
                try:
                    fallback_minutes = float(fallback_raw) * 60.0
                except (TypeError, ValueError):
                    fallback_minutes = 0.0

    if fallback_minutes <= 0:
        fallback_minutes = 480.0

    return role_minutes, fallback_minutes


def _build_due_hour_objective_terms(
    context: ModelContext,
    bundle: Mapping[str, object],
    hour_info: Mapping[str, Mapping[tuple[int, str], cp_model.IntVar]],
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    under_vars: Mapping[tuple[int, str], cp_model.IntVar] = hour_info.get(  # type: ignore[assignment]
        "monthly_under_effective",
        {},
    )
    over_vars: Mapping[tuple[int, str], cp_model.IntVar] = hour_info.get(  # type: ignore[assignment]
        "monthly_over_effective",
        {},
    )
    if not under_vars and not over_vars:
        return [], []

    weight_under, weight_over = _resolve_due_hour_penalty_weights(context.cfg)
    if weight_under <= 0 and weight_over <= 0:
        return [], []

    role_minutes, fallback_minutes = _resolve_role_daily_minutes(context.cfg)
    try:
        role_map = _build_employee_role_map(context.employees, bundle)
    except ValueError:
        role_map = {}

    emp_of: Mapping[int, str] = bundle.get("emp_of", {})  # type: ignore[assignment]
    keys = set(under_vars.keys()) | set(over_vars.keys())
    employee_ids = [emp_of.get(emp_idx) for emp_idx, _ in keys]
    employee_ids = [emp_id for emp_id in employee_ids if emp_id is not None]
    employee_ids = list(dict.fromkeys(employee_ids))

    if not employee_ids:
        return [], []

    start_balances = _extract_start_balance_series(context)
    balance_series = pd.Series(
        [start_balances.get(emp_id, 0.0) for emp_id in employee_ids],
        index=pd.Index(employee_ids, name="employee_id"),
        dtype=float,
    )
    coeffs_df = compute_adaptive_coefficients(balance_series)
    coeffs_under = coeffs_df["c_under"].to_dict() if not coeffs_df.empty else {}
    coeffs_over = coeffs_df["c_over"].to_dict() if not coeffs_df.empty else {}

    terms: List[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []
    for emp_idx, month_id in keys:
        emp_id = emp_of.get(emp_idx)
        if emp_id is None:
            continue
        role_u = role_map.get(emp_idx, "")
        minutes = role_minutes.get(role_u, 0.0)
        if minutes <= 0:
            minutes = fallback_minutes
        if minutes <= 0:
            continue

        under_var = under_vars.get((emp_idx, month_id))
        if under_var is not None and weight_under > 0:
            coeff = float(coeffs_under.get(emp_id, 1.5))
            penalty = (weight_under * coeff) / minutes
            scaled = int(round(penalty * DUE_HOUR_OBJECTIVE_SCALE))
            if scaled <= 0:
                scaled = 1
            terms.append(under_var * scaled)
            contributions.append(
                ObjectiveTermContribution(
                    component="ore_mensili_sotto",
                    var=under_var,
                    coeff=scaled,
                    unit_scale=minutes,
                )
            )

        over_var = over_vars.get((emp_idx, month_id))
        if over_var is not None and weight_over > 0:
            coeff = float(coeffs_over.get(emp_id, 1.5))
            penalty = (weight_over * coeff) / minutes
            scaled = int(round(penalty * DUE_HOUR_OBJECTIVE_SCALE))
            if scaled <= 0:
                scaled = 1
            terms.append(over_var * scaled)
            contributions.append(
                ObjectiveTermContribution(
                    component="ore_mensili_sopra",
                    var=over_var,
                    coeff=scaled,
                    unit_scale=minutes,
                )
            )

    return terms, contributions


def _build_final_balance_objective_terms(
    context: ModelContext,
    bundle: Mapping[str, object],
    hour_info: Mapping[str, Mapping[tuple[int, str], cp_model.IntVar]],
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    abs_vars: Mapping[tuple[int, str], cp_model.IntVar] = hour_info.get(  # type: ignore[assignment]
        "final_balance_abs",
        {},
    )
    if not abs_vars:
        return [], []

    weight = _resolve_final_balance_penalty_weight(context.cfg)
    if weight <= 0:
        return [], []

    role_minutes, fallback_minutes = _resolve_role_daily_minutes(context.cfg)
    try:
        role_map = _build_employee_role_map(context.employees, bundle)
    except ValueError:
        role_map = {}

    emp_of: Mapping[int, str] = bundle.get("emp_of", {})  # type: ignore[assignment]
    employee_ids = [emp_of.get(emp_idx) for emp_idx, _ in abs_vars.keys()]
    employee_ids = [emp_id for emp_id in employee_ids if emp_id is not None]
    employee_ids = list(dict.fromkeys(employee_ids))

    if not employee_ids:
        return [], []

    start_balances = _extract_start_balance_series(context)
    balance_series = pd.Series(
        [start_balances.get(emp_id, 0.0) for emp_id in employee_ids],
        index=pd.Index(employee_ids, name="employee_id"),
        dtype=float,
    )
    coeffs_df = compute_adaptive_coefficients(balance_series)
    coeffs_under = coeffs_df["c_under"].to_dict() if not coeffs_df.empty else {}
    coeffs_over = coeffs_df["c_over"].to_dict() if not coeffs_df.empty else {}

    terms: List[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []
    for (emp_idx, _), abs_var in abs_vars.items():
        emp_id = emp_of.get(emp_idx)
        if emp_id is None:
            continue

        role_u = role_map.get(emp_idx, "")
        minutes = role_minutes.get(role_u, 0.0)
        if minutes <= 0:
            minutes = fallback_minutes
        if minutes <= 0:
            continue

        coeff_under = float(coeffs_under.get(emp_id, 1.5))
        coeff_over = float(coeffs_over.get(emp_id, 1.5))
        coeff = (coeff_under + coeff_over) / 2.0

        penalty = (weight * coeff) / minutes
        scaled = int(round(penalty * DUE_HOUR_OBJECTIVE_SCALE))
        if scaled <= 0:
            scaled = 1

        terms.append(abs_var * scaled)
        contributions.append(
            ObjectiveTermContribution(
                component="bilancio_finale",
                var=abs_var,
                coeff=scaled,
                unit_scale=minutes,
            )
        )

    return terms, contributions


def _build_consecutive_night_objective_terms(
    night_info: Mapping[str, Any]
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    totals: Mapping[int, cp_model.IntVar] = night_info.get("extra_totals", {})  # type: ignore[assignment]
    if not totals:
        return [], []

    weight = float(night_info.get("penalty_weight", 0.0) or 0.0)
    if weight <= 0:
        return [], []

    coeff = int(round(weight * CONSECUTIVE_NIGHT_OBJECTIVE_SCALE))
    if coeff <= 0:
        coeff = 1

    terms: List[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []
    for total in totals.values():
        terms.append(total * coeff)
        contributions.append(
            ObjectiveTermContribution(
                component="notti_consecutive",
                var=total,
                coeff=coeff,
            )
        )

    return terms, contributions


def _build_single_night_recovery_objective_terms(
    single_night_info: Mapping[str, Any]
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    violations: Mapping[tuple[int, int], cp_model.BoolVar] = single_night_info.get("violations", {})  # type: ignore[assignment]
    if not violations:
        return [], []

    weight = float(single_night_info.get("weight", 0.0) or 0.0)
    if weight <= 0:
        return [], []

    coeff = int(round(weight * SINGLE_NIGHT_RECOVERY_OBJECTIVE_SCALE))
    if coeff <= 0:
        coeff = 1

    terms: List[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []
    for var in violations.values():
        terms.append(var * coeff)
        contributions.append(
            ObjectiveTermContribution(
                component="recupero_singola_notte",
                var=var,
                coeff=coeff,
            )
        )

    return terms, contributions


def _build_night_fairness_objective_terms(
    model: cp_model.CpModel,
    context: ModelContext,
    bundle: Mapping[str, object],
    night_info: Mapping[str, Any],
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    night_by_day: Mapping[tuple[int, int], cp_model.BoolVar] = night_info.get("night_by_day", {})  # type: ignore[assignment]
    if not night_by_day:
        return [], []

    employees = context.employees
    if employees is None or employees.empty:
        return [], []

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    if not eid_of:
        return [], []

    cfg = context.cfg if isinstance(context.cfg, Mapping) else {}
    night_codes = _resolve_night_codes(cfg)
    if not night_codes:
        return [], []

    weight_scale = _resolve_night_fairness_weight(cfg)
    if weight_scale <= 0:
        return [], []

    history_counts = _count_history_night_totals(context.history, night_codes, eid_of)
    emp_to_dept = _build_employee_reparto_index(employees, bundle)
    if not emp_to_dept:
        return [], []

    night_eligible = _build_employee_night_eligibility_map(employees, eid_of)
    weight_float_map = _build_employee_night_weight_map(employees, eid_of)
    weight_int_map = _convert_weight_map_to_int(
        weight_float_map,
        scale=NIGHT_FAIRNESS_WEIGHT_SCALE,
        reduce_gcd=False,
    )

    dept_members: dict[str, list[int]] = {}
    for emp_idx, dept in emp_to_dept.items():
        if not dept:
            continue
        if not night_eligible.get(emp_idx, True):
            continue
        dept_members.setdefault(dept, []).append(emp_idx)

    if not dept_members:
        return [], []

    terms: list[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []

    night_vars_by_emp: dict[int, list[cp_model.BoolVar]] = defaultdict(list)
    night_days_by_dept: dict[str, set[int]] = defaultdict(set)
    for (emp_idx, day_idx), var in night_by_day.items():
        dept = emp_to_dept.get(emp_idx)
        if not dept:
            continue
        if not night_eligible.get(emp_idx, True):
            continue
        night_vars_by_emp[emp_idx].append(var)
        night_days_by_dept[dept].add(day_idx)

    for dept, members in dept_members.items():
        if len(members) < 2:
            continue

        positive_members = [emp for emp in members if weight_int_map.get(emp, 0) > 0]
        if len(positive_members) < 2:
            continue

        dept_night_days = night_days_by_dept.get(dept)
        if not dept_night_days:
            continue

        weights = {emp: weight_int_map.get(emp, 0) for emp in positive_members}
        total_weight = sum(weights.values())
        if total_weight <= 0:
            continue

        employee_totals: dict[int, cp_model.LinearExpr] = {}
        employee_upper: dict[int, int] = {}
        employee_history: dict[int, int] = {}

        for emp_idx in positive_members:
            history_value = history_counts.get(emp_idx, 0)
            employee_history[emp_idx] = history_value
            planned_vars = night_vars_by_emp.get(emp_idx, [])
            eligible_count = len(planned_vars)
            if planned_vars:
                planned_sum: cp_model.LinearExpr = sum(planned_vars)
            else:
                planned_sum = 0

            total_expr = planned_sum + history_value
            employee_totals[emp_idx] = total_expr
            employee_upper[emp_idx] = history_value + eligible_count

        if not employee_totals:
            continue

        dept_total_slots = len(dept_night_days)
        for emp_idx in positive_members:
            history_value = employee_history.get(emp_idx, 0)
            upper_bound = employee_upper.get(emp_idx, 0)
            if upper_bound <= history_value:
                upper_bound = history_value + dept_total_slots
            if upper_bound < 0:
                upper_bound = 0
            employee_upper[emp_idx] = upper_bound

        total_sum_expr = sum(employee_totals.values())
        total_sum_upper = sum(employee_upper.get(emp_idx, 0) for emp_idx in positive_members)
        dept_slug = _sanitize_identifier(dept)
        penalty_vars: list[cp_model.IntVar] = []

        for emp_idx in positive_members:
            total_expr = employee_totals[emp_idx]
            upper_bound = employee_upper.get(emp_idx, 0)
            deviation_upper = (
                total_weight * max(upper_bound, 0)
                + weights[emp_idx] * max(total_sum_upper, 0)
            )
            deviation_var = model.NewIntVar(
                0,
                max(deviation_upper, 0),
                f"night_fair_dev_e{emp_idx}_{dept_slug}",
            )

            scaled_total_expr = total_weight * total_expr
            target_expr = total_sum_expr * weights[emp_idx]
            model.Add(deviation_var >= scaled_total_expr - target_expr)
            model.Add(deviation_var >= target_expr - scaled_total_expr)

            scaled_upper = max(deviation_upper, 0) * NIGHT_FAIRNESS_WEIGHT_SCALE
            scaled_diff_var = model.NewIntVar(
                0,
                scaled_upper,
                f"night_fair_dev_scaled_e{emp_idx}_{dept_slug}",
            )
            model.Add(scaled_diff_var == deviation_var * NIGHT_FAIRNESS_WEIGHT_SCALE)

            rounded_numerator_upper = scaled_upper + total_weight // 2
            rounded_numerator = model.NewIntVar(
                0,
                rounded_numerator_upper,
                f"night_fair_dev_round_num_e{emp_idx}_{dept_slug}",
            )
            model.Add(rounded_numerator == scaled_diff_var + total_weight // 2)

            if total_weight > 0:
                units_upper = (rounded_numerator_upper + total_weight - 1) // total_weight
            else:
                units_upper = 0
            deviation_units = model.NewIntVar(
                0,
                units_upper,
                f"night_fair_dev_units_e{emp_idx}_{dept_slug}",
            )
            model.AddDivisionEquality(
                deviation_units,
                rounded_numerator,
                total_weight,
            )

            penalty_vars.append(deviation_units)

        if not penalty_vars:
            continue

        coeff = int(round(weight_scale * NIGHT_FAIRNESS_OBJECTIVE_SCALE))
        if coeff <= 0:
            continue

        for deviation_units in penalty_vars:
            terms.append(deviation_units * coeff)
            contributions.append(
                ObjectiveTermContribution(
                    component="fairness_notti",
                    var=deviation_units,
                    coeff=coeff,
                    unit_scale=NIGHT_FAIRNESS_WEIGHT_SCALE,
                )
            )

    return terms, contributions


def _build_weekend_fairness_objective_terms(
    model: cp_model.CpModel,
    context: ModelContext,
    bundle: Mapping[str, object],
    assign_vars: Mapping[tuple[int, int], cp_model.IntVar],
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    if not assign_vars:
        return [], []

    gamma = _resolve_weekend_fairness_weight(
        context.cfg if isinstance(context.cfg, Mapping) else None
    )
    if gamma <= 0:
        return [], []

    employees = context.employees
    if employees is None or employees.empty:
        return [], []

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    if not eid_of:
        return [], []

    slot_date2: Mapping[int, int] = bundle.get("slot_date2", {})  # type: ignore[assignment]
    if not slot_date2:
        return [], []

    weekend_day_flags = _build_weekend_or_holiday_day_flags(context, bundle)
    if not weekend_day_flags:
        return [], []

    weekend_slot_indices = sorted(
        {
            slot_idx
            for slot_idx, day_idx in slot_date2.items()
            if weekend_day_flags.get(day_idx, False)
        }
    )
    if not weekend_slot_indices:
        return [], []

    emp_to_dept = _build_employee_reparto_index(employees, bundle)
    if not emp_to_dept:
        return [], []

    slot_reparto_map = _build_slot_reparto_index(context, bundle)
    if not slot_reparto_map:
        return [], []

    weight_float_map = _build_employee_weekend_weight_map(employees, eid_of)
    weight_int_map = _convert_weight_map_to_int(
        weight_float_map,
        scale=WEEKEND_FAIRNESS_WEIGHT_SCALE,
        reduce_gcd=False,
    )

    dept_members: dict[str, list[int]] = defaultdict(list)
    for emp_idx, dept in emp_to_dept.items():
        if not dept:
            continue
        dept_members[dept].append(emp_idx)

    if not dept_members:
        return [], []

    weekend_slots_by_dept: dict[str, list[int]] = defaultdict(list)
    for slot_idx in weekend_slot_indices:
        dept = slot_reparto_map.get(slot_idx)
        if dept:
            weekend_slots_by_dept[dept].append(slot_idx)

    history_counts = _count_history_weekend_totals(context.history, eid_of)

    terms: list[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []

    for dept, members in dept_members.items():
        weekend_slots_for_dept = weekend_slots_by_dept.get(dept, [])
        if not weekend_slots_for_dept:
            continue

        positive_members = [emp for emp in members if weight_int_map.get(emp, 0) > 0]
        if len(positive_members) < 2:
            continue

        weights = {emp: weight_int_map.get(emp, 0) for emp in positive_members}
        total_weight = sum(weights.values())
        if total_weight <= 0:
            continue

        employee_totals: dict[int, cp_model.LinearExpr] = {}
        employee_upper: dict[int, int] = {}
        employee_history: dict[int, int] = {}

        for emp_idx in positive_members:
            history_value = history_counts.get(emp_idx, 0)
            employee_history[emp_idx] = history_value
            planned_vars = [
                assign_vars[(emp_idx, slot_idx)]
                for slot_idx in weekend_slots_for_dept
                if (emp_idx, slot_idx) in assign_vars
            ]
            eligible_count = len(planned_vars)
            planned_sum: cp_model.LinearExpr
            if planned_vars:
                planned_sum = sum(planned_vars)
            else:
                planned_sum = 0

            total_expr = planned_sum + history_value
            employee_totals[emp_idx] = total_expr
            employee_upper[emp_idx] = history_value + eligible_count

        if not employee_totals:
            continue

        for emp_idx in positive_members:
            history_value = employee_history.get(emp_idx, 0)
            upper_bound = employee_upper.get(emp_idx, 0)
            if upper_bound <= history_value:
                upper_bound = history_value + len(weekend_slots_for_dept)
            if upper_bound < 0:
                upper_bound = 0
            employee_upper[emp_idx] = upper_bound

        total_sum_expr = sum(employee_totals.values())
        total_sum_upper = sum(employee_upper.get(emp_idx, 0) for emp_idx in positive_members)
        dept_slug = _sanitize_identifier(dept)
        penalty_vars: list[cp_model.IntVar] = []

        for emp_idx in positive_members:
            total_expr = employee_totals[emp_idx]
            upper_bound = employee_upper.get(emp_idx, 0)
            deviation_upper = (
                total_weight * max(upper_bound, 0)
                + weights[emp_idx] * max(total_sum_upper, 0)
            )
            deviation_var = model.NewIntVar(
                0,
                max(deviation_upper, 0),
                f"weekend_fair_dev_e{emp_idx}_{dept_slug}",
            )

            scaled_total_expr = total_weight * total_expr
            target_expr = total_sum_expr * weights[emp_idx]
            model.Add(deviation_var >= scaled_total_expr - target_expr)
            model.Add(deviation_var >= target_expr - scaled_total_expr)

            scaled_upper = max(deviation_upper, 0) * WEEKEND_FAIRNESS_WEIGHT_SCALE
            scaled_diff_var = model.NewIntVar(
                0,
                scaled_upper,
                f"weekend_fair_dev_scaled_e{emp_idx}_{dept_slug}",
            )
            model.Add(scaled_diff_var == deviation_var * WEEKEND_FAIRNESS_WEIGHT_SCALE)

            rounded_numerator_upper = scaled_upper + total_weight // 2
            rounded_numerator = model.NewIntVar(
                0,
                rounded_numerator_upper,
                f"weekend_fair_dev_round_num_e{emp_idx}_{dept_slug}",
            )
            model.Add(rounded_numerator == scaled_diff_var + total_weight // 2)

            if total_weight > 0:
                units_upper = (rounded_numerator_upper + total_weight - 1) // total_weight
            else:
                units_upper = 0
            deviation_units = model.NewIntVar(
                0,
                units_upper,
                f"weekend_fair_dev_units_e{emp_idx}_{dept_slug}",
            )
            model.AddDivisionEquality(
                deviation_units,
                rounded_numerator,
                total_weight,
            )

            penalty_vars.append(deviation_units)

        if not penalty_vars:
            continue

        coeff = int(round(gamma * WEEKEND_FAIRNESS_OBJECTIVE_SCALE))
        if coeff <= 0:
            continue

        for deviation_var in penalty_vars:
            terms.append(deviation_var * coeff)
            contributions.append(
                ObjectiveTermContribution(
                    component="fairness_weekend",
                    var=deviation_var,
                    coeff=coeff,
                    unit_scale=WEEKEND_FAIRNESS_WEIGHT_SCALE,
                )
            )

    return terms, contributions


def _build_rest11_objective_terms(
    violations: Mapping[tuple[int, int, int], cp_model.BoolVar],
    weight: float,
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    if not violations or weight <= 0:
        return [], []

    coeff = int(round(weight * REST11_OBJECTIVE_SCALE))
    if coeff <= 0:
        coeff = 1

    terms: List[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []
    for var in violations.values():
        terms.append(var * coeff)
        contributions.append(
            ObjectiveTermContribution(
                component="riposo_11h",
                var=var,
                coeff=coeff,
            )
        )

    return terms, contributions


def _build_weekly_rest_objective_terms(
    violations: Mapping[tuple[int, int], cp_model.BoolVar],
    weight: float,
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    if not violations or weight <= 0:
        return [], []

    coeff = int(round(weight * WEEKLY_REST_OBJECTIVE_SCALE))
    if coeff <= 0:
        coeff = 1

    terms: List[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []
    for var in violations.values():
        terms.append(var * coeff)
        contributions.append(
            ObjectiveTermContribution(
                component="riposo_settimanale",
                var=var,
                coeff=coeff,
            )
        )

    return terms, contributions


def _add_cross_assignment_limits(
    context: ModelContext,
    model: cp_model.CpModel,
    assign_vars: Mapping[tuple[int, int], cp_model.IntVar],
    bundle: Mapping[str, object],
) -> None:
    """Impose hard limits on cross-department assignments per employee."""

    if not assign_vars:
        return

    cross_limits = _build_employee_cross_limit_map(context.employees, bundle)
    if not cross_limits:
        return

    slot_reparto_map = _build_slot_reparto_index(context, bundle)
    if not slot_reparto_map:
        return

    employee_reparto_map = _build_employee_reparto_index(context.employees, bundle)
    if not employee_reparto_map:
        return

    cross_assignments_by_emp: dict[int, list[cp_model.IntVar]] = {}

    for (emp_idx, slot_idx), var in assign_vars.items():
        limit = cross_limits.get(emp_idx)
        if limit is None:
            continue

        emp_reparto = employee_reparto_map.get(emp_idx)
        slot_reparto = slot_reparto_map.get(slot_idx)

        if not emp_reparto or not slot_reparto:
            continue

        if emp_reparto != slot_reparto:
            cross_assignments_by_emp.setdefault(emp_idx, []).append(var)

    for emp_idx, vars_list in cross_assignments_by_emp.items():
        if not vars_list:
            continue
        limit = cross_limits.get(emp_idx)
        if limit is None:
            continue
        model.Add(sum(vars_list) <= int(limit))


def _build_cross_assignment_objective_terms(
    context: ModelContext,
    bundle: Mapping[str, object],
    assign_vars: Mapping[tuple[int, int], cp_model.IntVar],
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    if not assign_vars:
        return [], []

    weight = _resolve_cross_penalty_weight(
        context.cfg if isinstance(context.cfg, Mapping) else None
    )
    if weight <= 0:
        return [], []

    slot_reparto_map = _build_slot_reparto_index(context, bundle)
    employee_reparto_map = _build_employee_reparto_index(context.employees, bundle)
    if not slot_reparto_map or not employee_reparto_map:
        return [], []

    cross_assignments: list[cp_model.IntVar] = []
    for (emp_idx, slot_idx), var in assign_vars.items():
        slot_rep = slot_reparto_map.get(slot_idx)
        emp_rep = employee_reparto_map.get(emp_idx)
        if not slot_rep or not emp_rep:
            continue
        if slot_rep != emp_rep:
            cross_assignments.append(var)

    if not cross_assignments:
        return [], []

    coeff = int(round(weight * CROSS_ASSIGNMENT_OBJECTIVE_SCALE))
    if coeff <= 0:
        coeff = 1

    terms: List[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []
    for var in cross_assignments:
        terms.append(var * coeff)
        contributions.append(
            ObjectiveTermContribution(
                component="assegnazioni_cross",
                var=var,
                coeff=coeff,
            )
        )

    return terms, contributions


def _build_preassignment_objective_terms(
    context: ModelContext,
    bundle: Mapping[str, object],
    state_vars: Mapping[tuple[int, int, str], cp_model.IntVar],
) -> tuple[List[cp_model.LinearExpr], list[ObjectiveTermContribution]]:
    pairs = bundle.get("preassignment_pairs", [])
    if not isinstance(pairs, list) or not pairs:
        return [], []

    weight = _resolve_preassignment_penalty_weight(
        context.cfg if isinstance(context.cfg, Mapping) else None
    )
    if weight <= 0:
        return [], []

    coeff = int(round(weight * PREASSIGNMENT_OBJECTIVE_SCALE))
    if coeff <= 0:
        coeff = 1

    terms: list[cp_model.LinearExpr] = []
    contributions: list[ObjectiveTermContribution] = []
    for entry in pairs:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            continue
        emp_idx, day_idx, state_code = entry
        try:
            emp_idx_int = int(emp_idx)
            day_idx_int = int(day_idx)
        except (TypeError, ValueError):
            continue
        key = (emp_idx_int, day_idx_int, str(state_code).strip().upper())
        var = state_vars.get(key)
        if var is None:
            continue
        terms.append((1 - var) * coeff)
        contributions.append(
            ObjectiveTermContribution(
                component="preassegnazioni",
                var=var,
                coeff=coeff,
                is_complement=True,
            )
        )

    return terms, contributions


def _build_slot_reparto_index(
    context: ModelContext, bundle: Mapping[str, object]
) -> dict[int, str]:
    sid_of: Mapping[object, int] = bundle.get("sid_of", {})  # type: ignore[assignment]
    result: dict[int, str] = {}

    raw_map = bundle.get("slot_reparto")
    if isinstance(raw_map, Mapping):
        for slot_id, reparto in raw_map.items():
            slot_idx = sid_of.get(slot_id)
            if slot_idx is None:
                continue
            if reparto is None or pd.isna(reparto):
                continue
            text = str(reparto).strip().upper()
            if text:
                result[slot_idx] = text

    if result:
        return result

    if context.slots.empty or "slot_id" not in context.slots.columns:
        return result
    if "reparto_id" not in context.slots.columns:
        return result

    for row in context.slots.loc[:, ["slot_id", "reparto_id"]].itertuples(index=False):
        slot_id = getattr(row, "slot_id")
        slot_idx = sid_of.get(slot_id)
        if slot_idx is None:
            continue
        reparto = getattr(row, "reparto_id")
        if reparto is None or pd.isna(reparto):
            continue
        text = str(reparto).strip().upper()
        if text and slot_idx not in result:
            result[slot_idx] = text

    return result


def _build_employee_reparto_index(
    employees: pd.DataFrame, bundle: Mapping[str, object]
) -> dict[int, str]:
    if employees.empty or "employee_id" not in employees.columns:
        return {}

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    if not eid_of:
        return {}

    reparto_cols = [
        col
        for col in ("employee_reparto_id", "reparto_id", "department_id")
        if col in employees.columns
    ]
    if not reparto_cols:
        return {}

    result: dict[int, str] = {}
    for row in employees.itertuples(index=False):
        emp_id = str(getattr(row, "employee_id", "")).strip()
        if not emp_id:
            continue
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue

        reparto_value = None
        for col in reparto_cols:
            value = getattr(row, col, None)
            if value is None or pd.isna(value):
                continue
            text = str(value).strip().upper()
            if text:
                reparto_value = text
                break

        if reparto_value:
            result[emp_idx] = reparto_value

    return result


def _build_employee_cross_limit_map(
    employees: pd.DataFrame, bundle: Mapping[str, object]
) -> dict[int, int]:
    if employees.empty or "employee_id" not in employees.columns:
        return {}

    column = "cross_max_shifts_month"
    if column not in employees.columns:
        return {}

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    if not eid_of:
        return {}

    result: dict[int, int] = {}

    for row in employees.itertuples(index=False):
        emp_id = str(getattr(row, "employee_id", "")).strip()
        if not emp_id:
            continue

        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue

        raw_value = getattr(row, column, None)
        if raw_value is None:
            continue

        if isinstance(raw_value, float) and math.isnan(raw_value):
            continue

        try:
            limit = int(float(raw_value))
        except (TypeError, ValueError):
            continue

        if limit < 0:
            continue

        result[emp_idx] = limit

    return result


def _count_history_night_totals(
    history: pd.DataFrame,
    night_codes: set[str],
    eid_of: Mapping[str, int],
) -> dict[int, int]:
    if history is None or history.empty or not night_codes:
        return {}

    id_col = None
    for candidate in ("employee_id", "dipendente_id", "resource_id", "matricola"):
        if candidate in history.columns:
            id_col = candidate
            break
    if id_col is None:
        return {}

    shift_col = None
    for candidate in ("turno", "shift_code", "shift", "state", "codice_turno"):
        if candidate in history.columns:
            shift_col = candidate
            break
    if shift_col is None:
        return {}

    df = history.loc[:, [id_col, shift_col]].copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df[shift_col] = df[shift_col].astype(str).str.strip().str.upper()
    df = df[df[shift_col].isin({code.upper() for code in night_codes})]
    if df.empty:
        return {}

    counts = df.groupby(id_col)[shift_col].count()
    result: dict[int, int] = {}
    for emp_id, count in counts.items():
        emp_idx = eid_of.get(str(emp_id).strip())
        if emp_idx is not None:
            result[emp_idx] = int(count)
    return result


def _build_employee_night_eligibility_map(
    employees: pd.DataFrame, eid_of: Mapping[str, int]
) -> dict[int, bool]:
    if employees.empty or "employee_id" not in employees.columns:
        return {}

    id_series = employees["employee_id"].astype(str).str.strip()
    eligibility_col = None
    for candidate in (
        "can_work_night",
        "night_eligible",
        "abilitato_notte",
        "puo_notte",
    ):
        if candidate in employees.columns:
            eligibility_col = candidate
            break

    result: dict[int, bool] = {}
    if eligibility_col is None:
        for emp_id in id_series:
            emp_idx = eid_of.get(emp_id)
            if emp_idx is not None:
                result[emp_idx] = True
        return result

    values = employees[eligibility_col]
    for emp_id, raw_value in zip(id_series, values, strict=False):
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue
        result[emp_idx] = _parse_bool(raw_value, default=False)
    return result


def _build_employee_weight_map(
    employees: pd.DataFrame,
    eid_of: Mapping[str, int],
    candidate_cols: Iterable[str],
    *,
    default_value: float = 1.0,
) -> dict[int, float]:
    if employees.empty or "employee_id" not in employees.columns:
        return {}

    weight_series = None
    for col in candidate_cols:
        if col in employees.columns:
            numeric = pd.to_numeric(employees[col], errors="coerce")
            if numeric.notna().any():
                weight_series = numeric.fillna(0.0)
                break

    if weight_series is None:
        weight_series = pd.Series(default_value, index=employees.index)

    result: dict[int, float] = {}
    id_series = employees["employee_id"].astype(str).str.strip()
    for emp_id, value in zip(id_series, weight_series, strict=False):
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue
        weight_value = float(value)
        if not math.isfinite(weight_value):
            weight_value = 0.0
        result[emp_idx] = max(weight_value, 0.0)
    return result


def _build_employee_night_weight_map(
    employees: pd.DataFrame, eid_of: Mapping[str, int]
) -> dict[int, float]:
    candidate_cols = [
        "night_weight",
        "night_quota",
        "night_fte",
        "night_share",
        "night_fraction",
        "night_capacity",
        "night_ratio",
        "fte",
        "FTE",
    ]

    return _build_employee_weight_map(employees, eid_of, candidate_cols)


def _build_employee_weekend_weight_map(
    employees: pd.DataFrame, eid_of: Mapping[str, int]
) -> dict[int, float]:
    candidate_cols = [
        "weekend_weight",
        "weekend_quota",
        "weekend_fairness_weight",
        "weekend_fte",
        "fairness_weight",
        "fte",
        "FTE",
    ]

    return _build_employee_weight_map(employees, eid_of, candidate_cols)


def _convert_weight_map_to_int(
    weight_map: Mapping[int, float], *, scale: int = NIGHT_FAIRNESS_WEIGHT_SCALE, reduce_gcd: bool = True
) -> dict[int, int]:
    result: dict[int, int] = {}
    positives: list[int] = []
    for emp_idx, value in weight_map.items():
        if value <= 0:
            result[emp_idx] = 0
            continue
        scaled = int(round(value * scale))
        if scaled <= 0:
            scaled = 1
        result[emp_idx] = scaled
        positives.append(scaled)

    if reduce_gcd and positives:
        gcd_value = positives[0]
        for item in positives[1:]:
            gcd_value = math.gcd(gcd_value, item)
        if gcd_value > 1:
            for emp_idx, value in list(result.items()):
                if value > 0:
                    result[emp_idx] = value // gcd_value

    return result


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "t", "yes", "y", "si", "s", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _build_weekend_or_holiday_day_flags(
    context: ModelContext, bundle: Mapping[str, object]
) -> dict[int, bool]:
    calendars = context.calendars
    if calendars is None or calendars.empty or "data" not in calendars.columns:
        return {}

    did_of: Mapping[object, int] = bundle.get("did_of", {})  # type: ignore[assignment]
    if not did_of:
        return {}

    work = calendars.copy()
    data_dt = pd.to_datetime(work["data"], errors="coerce")
    work = work.assign(_data_dt=data_dt).dropna(subset=["_data_dt"])
    work["_data_dt"] = work["_data_dt"].dt.tz_localize(None)
    work["_data_dt"] = work["_data_dt"].dt.normalize()
    work["_date"] = work["_data_dt"].dt.date
    work = work.dropna(subset=["_date"]).drop_duplicates(subset=["_date"], keep="last")

    if work.empty:
        return {}

    if "is_weekend_or_holiday" in work.columns:
        raw_flags = work["is_weekend_or_holiday"].apply(lambda v: _parse_bool(v, False))
    else:
        if "is_weekend" in work.columns:
            weekend_series = work["is_weekend"].apply(lambda v: _parse_bool(v, False))
        else:
            weekend_series = work["_data_dt"].dt.weekday >= 5

        if "is_weekday_holiday" in work.columns:
            holiday_series = work["is_weekday_holiday"].apply(lambda v: _parse_bool(v, False))
        elif "holiday_desc" in work.columns:
            holiday_series = work["holiday_desc"].astype(str).str.strip().ne("")
        else:
            holiday_series = pd.Series(False, index=work.index)

        raw_flags = weekend_series.astype(bool) | holiday_series.astype(bool)

    result: dict[int, bool] = {}
    for day, flag in zip(work["_date"], raw_flags, strict=False):
        day_idx = did_of.get(day)
        if day_idx is not None:
            result[day_idx] = bool(flag)

    return result


def _count_history_weekend_totals(
    history: pd.DataFrame, eid_of: Mapping[str, int]
) -> dict[int, int]:
    if history is None or history.empty:
        return {}

    id_col = None
    for candidate in ("employee_id", "dipendente_id", "resource_id", "matricola"):
        if candidate in history.columns:
            id_col = candidate
            break
    if id_col is None:
        return {}

    work = history.copy()
    work[id_col] = work[id_col].astype(str).str.strip()

    if "is_weekend_or_holiday" in work.columns:
        flags = work["is_weekend_or_holiday"].apply(lambda v: _parse_bool(v, False))
    else:
        if "is_weekend" in work.columns:
            weekend_series = work["is_weekend"].apply(lambda v: _parse_bool(v, False))
        else:
            date_col = next((c for c in ("data", "date") if c in work.columns), None)
            if date_col is not None:
                parsed_dates = pd.to_datetime(work[date_col], errors="coerce")
                weekend_series = parsed_dates.dt.dayofweek.isin([5, 6])
            else:
                weekend_series = pd.Series(False, index=work.index)

        if "is_weekday_holiday" in work.columns:
            holiday_series = work["is_weekday_holiday"].apply(lambda v: _parse_bool(v, False))
        elif "holiday_desc" in work.columns:
            holiday_series = work["holiday_desc"].astype(str).str.strip().ne("")
        else:
            holiday_series = pd.Series(False, index=work.index)

        flags = weekend_series.fillna(False).astype(bool) | holiday_series.astype(bool)

    if not flags.any():
        return {}

    filtered = work.loc[flags.astype(bool)]
    counts = filtered.groupby(id_col).size()

    result: dict[int, int] = {}
    for emp_id, count in counts.items():
        emp_idx = eid_of.get(str(emp_id).strip())
        if emp_idx is not None:
            result[emp_idx] = int(count)

    return result


def _sanitize_identifier(value: str) -> str:
    if not value:
        return "dept"
    cleaned = ''.join(ch if ch.isalnum() else '_' for ch in str(value))
    cleaned = cleaned.strip('_')
    return cleaned or "dept"


def _build_slot_night_flags(
    context: ModelContext,
    bundle: Mapping[str, object],
    night_codes: set[str],
) -> dict[int, bool]:
    sid_of: Mapping[object, int] = bundle.get("sid_of", {})  # type: ignore[assignment]
    flags: dict[int, bool] = {}

    if not context.slots.empty and "slot_id" in context.slots.columns:
        if "is_night" in context.slots.columns:
            for row in context.slots.loc[:, ["slot_id", "is_night"]].itertuples(index=False):
                slot_idx = sid_of.get(getattr(row, "slot_id"))
                if slot_idx is None:
                    continue
                flags[slot_idx] = bool(getattr(row, "is_night"))

        if not flags and "shift_code" in context.slots.columns:
            shift_series = context.slots["shift_code"].astype(str).str.strip().str.upper()
            for slot_id, code in zip(context.slots["slot_id"], shift_series, strict=False):
                slot_idx = sid_of.get(slot_id)
                if slot_idx is not None:
                    flags[slot_idx] = code in night_codes

    return flags


def _resolve_consecutive_night_penalty_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0

    def _extract(mapping: Mapping[str, Any] | None) -> float | None:
        if not isinstance(mapping, Mapping):
            return None
        for key in (
            "extra_consecutive_penalty_weight",
            "penalty_extra_consecutive_night",
            "penalty_extra_night",
            "penalty_consecutive_night",
            "penalty_after_first_night",
        ):
            raw = mapping.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
        return None

    direct = _extract(cfg.get("night"))
    if direct is not None:
        return direct

    defaults = cfg.get("defaults")
    if isinstance(defaults, Mapping):
        fallback = _extract(defaults.get("night"))
        if fallback is not None:
            return fallback

    return 0.0


def _resolve_single_night_recovery_penalty_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0

    def _extract(mapping: Mapping[str, Any] | None) -> float | None:
        if not isinstance(mapping, Mapping):
            return None
        for key in (
            "single_night_recovery_penalty_weight",
            "penalty_single_night_recovery",
            "penalty_single_night_rest",
            "penalty_single_night_sequence",
        ):
            raw = mapping.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
        return None

    direct = _extract(cfg.get("night"))
    if direct is not None:
        return direct

    defaults = cfg.get("defaults")
    if isinstance(defaults, Mapping):
        fallback = _extract(defaults.get("night"))
        if fallback is not None:
            return fallback

    return 0.0


def _resolve_rest11_penalty_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0

    def _extract(mapping: Mapping[str, Any] | None) -> float | None:
        if not isinstance(mapping, Mapping):
            return None
        for key in (
            "rest11_penalty_weight",
            "penalty_rest11",
            "penalty_rest11h",
            "penalty_rest_gap",
            "penalty_weight",
        ):
            raw = mapping.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
        return None

    direct = _extract(cfg.get("rest_rules"))
    if direct is not None:
        return direct

    defaults = cfg.get("defaults")
    if isinstance(defaults, Mapping):
        fallback = _extract(defaults.get("rest11h"))
        if fallback is not None:
            return fallback

    return 0.0


def _resolve_weekly_rest_penalty_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0

    def _extract(mapping: Mapping[str, Any] | None) -> float | None:
        if not isinstance(mapping, Mapping):
            return None
        for key in (
            "weekly_rest_penalty_weight",
            "penalty_weekly_rest",
            "penalty_rest_week",
            "penalty_rest_day_weekly",
            "penalty_weight",
        ):
            raw = mapping.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
        return None

    direct = _extract(cfg.get("rest_rules"))
    if direct is not None:
        return direct

    defaults = cfg.get("defaults")
    fallback = None
    if isinstance(defaults, Mapping):
        fallback = _extract(defaults)
        if fallback is not None:
            return fallback
        rest_days = defaults.get("rest_days")
        fallback = _extract(rest_days)
        if fallback is not None:
            return fallback

    return 0.0


def _resolve_preassignment_penalty_weight(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0

    def _extract(mapping: Mapping[str, Any] | None) -> float | None:
        if not isinstance(mapping, Mapping):
            return None
        for key in ("change_penalty_weight", "penalty_weight", "weight"):
            raw = mapping.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
        return None

    direct = _extract(cfg.get("preassignments"))
    if direct is not None:
        return direct

    fallback = cfg.get("preassignments_penalty_weight")
    if fallback is not None:
        try:
            value = float(fallback)
        except (TypeError, ValueError):
            return 0.0
        return max(value, 0.0)

    return 0.0


def _add_night_pattern_constraints(
    context: ModelContext,
    model: cp_model.CpModel,
    state_vars: Dict[tuple[int, int, str], cp_model.IntVar],
    state_codes: Iterable[str],
    bundle: Mapping[str, object],
    prev_day_index: Mapping[int, int | None],
    next_day_index: Mapping[int, int | None],
    history_prev_night: set[tuple[int, int]],
    absence_state: str | None,
    forced_absences: set[tuple[int, int]],
) -> Mapping[str, Any]:
    single_recovery_weight = _resolve_single_night_recovery_penalty_weight(
        context.cfg if isinstance(context.cfg, Mapping) else None
    )
    single_recovery: dict[str, Any] = {
        "violations": {},
        "weight": single_recovery_weight,
    }
    result: dict[str, Any] = {"single_night_recovery": single_recovery}

    state_code_set = {str(code).strip().upper() for code in state_codes}

    night_codes = _resolve_night_codes(
        context.cfg if isinstance(context.cfg, Mapping) else {}
    )
    night_states = tuple(code for code in night_codes if code in state_code_set)
    if not night_states:
        return result

    sn_code = "SN" if "SN" in state_code_set else None
    rest_code = "R" if "R" in state_code_set else None
    if sn_code is None or rest_code is None:
        return result

    p_code = "P" if "P" in state_code_set else None
    absence_code = None
    if absence_state is not None:
        normalized = str(absence_state).strip().upper()
        if normalized in state_code_set:
            absence_code = normalized
    forced_absence_pairs = forced_absences if forced_absences else set()

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    num_employees = int(bundle.get("num_employees", len(eid_of)))
    date_of: Mapping[int, object] = bundle.get("date_of", {})  # type: ignore[assignment]
    num_days = int(bundle.get("num_days", len(date_of)))
    if num_employees <= 0 or num_days <= 0:
        return result

    day_dates: list[date | None] = []
    for day_idx in range(num_days):
        day_dates.append(_parse_date_value(date_of.get(day_idx)))
    first_day = next((day for day in day_dates if day is not None), None)

    initial_history = _compute_initial_night_streaks(
        context.history,
        eid_of,
        night_codes,
        first_day,
    )

    indicator_cache: dict[tuple[int, int, tuple[str, ...]], cp_model.IntVar] = {}

    single_recovery_flags: dict[tuple[int, int], cp_model.BoolVar] = {}

    for emp_idx in range(num_employees):
        for day_idx in range(num_days):
            night_var = _ensure_state_indicator(
                model,
                state_vars,
                emp_idx,
                day_idx,
                night_states,
                indicator_cache,
                "is_night",
            )
            if night_var is None:
                continue

            next_idx = next_day_index.get(day_idx)
            if next_idx is None:
                continue

            next_night_var = _ensure_state_indicator(
                model,
                state_vars,
                emp_idx,
                next_idx,
                night_states,
                indicator_cache,
                "is_night",
            )
            if next_night_var is None:
                continue

            sn_var = state_vars.get((emp_idx, next_idx, sn_code))
            if sn_var is None:
                continue

            next2_idx = next_day_index.get(next_idx)
            rest_var = (
                state_vars.get((emp_idx, next2_idx, rest_code))
                if next2_idx is not None
                else None
            )
            absence_rest_var = (
                state_vars.get((emp_idx, next2_idx, absence_code))
                if absence_code is not None and next2_idx is not None
                else None
            )
            rest_day_forced_absence = (
                absence_rest_var is not None
                and (emp_idx, next2_idx) in forced_absence_pairs
            )

            prev_idx = prev_day_index.get(day_idx)
            prev_night_var = None
            if prev_idx is not None:
                prev_night_var = _ensure_state_indicator(
                    model,
                    state_vars,
                    emp_idx,
                    prev_idx,
                    night_states,
                    indicator_cache,
                    "is_night",
                )
                if prev_night_var is None:
                    continue

                trigger = model.NewBoolVar(f"after_long_nights_e{emp_idx}_d{day_idx}")
                model.AddBoolAnd(
                    [prev_night_var, night_var, next_night_var.Not()]
                ).OnlyEnforceIf(trigger)
                model.AddBoolOr(
                    [trigger, prev_night_var.Not(), night_var.Not(), next_night_var]
                )
            elif (emp_idx, day_idx) in history_prev_night:
                trigger = model.NewBoolVar(f"after_history_nights_e{emp_idx}_d{day_idx}")
                model.AddBoolAnd([night_var, next_night_var.Not()]).OnlyEnforceIf(trigger)
                model.AddBoolOr([trigger, night_var.Not(), next_night_var])
            else:
                continue

            model.Add(sn_var == 1).OnlyEnforceIf(trigger)
            if rest_day_forced_absence and absence_rest_var is not None:
                model.Add(absence_rest_var == 1).OnlyEnforceIf(trigger)
            elif rest_var is not None and not rest_day_forced_absence:
                model.Add(rest_var == 1).OnlyEnforceIf(trigger)

            if (
                rest_var is not None
                and not rest_day_forced_absence
                and next2_idx is not None
                and (prev_idx is None or prev_night_var is not None)
                and (prev_idx is not None or (emp_idx, day_idx) not in history_prev_night)
            ):
                violation_var = model.NewBoolVar(
                    f"single_night_recovery_violation_e{emp_idx}_d{day_idx}"
                )
                single_recovery_flags[(emp_idx, day_idx)] = violation_var
                model.AddImplication(violation_var, night_var)
                model.AddImplication(violation_var, sn_var)
                model.AddImplication(violation_var, next_night_var.Not())
                model.AddImplication(violation_var, rest_var.Not())
                if prev_idx is not None and prev_night_var is not None:
                    model.AddImplication(violation_var, prev_night_var.Not())
                    model.AddBoolOr(
                        [
                            violation_var,
                            night_var.Not(),
                            sn_var.Not(),
                            next_night_var,
                            rest_var,
                            prev_night_var,
                        ]
                    )
                else:
                    model.AddBoolOr(
                        [
                            violation_var,
                            night_var.Not(),
                            sn_var.Not(),
                            next_night_var,
                            rest_var,
                        ]
                    )

        initial_streak = int(initial_history.get(emp_idx, 0))
        if initial_streak >= 2 and num_days > 0:
            first_idx = 0
            first_night = _ensure_state_indicator(
                model,
                state_vars,
                emp_idx,
                first_idx,
                night_states,
                indicator_cache,
                "is_night",
            )
            sn_first = state_vars.get((emp_idx, first_idx, sn_code))
            next_idx = next_day_index.get(first_idx)
            rest_first = (
                state_vars.get((emp_idx, next_idx, rest_code))
                if next_idx is not None
                else None
            )
            absence_first = (
                state_vars.get((emp_idx, next_idx, absence_code))
                if absence_code is not None and next_idx is not None
                else None
            )
            first_forced_absence = (
                absence_first is not None
                and next_idx is not None
                and (emp_idx, next_idx) in forced_absence_pairs
            )
            if first_night is not None and sn_first is not None:
                model.Add(sn_first == 1).OnlyEnforceIf(first_night.Not())
            if first_night is not None and next_idx is not None:
                if first_forced_absence and absence_first is not None:
                    model.Add(absence_first == 1).OnlyEnforceIf(first_night.Not())
                elif rest_first is not None and not first_forced_absence:
                    model.Add(rest_first == 1).OnlyEnforceIf(first_night.Not())

    single_recovery["violations"] = single_recovery_flags
    return result

def _add_night_constraints(
    context: ModelContext,
    model: cp_model.CpModel,
    assign_vars: Dict[tuple[int, int], cp_model.IntVar],
    bundle: Mapping[str, object],
) -> dict[str, Any]:
    global_penalty_weight = _resolve_consecutive_night_penalty_weight(
        context.cfg if isinstance(context.cfg, Mapping) else None
    )

    result: dict[str, Any] = {
        "night_by_day": {},
        "extra_penalties": {},
        "extra_totals": {},
        "penalty_weight": global_penalty_weight,
    }

    if not assign_vars:
        return result

    night_codes = _resolve_night_codes(context.cfg if isinstance(context.cfg, Mapping) else {})
    if not night_codes:
        return result

    (
        week_by_date,
        month_by_date,
        day_week_map,
        day_month_map,
        horizon_start,
        _horizon_end,
        horizon_week_ids,
    ) = _extract_calendar_maps(context, bundle)

    if not week_by_date and not month_by_date:
        return result

    slot_date2: Mapping[int, int] = bundle.get("slot_date2", {})  # type: ignore[assignment]
    if not slot_date2:
        return result

    slot_night_flags = _build_slot_night_flags(context, bundle, night_codes)
    if not slot_night_flags:
        return result

    week_slots: dict[str, list[int]] = defaultdict(list)
    month_slots: dict[str, list[int]] = defaultdict(list)
    day_slots: dict[int, list[int]] = defaultdict(list)

    for slot_idx, day_idx in slot_date2.items():
        if not slot_night_flags.get(slot_idx, False):
            continue
        week_id = day_week_map.get(day_idx)
        if week_id is not None:
            week_slots[week_id].append(slot_idx)
        month_id = day_month_map.get(day_idx)
        if month_id is not None:
            month_slots[month_id].append(slot_idx)
        day_slots[day_idx].append(slot_idx)

    employee_limits = _extract_employee_night_params(context, bundle)
    if not employee_limits:
        return result

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    month_history = _build_month_history_nights(bundle, eid_of)
    week_history = _compute_history_nights_by_week(
        context,
        week_by_date,
        horizon_start,
        horizon_week_ids,
        night_codes,
        eid_of,
    )

    for _, month_id in month_history.keys():
        month_slots.setdefault(month_id, [])
    for _, week_id in week_history.keys():
        week_slots.setdefault(week_id, [])

    for month_id, slots_in_month in month_slots.items():
        for emp_idx, limits in employee_limits.items():
            limit = limits.get("max_month_nights")
            if limit is None:
                continue
            history_count = month_history.get((emp_idx, month_id), 0)
            terms = [
                assign_vars[(emp_idx, slot_idx)]
                for slot_idx in slots_in_month
                if (emp_idx, slot_idx) in assign_vars
            ]
            if terms or history_count:
                expr = sum(terms) if terms else 0
                model.Add(expr + history_count <= limit)

    for week_id, slots_in_week in week_slots.items():
        for emp_idx, limits in employee_limits.items():
            limit = limits.get("max_week_nights")
            if limit is None:
                continue
            history_count = week_history.get((emp_idx, week_id), 0)
            terms = [
                assign_vars[(emp_idx, slot_idx)]
                for slot_idx in slots_in_week
                if (emp_idx, slot_idx) in assign_vars
            ]
            if terms or history_count:
                expr = sum(terms) if terms else 0
                model.Add(expr + history_count <= limit)

    prev_day_index = _build_previous_day_index(bundle)
    date_of: Mapping[int, object] = bundle.get("date_of", {})  # type: ignore[assignment]
    num_days = int(bundle.get("num_days", len(date_of)))
    num_employees = int(bundle.get("num_employees", len(eid_of)))

    if num_days <= 0 or num_employees <= 0:
        return result

    day_dates: list[date | None] = []
    for day_idx in range(num_days):
        day_dates.append(_parse_date_value(date_of.get(day_idx)))

    first_day = next((day for day in day_dates if day is not None), None)
    initial_history = _compute_initial_night_streaks(
        context.history,
        eid_of,
        night_codes,
        first_day,
    )
    history_dates_all = _history_night_dates_by_employee(context.history, night_codes, eid_of)
    history_before_day: dict[tuple[int, int], int] = {}
    history_night_flags: dict[tuple[int, int], bool] = {}
    for emp_idx, dates in history_dates_all.items():
        streak = 0
        for day_idx in range(num_days):
            history_before_day[(emp_idx, day_idx)] = streak
            day_date = day_dates[day_idx]
            if day_date is not None and day_date in dates:
                history_night_flags[(emp_idx, day_idx)] = True
                streak += 1
            else:
                history_night_flags[(emp_idx, day_idx)] = False
                streak = 0

    night_by_day: dict[tuple[int, int], cp_model.BoolVar] = {}
    streak_vars: dict[tuple[int, int], cp_model.IntVar] = {}
    extra_penalties: dict[tuple[int, int], cp_model.IntVar] = {}
    penalty_totals: dict[int, cp_model.IntVar] = {}

    for emp_idx in range(num_employees):
        limits = employee_limits.get(emp_idx, {})
        consecutive_limit = limits.get("max_consecutive_nights")
        history_streak = int(initial_history.get(emp_idx, 0))

        has_restriction = (
            (consecutive_limit is not None and consecutive_limit >= 0)
            or global_penalty_weight > 0
            or history_streak > 0
        )

        if not has_restriction:
            continue

        streak_upper = consecutive_limit if consecutive_limit is not None else num_days
        if streak_upper is None or streak_upper < 0:
            streak_upper = num_days
        streak_upper = max(int(streak_upper), 0)
        extra_upper = max(streak_upper - 1, 0)

        extras_for_emp: list[cp_model.IntVar] = []

        for day_idx in range(num_days):
            night_var = model.NewBoolVar(f"is_night_e{emp_idx}_d{day_idx}")
            night_by_day[(emp_idx, day_idx)] = night_var

            slots_in_day = day_slots.get(day_idx, [])
            assign_list = [
                assign_vars[(emp_idx, slot_idx)]
                for slot_idx in slots_in_day
                if (emp_idx, slot_idx) in assign_vars
            ]
            if assign_list:
                model.Add(sum(assign_list) >= night_var)
                for var in assign_list:
                    model.Add(var <= night_var)
            else:
                if history_night_flags.get((emp_idx, day_idx), False):
                    model.Add(night_var == 1)
                else:
                    model.Add(night_var == 0)

            streak_var = model.NewIntVar(
                0,
                streak_upper,
                f"night_streak_e{emp_idx}_d{day_idx}",
            )
            streak_vars[(emp_idx, day_idx)] = streak_var

            extra_var = model.NewIntVar(
                0,
                extra_upper,
                f"night_extra_e{emp_idx}_d{day_idx}",
            )
            extra_penalties[(emp_idx, day_idx)] = extra_var
            extras_for_emp.append(extra_var)

            not_night = night_var.Not()
            model.Add(streak_var == 0).OnlyEnforceIf(not_night)
            model.Add(extra_var == 0).OnlyEnforceIf(not_night)

            history_before = history_before_day.get((emp_idx, day_idx), 0)
            history_is_night = history_night_flags.get((emp_idx, day_idx), False)

            if history_is_night and not assign_list:
                history_streak = history_before + 1
                model.Add(streak_var == history_streak)
                model.Add(extra_var == 0)
                continue

            prev_idx = prev_day_index.get(day_idx)
            if prev_idx is None:
                base_streak = max(history_before, 0)
                model.Add(streak_var == base_streak + 1).OnlyEnforceIf(night_var)
            else:
                prev_streak = streak_vars.get((emp_idx, prev_idx))
                if prev_streak is None:
                    base_streak = max(history_before, 0)
                    model.Add(streak_var == base_streak + 1).OnlyEnforceIf(night_var)
                else:
                    model.Add(streak_var == prev_streak + 1).OnlyEnforceIf(night_var)

            model.Add(streak_var - extra_var == 1).OnlyEnforceIf(night_var)

        if extras_for_emp:
            bound = len(extras_for_emp) * max(extra_upper, 0)
            total = model.NewIntVar(0, bound, f"night_extra_total_e{emp_idx}")
            model.Add(total == sum(extras_for_emp))
            penalty_totals[emp_idx] = total

    result["night_by_day"] = night_by_day
    result["extra_penalties"] = extra_penalties
    result["extra_totals"] = penalty_totals
    return result


def _resolve_rest_threshold(cfg: Mapping[str, Any] | None) -> float:
    if not isinstance(cfg, Mapping):
        return 0.0
    rest_cfg = cfg.get("rest_rules") if isinstance(cfg.get("rest_rules"), Mapping) else None
    if rest_cfg is None:
        rest_cfg = {}
    raw = rest_cfg.get("min_between_shifts_h", 0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


def _extract_rest11_limits(
    context: ModelContext, bundle: Mapping[str, object]
) -> dict[int, dict[str, int | None]]:
    employees = context.employees
    if employees is None or employees.empty or "employee_id" not in employees.columns:
        return {}

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    if not eid_of:
        return {}

    defaults = context.cfg.get("defaults") if isinstance(context.cfg, Mapping) else None
    rest_defaults = defaults.get("rest11h") if isinstance(defaults, Mapping) else None

    def _coerce_limit(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        try:
            num = int(round(float(value)))
        except (TypeError, ValueError):
            return None
        return max(num, 0)

    global_monthly = _coerce_limit(rest_defaults.get("max_monthly_exceptions")) if isinstance(rest_defaults, Mapping) else None
    global_consecutive = _coerce_limit(rest_defaults.get("max_consecutive_exceptions")) if isinstance(rest_defaults, Mapping) else None

    limits: dict[int, dict[str, int | None]] = {}
    for row in employees.itertuples(index=False):
        emp_id = str(getattr(row, "employee_id", "")).strip()
        if not emp_id:
            continue
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue

        monthly = _coerce_limit(getattr(row, "rest11h_max_monthly_exceptions", None))
        consecutive = _coerce_limit(getattr(row, "rest11h_max_consecutive_exceptions", None))

        if monthly is None:
            monthly = global_monthly
        if consecutive is None:
            consecutive = global_consecutive

        limits[emp_idx] = {"monthly": monthly, "consecutive": consecutive}

    return limits


def _build_rest_history_counts(
    bundle: Mapping[str, object], eid_of: Mapping[str, int]
) -> dict[tuple[int, str], int]:
    summary = bundle.get("history_month_to_date")
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return {}
    required = {"employee_id", "rest11_exceptions_count", "window_end_date"}
    if not required.issubset(summary.columns):
        return {}

    result: dict[tuple[int, str], int] = {}
    for row in summary.itertuples(index=False):
        emp_id = str(getattr(row, "employee_id", "")).strip()
        if not emp_id:
            continue
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue

        count = getattr(row, "rest11_exceptions_count", 0)
        try:
            count_val = int(round(float(count)))
        except (TypeError, ValueError):
            count_val = 0

        end_date_raw = getattr(row, "window_end_date", None)
        month_id = None
        if end_date_raw is not None and not pd.isna(end_date_raw):
            end_dt = pd.to_datetime(end_date_raw, errors="coerce")
            if pd.notna(end_dt):
                end_dt = end_dt.tz_localize(None) if getattr(end_dt, "tzinfo", None) else end_dt
                end_dt = end_dt.normalize()
                month_id = f"{end_dt:%Y-%m}"
        if month_id is None:
            continue

        result[(emp_idx, month_id)] = result.get((emp_idx, month_id), 0) + max(count_val, 0)

    return result


def _add_rest_constraints(
    context: ModelContext,
    model: cp_model.CpModel,
    assign_vars: Dict[tuple[int, int], cp_model.IntVar],
    bundle: Mapping[str, object],
) -> dict[str, Mapping]:
    result: dict[str, Mapping] = {
        "pair_violations": {},
        "day_flags": {},
        "monthly_totals": {},
        "streak_vars": {},
    }

    if not assign_vars:
        return result

    rest_threshold = _resolve_rest_threshold(context.cfg if isinstance(context.cfg, Mapping) else {})
    if rest_threshold <= 0:
        return result

    gap_pairs = context.gap_pairs
    if gap_pairs is None or gap_pairs.empty:
        return result

    if not {"s1_id", "s2_id", "gap_hours"}.issubset(gap_pairs.columns):
        return result

    sid_of: Mapping[int, int] = bundle.get("sid_of", {})  # type: ignore[assignment]
    if not sid_of:
        return result

    slot_date2: Mapping[int, int] = bundle.get("slot_date2", {})  # type: ignore[assignment]
    if not slot_date2:
        return result

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    eligible_eids: Mapping[int, Iterable[int]] = bundle.get("eligible_eids", {})  # type: ignore[assignment]
    if not eligible_eids:
        return result

    (
        _,
        _,
        _,
        day_month_map,
        _,
        _,
        _,
    ) = _extract_calendar_maps(context, bundle)

    rest_limits = _extract_rest11_limits(context, bundle)

    history_counts = _build_rest_history_counts(bundle, eid_of)

    eligible_sets: dict[int, set[int]] = {}
    for slot_idx, emp_iter in eligible_eids.items():
        if isinstance(emp_iter, set):
            eligible_sets[slot_idx] = emp_iter
        else:
            eligible_sets[slot_idx] = {int(emp) for emp in emp_iter}

    pair_map: dict[tuple[int, int], list[cp_model.BoolVar]] = {}
    month_pair_map: dict[tuple[int, str], list[cp_model.BoolVar]] = {}
    pair_vars: dict[tuple[int, int, int], cp_model.BoolVar] = {}

    work = gap_pairs.loc[:, ["s1_id", "s2_id", "gap_hours"]].copy()
    work = work.dropna(subset=["s1_id", "s2_id", "gap_hours"])
    if work.empty:
        return result

    work["gap_hours"] = pd.to_numeric(work["gap_hours"], errors="coerce")
    work = work[pd.notna(work["gap_hours"])]
    if work.empty:
        return result

    violation_pairs = work.loc[work["gap_hours"] < rest_threshold - 1e-6]
    if violation_pairs.empty:
        return result

    for row in violation_pairs.itertuples(index=False):
        try:
            s1_id = int(getattr(row, "s1_id"))
            s2_id = int(getattr(row, "s2_id"))
        except Exception:
            continue
        s1_idx = sid_of.get(s1_id)
        s2_idx = sid_of.get(s2_id)
        if s1_idx is None or s2_idx is None:
            continue

        day_idx = slot_date2.get(s2_idx)
        if day_idx is None:
            continue
        month_id = day_month_map.get(day_idx)

        elig1 = eligible_sets.get(s1_idx, set())
        elig2 = eligible_sets.get(s2_idx, set())
        if not elig1 or not elig2:
            continue
        common = elig1 & elig2
        if not common:
            continue

        for emp_idx in common:
            assign1 = assign_vars.get((emp_idx, s1_idx))
            assign2 = assign_vars.get((emp_idx, s2_idx))
            if assign1 is None or assign2 is None:
                continue

            var = model.NewBoolVar(f"rest11_violation_e{emp_idx}_s{s1_idx}_s{s2_idx}")
            model.Add(assign1 + assign2 - 1 <= var)
            model.Add(var <= assign1)
            model.Add(var <= assign2)

            pair_vars[(emp_idx, s1_idx, s2_idx)] = var

            day_key = (emp_idx, day_idx)
            pair_map.setdefault(day_key, []).append(var)

            if month_id is not None:
                month_pair_map.setdefault((emp_idx, month_id), []).append(var)

    num_employees = int(bundle.get("num_employees", len(eid_of)))
    num_days = int(bundle.get("num_days", len(slot_date2)))

    day_flags: dict[tuple[int, int], cp_model.BoolVar] = {}
    for emp_idx in range(num_employees):
        for day_idx in range(num_days):
            pair_list = pair_map.get((emp_idx, day_idx), [])
            day_var = model.NewBoolVar(f"rest11_violation_day_e{emp_idx}_d{day_idx}")
            if pair_list:
                for pair_var in pair_list:
                    model.Add(pair_var <= day_var)
                model.Add(day_var <= sum(pair_list))
            else:
                model.Add(day_var == 0)
            day_flags[(emp_idx, day_idx)] = day_var

    monthly_totals: dict[tuple[int, str], cp_model.IntVar] = {}
    for (emp_idx, month_id), pair_list in month_pair_map.items():
        limit = rest_limits.get(emp_idx, {}).get("monthly")
        upper = len(pair_list)
        total_var = model.NewIntVar(0, upper if upper > 0 else 0, f"rest11_month_total_e{emp_idx}_{month_id}")
        if pair_list:
            model.Add(total_var == sum(pair_list))
        else:
            model.Add(total_var == 0)
        history_count = history_counts.get((emp_idx, month_id), 0)
        if limit is not None:
            model.Add(total_var + history_count <= limit)
        monthly_totals[(emp_idx, month_id)] = total_var

    # handle months with history but no future pairs
    for (emp_idx, month_id), history_value in history_counts.items():
        limit = rest_limits.get(emp_idx, {}).get("monthly")
        if limit is None:
            continue
        if (emp_idx, month_id) in monthly_totals:
            continue
        total_var = model.NewIntVar(0, 0, f"rest11_month_total_e{emp_idx}_{month_id}")
        model.Add(total_var == 0)
        model.Add(total_var + history_value <= limit)
        monthly_totals[(emp_idx, month_id)] = total_var

    streak_vars: dict[tuple[int, int], cp_model.IntVar] = {}
    for emp_idx in range(num_employees):
        limits = rest_limits.get(emp_idx, {})
        consecutive_limit = limits.get("consecutive")
        if consecutive_limit is None:
            continue
        limit_int = max(int(consecutive_limit), 0)
        streak_cap = max(limit_int, 1)

        prev_streak: cp_model.IntVar | None = None
        for day_idx in range(num_days):
            day_var = day_flags[(emp_idx, day_idx)]
            streak_var = model.NewIntVar(0, streak_cap, f"rest11_streak_e{emp_idx}_d{day_idx}")
            model.Add(streak_var == 0).OnlyEnforceIf(day_var.Not())

            if prev_streak is None:
                model.Add(streak_var == 1).OnlyEnforceIf(day_var)
            else:
                model.Add(streak_var == prev_streak + 1).OnlyEnforceIf(day_var)

            model.Add(streak_var <= limit_int)

            streak_vars[(emp_idx, day_idx)] = streak_var
            prev_streak = streak_var

    result["pair_violations"] = pair_vars
    result["day_flags"] = day_flags
    result["monthly_totals"] = monthly_totals
    result["streak_vars"] = streak_vars
    return result


def _extract_rest_day_limits(
    context: ModelContext, bundle: Mapping[str, object]
) -> tuple[dict[int, dict[str, int]], int, int]:
    defaults = context.cfg.get("defaults") if isinstance(context.cfg, Mapping) else {}

    def _coerce(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        try:
            num = int(round(float(value)))
        except (TypeError, ValueError):
            return None
        return max(num, 0)

    weekly_default = _coerce(defaults.get("weekly_rest_min_days")) if isinstance(defaults, Mapping) else None
    if weekly_default is None:
        weekly_default = 0

    biweekly_default = _coerce(defaults.get("biweekly_rest_min_days")) if isinstance(defaults, Mapping) else None
    if biweekly_default is None:
        biweekly_default = 0

    limits: dict[int, dict[str, int]] = {}

    employees = context.employees
    if employees is None or employees.empty or "employee_id" not in employees.columns:
        return limits, weekly_default, biweekly_default

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    if not eid_of:
        return limits, weekly_default, biweekly_default

    week_cols = ("weekly_rest_min_days", "rest_week_min_days")
    biweek_cols = ("biweekly_rest_min_days", "rest_2week_min_days")

    for row in employees.itertuples(index=False):
        emp_id = str(getattr(row, "employee_id", "")).strip()
        if not emp_id:
            continue
        emp_idx = eid_of.get(emp_id)
        if emp_idx is None:
            continue

        weekly_val = None
        for col in week_cols:
            if hasattr(row, col):
                weekly_val = _coerce(getattr(row, col))
            if weekly_val is not None:
                break

        biweekly_val = None
        for col in biweek_cols:
            if hasattr(row, col):
                biweekly_val = _coerce(getattr(row, col))
            if biweekly_val is not None:
                break

        if weekly_val is None:
            weekly_val = weekly_default
        if biweekly_val is None:
            biweekly_val = biweekly_default

        limits[emp_idx] = {"weekly": int(weekly_val), "biweekly": int(biweekly_val)}

    return limits, weekly_default, biweekly_default


def _history_rest_dates_by_employee(
    history: pd.DataFrame | None,
    rest_codes: set[str],
    eid_of: Mapping[str, int],
    before_day: date | None = None,
) -> dict[int, set[date]]:
    if history is None or history.empty:
        return {}

    if "employee_id" not in history.columns:
        return {}

    shift_col = next(
        (
            col
            for col in ("turno", "shift_code", "shift", "codice")
            if col in history.columns
        ),
        None,
    )
    if shift_col is None:
        return {}

    date_col = next(
        (
            col
            for col in ("data", "date", "giorno", "day")
            if col in history.columns
        ),
        None,
    )
    if date_col is None:
        return {}

    work = history.loc[:, ["employee_id", shift_col, date_col]].copy()
    work["employee_id"] = work["employee_id"].astype(str).str.strip()
    work[shift_col] = work[shift_col].astype(str).str.strip().str.upper()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.date

    codes = {code.upper() for code in rest_codes}
    mask = work[shift_col].isin(codes) & work[date_col].notna()
    if before_day is not None:
        mask &= work[date_col] < before_day

    result: dict[int, set[date]] = defaultdict(set)
    for employee_id, day in work.loc[mask, ["employee_id", date_col]].itertuples(index=False):
        emp_idx = eid_of.get(employee_id)
        if emp_idx is None:
            continue
        result[emp_idx].add(day)
    return result


def _history_dates_by_employee(
    history: pd.DataFrame | None,
    eid_of: Mapping[str, int],
    before_day: date | None = None,
) -> dict[int, list[date]]:
    if history is None or history.empty:
        return {}

    if "employee_id" not in history.columns:
        return {}

    date_col = next(
        (
            col
            for col in ("data", "date", "giorno", "day")
            if col in history.columns
        ),
        None,
    )
    if date_col is None:
        return {}

    work = history.loc[:, ["employee_id", date_col]].copy()
    work["employee_id"] = work["employee_id"].astype(str).str.strip()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.date

    if before_day is not None:
        work = work.loc[work[date_col] < before_day]

    result: dict[int, set[date]] = defaultdict(set)
    for employee_id, day in work.itertuples(index=False):
        if pd.isna(day):
            continue
        emp_idx = eid_of.get(employee_id)
        if emp_idx is None:
            continue
        result[emp_idx].add(day)

    return {emp_idx: sorted(days) for emp_idx, days in result.items()}


def _count_history_dates(dates: list[date], start: date | None, end: date | None) -> int:
    if not dates or start is None or end is None:
        return 0
    left = bisect_left(dates, start)
    right = bisect_right(dates, end)
    return max(right - left, 0)


def _add_rest_day_windows(
    context: ModelContext,
    model: cp_model.CpModel,
    state_vars: Dict[tuple[int, int, str], cp_model.IntVar],
    bundle: Mapping[str, object],
    state_codes: Iterable[str],
) -> dict[str, Mapping]:
    result: dict[str, Mapping] = {
        "rest_day_flags": {},
        "weekly_violations": {},
    }

    rest_codes = tuple(code for code in state_codes if code.upper() in {"R", "F"})
    if not rest_codes:
        return result

    eid_of: Mapping[str, int] = bundle.get("eid_of", {})  # type: ignore[assignment]
    did_of: Mapping[object, int] = bundle.get("did_of", {})  # type: ignore[assignment]
    date_of: Mapping[int, object] = bundle.get("date_of", {})  # type: ignore[assignment]

    num_employees = int(bundle.get("num_employees", len(eid_of)))
    num_days = int(bundle.get("num_days", len(did_of)))

    if num_employees <= 0 or num_days <= 0:
        return result

    day_dates: list[date | None] = [
        _parse_date_value(date_of.get(day_idx)) for day_idx in range(num_days)
    ]

    horizon_cfg = context.cfg.get("horizon") if isinstance(context.cfg, Mapping) else None
    planning_start = (
        _parse_date_value(horizon_cfg.get("start_date"))
        if isinstance(horizon_cfg, Mapping)
        else None
    )
    if planning_start is None:
        planning_start = next((day for day in day_dates if day is not None), None)

    origin_idx = 0
    if planning_start is not None:
        for idx, current_day in enumerate(day_dates):
            if current_day == planning_start:
                origin_idx = idx
                break

    rest_limits, weekly_default, biweekly_default = _extract_rest_day_limits(context, bundle)

    history_sets = _history_rest_dates_by_employee(
        context.history, set(rest_codes), eid_of, before_day=planning_start
    )
    history_dates = {emp_idx: sorted(days) for emp_idx, days in history_sets.items()}
    history_coverage = _history_dates_by_employee(
        context.history, eid_of, before_day=planning_start
    )

    rest_day_vars: dict[tuple[int, int], cp_model.BoolVar] = {}
    for emp_idx in range(num_employees):
        for day_idx in range(num_days):
            var = model.NewBoolVar(f"is_rest_e{emp_idx}_d{day_idx}")
            rest_day_vars[(emp_idx, day_idx)] = var
            terms = [
                state_vars[(emp_idx, day_idx, code)]
                for code in rest_codes
                if (emp_idx, day_idx, code) in state_vars
            ]
            if terms:
                model.Add(var == sum(terms))
            else:
                model.Add(var == 0)

    weekly_window = 7
    biweekly_window = 14

    weekly_vars: dict[tuple[int, int], cp_model.BoolVar] = {}

    for emp_idx in range(num_employees):
        limits = rest_limits.get(emp_idx, {})
        weekly_req = limits.get("weekly", weekly_default) if limits else weekly_default
        biweekly_req = limits.get("biweekly", biweekly_default) if limits else biweekly_default

        weekly_req = max(int(weekly_req), 0)
        biweekly_req = max(int(biweekly_req), 0)

        history_list = history_dates.get(emp_idx, [])
        history_days = history_coverage.get(emp_idx, [])

        for end_idx in range(origin_idx, num_days):
            end_date = day_dates[end_idx]

            if weekly_req > 0:
                raw_start = end_idx - weekly_window + 1
                missing_days = max(0, origin_idx - raw_start)
                window_start_idx = max(raw_start, origin_idx)
                window_terms = [
                    rest_day_vars[(emp_idx, idx)]
                    for idx in range(max(window_start_idx, 0), end_idx + 1)
                ]
                history_count = 0
                if missing_days > 0 and end_date is not None:
                    start_date = end_date - timedelta(days=weekly_window - 1)
                    history_count = _count_history_dates(history_list, start_date, end_date)
                rest_sum = sum(window_terms) if window_terms else 0

                violation = model.NewBoolVar(f"weekly_rest_violation_e{emp_idx}_d{end_idx}")
                model.Add(rest_sum + history_count >= weekly_req).OnlyEnforceIf(violation.Not())
                model.Add(rest_sum + history_count <= weekly_req - 1).OnlyEnforceIf(violation)
                weekly_vars[(emp_idx, end_idx)] = violation

            if biweekly_req > 0:
                raw_start = end_idx - biweekly_window + 1
                missing_days = max(0, origin_idx - raw_start)
                window_start_idx = max(raw_start, origin_idx)
                window_terms = [
                    rest_day_vars[(emp_idx, idx)]
                    for idx in range(max(window_start_idx, 0), end_idx + 1)
                ]
                history_count = 0
                if end_date is None:
                    continue

                start_date = end_date - timedelta(days=biweekly_window - 1)
                coverage_days = len(window_terms)

                if missing_days > 0:
                    history_count = _count_history_dates(history_list, start_date, end_date)
                    coverage_from_history = _count_history_dates(
                        history_days, start_date, end_date
                    )
                    coverage_days += min(coverage_from_history, missing_days)

                if coverage_days < biweekly_window:
                    continue
                rest_sum = sum(window_terms) if window_terms else 0
                model.Add(rest_sum + history_count >= biweekly_req)

    result["rest_day_flags"] = rest_day_vars
    result["weekly_violations"] = weekly_vars
    return result
def _build_employee_role_map(employees: pd.DataFrame, bundle: Mapping[str, object]) -> dict[int, str]:
    """Restituisce la mappa emp_idx -> ruolo (uppercase)."""
    role_col = next((col for col in ("role", "ruolo") if col in employees.columns), None)
    if role_col is None:
        raise ValueError("employees DataFrame privo della colonna 'role'/'ruolo'.")
    if "employee_id" not in employees.columns:
        raise ValueError("employees DataFrame privo della colonna 'employee_id'.")

    eid_of: Mapping[str, int] = bundle["eid_of"]
    roles = employees[role_col].astype(str).str.strip().str.upper()

    mapping: dict[int, str] = {}
    for employee_id, role_u in zip(employees["employee_id"], roles, strict=False):
        idx = eid_of.get(employee_id)
        if idx is not None:
            mapping[idx] = role_u
    return mapping


def _iter_role_requirements(context: ModelContext, sid_of: Mapping[int, int]):
    """Generatore che produce tuple (slot_idx, role_u, demand)."""
    req_df = context.slot_requirements
    if req_df.empty:
        return
    for col in ("slot_id", "role", "demand"):
        if col not in req_df.columns:
            raise ValueError(f"slot_requirements deve contenere la colonna '{col}'.")

    work = req_df.loc[:, ["slot_id", "role", "demand"]].copy()
    work["role_u"] = work["role"].astype(str).str.strip().str.upper()

    for row in work.itertuples(index=False):
        slot_id = getattr(row, "slot_id")
        slot_idx = sid_of.get(slot_id)
        if slot_idx is None:
            continue
        try:
            demand = int(getattr(row, "demand"))
        except Exception:
            continue
        if demand <= 0:
            continue
        yield slot_idx, getattr(row, "role_u"), demand


def _iter_group_requirements(context: ModelContext, sid_of: Mapping[int, int]):
    """Generatore che produce tuple (slot_idx, role_set, total_staff, cap)."""
    if context.coverage_totals.empty:
        return

    slots = context.slots.loc[:, ["slot_id", "coverage_code", "shift_code", "reparto_id"]].copy()
    for col in ("coverage_code", "shift_code", "reparto_id"):
        if col not in slots.columns:
            raise ValueError(f"slots DataFrame privo della colonna '{col}'.")
        slots[col] = slots[col].astype(str).str.strip().str.upper()

    groups = context.coverage_totals.copy()
    for col in ("coverage_code", "shift_code", "reparto_id"):
        if col not in groups.columns:
            raise ValueError(f"coverage_totals deve contenere la colonna '{col}'.")
        groups[col] = groups[col].astype(str).str.strip().str.upper()

    need_col = "total_staff" if "total_staff" in groups.columns else ("min" if "min" in groups.columns else None)
    if need_col is None:
        raise ValueError("coverage_totals deve avere la colonna 'total_staff' (o 'min').")

    roles_col = next(
        (col for col in ("ruoli_totale_set", "ruoli_totale", "roles_total", "roles_group", "roles") if col in groups.columns),
        None,
    )
    if roles_col is None:
        raise ValueError("coverage_totals deve indicare i ruoli del gruppo (es. 'ruoli_totale').")

    cap_cols = [col for col in ("overstaff_cap_effective", "overstaff_cap", "max", "cap") if col in groups.columns]

    merged = groups.merge(
        slots,
        on=["coverage_code", "shift_code", "reparto_id"],
        how="left",
        validate="many_to_many",
        suffixes=("", "_slot"),
    )

    if merged["slot_id"].isna().any():
        missing = (
            merged.loc[merged["slot_id"].isna(), ["coverage_code", "shift_code", "reparto_id"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            "coverage_totals contiene combinazioni senza slot corrispondente: "
            f"{missing}"
        )

    merged["slot_id"] = merged["slot_id"].astype(int)
    merged["_role_set"] = merged[roles_col].apply(_parse_role_set)

    for row in merged.to_dict(orient="records"):
        slot_idx = sid_of.get(row["slot_id"])
        if slot_idx is None:
            continue
        role_set = row["_role_set"]
        if not role_set:
            continue
        need_raw = row[need_col]
        try:
            need = int(float(need_raw))
        except Exception:
            continue
        if need <= 0:
            continue

        cap_value = None
        for col in cap_cols:
            raw_cap = row.get(col)
            if raw_cap is None:
                continue
            text = str(raw_cap).strip()
            if not text or text.upper() == "NAN":
                continue
            try:
                cap_value = int(float(text))
            except Exception:
                continue
            else:
                break

        yield slot_idx, role_set, need, cap_value


def add_coverage_constraints(context: ModelContext, artifacts: ModelArtifacts) -> None:
    """
    Vincoli di copertura:
    - per ruolo: ogni slot deve avere almeno ``demand`` assegnazioni del ruolo richiesto;
    - per gruppo: ogni slot deve soddisfare il fabbisogno aggregato dei ruoli nel gruppo,
      rispettando opzionalmente il tetto di overstaffing.
    """
    model = artifacts.model
    x = artifacts.assign_vars
    bundle = context.bundle

    sid_of: Mapping[int, int] = bundle["sid_of"]
    eligible_eids: Mapping[int, Iterable[int]] = bundle["eligible_eids"]

    emp_role_map = _build_employee_role_map(context.employees, bundle)

    # Copertura per ruolo
    for slot_idx, role_u, demand in _iter_role_requirements(context, sid_of):
        candidates = [
            emp_idx
            for emp_idx in eligible_eids.get(slot_idx, [])
            if emp_role_map.get(emp_idx) == role_u
        ]
        expr = sum(x[(emp_idx, slot_idx)] for emp_idx in candidates)
        model.Add(expr >= demand)

    # Copertura per gruppi
    for slot_idx, role_set, need, cap in _iter_group_requirements(context, sid_of):
        candidates = [
            emp_idx
            for emp_idx in eligible_eids.get(slot_idx, [])
            if emp_role_map.get(emp_idx) in role_set
        ]
        expr = sum(x[(emp_idx, slot_idx)] for emp_idx in candidates)
        model.Add(expr >= need)
        if cap is not None:
            model.Add(expr <= cap)


def _fetch(store: Any, *candidates: str) -> Any:
    """Recupera il primo valore disponibile tra mapping/attributi."""
    for name in candidates:
        if isinstance(store, Mapping) and name in store:
            value = store[name]
            if value is not None:
                return value
        if hasattr(store, name):
            value = getattr(store, name)
            if value is not None:
                return value
    return None


def _require_frame(store: Any, *candidates: str) -> pd.DataFrame:
    value = _fetch(store, *candidates)
    if value is None:
        joined = ", ".join(candidates)
        raise ValueError(f"DataFrame mancante: {joined}")
    return value


def _optional_frame(store: Any, *candidates: str) -> pd.DataFrame:
    value = _fetch(store, *candidates)
    if value is None:
        return pd.DataFrame()
    return value


def build_context_from_data(data: Any, bundle: Mapping[str, object]) -> ModelContext:
    """
    Costruisce un ModelContext a partire dai DataFrame caricati dal loader
    (dict o dataclass ``LoadedData``) e dal ``bundle`` prodotto dal preprocessing.
    """
    cfg = _fetch(data, "cfg") or {}

    employees = _require_frame(data, "employees", "employees_df")
    slots = _require_frame(data, "shift_slots", "shift_slots_df")
    coverage_roles = _require_frame(
        data,
        "coverage_roles",
        "coverage_roles_df",
        "groups_role_min_expanded",
    )
    coverage_totals = _require_frame(
        data,
        "coverage_totals",
        "coverage_totals_df",
        "groups_total_expanded",
    )
    slot_requirements = _require_frame(
        data,
        "slot_requirements",
        "slot_requirements_df",
    )

    availability = _optional_frame(data, "availability", "availability_df")
    leaves = _optional_frame(data, "leaves", "leaves_df")
    history = _optional_frame(data, "history", "history_df")
    locks_must = _optional_frame(data, "locks_must", "locks_must_df")
    locks_forbid = _optional_frame(data, "locks_forbid", "locks_forbid_df")
    gap_pairs = _optional_frame(data, "gaps", "gap_pairs_df")
    calendars = _require_frame(data, "calendar", "calendars", "calendar_df")
    preassignments = _optional_frame(data, "preassignments", "preassignments_df")

    return ModelContext(
        cfg=dict(cfg),
        employees=employees,
        slots=slots,
        coverage_roles=coverage_roles,
        coverage_totals=coverage_totals,
        slot_requirements=slot_requirements,
        availability=availability,
        leaves=leaves,
        history=history,
        locks_must=locks_must,
        locks_forbid=locks_forbid,
        gap_pairs=gap_pairs,
        calendars=calendars,
        preassignments=preassignments if preassignments is not None else pd.DataFrame(),
        bundle=bundle,
    )
