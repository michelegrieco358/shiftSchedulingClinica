from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, Mapping

from ortools.sat.python import cp_model

import pandas as pd


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
    preassign_must: pd.DataFrame
    preassign_forbid: pd.DataFrame
    gap_pairs: pd.DataFrame
    calendars: pd.DataFrame
    bundle: Mapping[str, object]

    def employees_for_slot(self, slot_id: int) -> Iterable[str]:
        """Restituisce gli employee_id candidati per lo slot indicato."""
        sid_map: Dict[int, int] = self.bundle["sid_of"]
        eligible_by_slot: Dict[int, Iterable[int]] = self.bundle["eligible_eids"]
        emp_map: Dict[int, str] = self.bundle["emp_of"]
        slot_idx = sid_map[slot_id]
        return (emp_map[eid_idx] for eid_idx in eligible_by_slot[slot_idx])



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


def build_model(context: ModelContext) -> ModelArtifacts:
    """Istanzia il modello CP-SAT e crea le variabili base di assegnazione."""
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
    prev_day_index = _build_previous_day_index(bundle)
    history_prev_night = _collect_prev_night_pairs(context.history, eid_of, did_of)

    restricted_after_night = tuple(
        state for state in ("R", "M", "F") if state in state_codes
    )

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

    return ModelArtifacts(
        model=model,
        assign_vars=assign_vars,
        state_vars=state_vars,
        employee_index=eid_of,
        slot_index=sid_of,
        day_index=did_of,
        state_codes=state_codes,
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
    preassign_must = _optional_frame(data, "preassign_must", "preassign_must_df")
    preassign_forbid = _optional_frame(data, "preassign_forbid", "preassign_forbid_df")
    gap_pairs = _optional_frame(data, "gaps", "gap_pairs_df")
    calendars = _require_frame(data, "calendar", "calendars", "calendar_df")

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
        preassign_must=preassign_must,
        preassign_forbid=preassign_forbid,
        gap_pairs=gap_pairs,
        calendars=calendars,
        bundle=bundle,
    )
