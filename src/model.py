from __future__ import annotations

from dataclasses import dataclass
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
    employee_index: Mapping[str, int]
    slot_index: Mapping[int, int]


def build_model(context: ModelContext) -> ModelArtifacts:
    """Istanzia il modello CP-SAT e crea le variabili base di assegnazione."""
    model = cp_model.CpModel()

    bundle = context.bundle
    eid_of: Mapping[str, int] = bundle["eid_of"]
    sid_of: Mapping[int, int] = bundle["sid_of"]
    eligible_eids: Mapping[int, Iterable[int]] = bundle["eligible_eids"]

    assign_vars: Dict[tuple[int, int], cp_model.IntVar] = {}

    for slot_id in context.slots["slot_id"]:
        slot_idx = sid_of[slot_id]
        for emp_idx in eligible_eids[slot_idx]:
            var_name = f"x_e{emp_idx}_s{slot_idx}"
            assign_vars[(emp_idx, slot_idx)] = model.NewBoolVar(var_name)

    return ModelArtifacts(
        model=model,
        assign_vars=assign_vars,
        employee_index=eid_of,
        slot_index=sid_of,
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
