from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

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

