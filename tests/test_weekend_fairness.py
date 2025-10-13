from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from loader.calendar import build_calendar
from src.model import (
    WEEKEND_FAIRNESS_OBJECTIVE_SCALE,
    ModelContext,
    build_model,
)


cp_model = pytest.importorskip("ortools.sat.python.cp_model")


def _make_context(slot_dates: list[date], fairness_weight: float) -> ModelContext:
    if not slot_dates:
        raise ValueError("slot_dates must contain at least one day")

    employees = pd.DataFrame(
        {
            "employee_id": ["E1", "E2"],
            "role": ["INFERMIERE", "INFERMIERE"],
            "reparto_id": ["D1", "D1"],
            "weekend_weight": [1.0, 1.0],
        }
    )

    slot_ids = list(range(1, len(slot_dates) + 1))
    slots = pd.DataFrame(
        {
            "slot_id": slot_ids,
            "shift_code": ["M"] * len(slot_ids),
            "reparto_id": ["D1"] * len(slot_ids),
            "coverage_code": ["BASE"] * len(slot_ids),
            "date": slot_dates,
            "duration_min": [480] * len(slot_ids),
        }
    )

    horizon_start = min(slot_dates)
    horizon_end = max(slot_dates)
    calendar_df = build_calendar(horizon_start, horizon_end)

    calendar_dates = (
        pd.to_datetime(calendar_df["data"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
        .dt.date
        .tolist()
    )
    calendar_dates = sorted(dict.fromkeys(calendar_dates))

    did_of = {day: idx for idx, day in enumerate(calendar_dates)}
    date_of = {idx: day for day, idx in did_of.items()}

    eid_of = {"E1": 0, "E2": 1}
    emp_of = {idx: emp for emp, idx in eid_of.items()}
    sid_of = {slot_id: idx for idx, slot_id in enumerate(slot_ids)}
    slot_of = {idx: slot_id for slot_id, idx in sid_of.items()}

    eligible_eids = {sid_of[slot_id]: [0, 1] for slot_id in slot_ids}
    eligible_sids = {0: list(sid_of.values()), 1: list(sid_of.values())}
    slot_date2 = {
        sid_of[slot_id]: did_of[pd.to_datetime(slot_date).date()]
        for slot_id, slot_date in zip(slot_ids, slot_dates, strict=False)
    }

    cfg = {
        "horizon": {
            "start_date": horizon_start.isoformat(),
            "end_date": horizon_end.isoformat(),
        },
        "fairness": {"weekend_penalty_weight": fairness_weight},
    }

    empty_df = pd.DataFrame()
    history_df = pd.DataFrame(
        columns=["data", "employee_id", "turno", "is_weekend", "is_weekday_holiday"]
    )

    bundle = {
        "eid_of": eid_of,
        "emp_of": emp_of,
        "sid_of": sid_of,
        "slot_of": slot_of,
        "did_of": did_of,
        "date_of": date_of,
        "num_employees": len(eid_of),
        "num_slots": len(sid_of),
        "num_days": len(did_of),
        "eligible_eids": eligible_eids,
        "eligible_sids": eligible_sids,
        "slot_date2": slot_date2,
    }

    preassignments = pd.DataFrame(columns=["employee_id", "data", "state_code"])

    return ModelContext(
        cfg=cfg,
        employees=employees,
        slots=slots,
        coverage_roles=empty_df,
        coverage_totals=empty_df,
        slot_requirements=empty_df,
        availability=empty_df,
        leaves=empty_df,
        history=history_df,
        locks_must=empty_df,
        locks_forbid=empty_df,
        gap_pairs=empty_df,
        calendars=calendar_df,
        preassignments=preassignments,
        bundle=bundle,
    )


def _objective_fairness_coefficients(proto: cp_model.cp_model_pb2.CpModelProto) -> list[tuple[str, int]]:
    var_names = {idx: var.name for idx, var in enumerate(proto.variables)}
    return [
        (var_names.get(var_idx, ""), coeff)
        for var_idx, coeff in zip(proto.objective.vars, proto.objective.coeffs, strict=False)
        if "weekend_fair_dev" in var_names.get(var_idx, "")
    ]


def test_weekend_fairness_adds_objective_terms() -> None:
    context = _make_context(
        [date(2025, 1, 4), date(2025, 1, 5)],
        fairness_weight=1.5,
    )

    artifacts = build_model(context)
    proto = artifacts.model.Proto()

    fairness_terms = _objective_fairness_coefficients(proto)

    assert len(fairness_terms) == 2

    expected_coeff = int(round(1.5 * WEEKEND_FAIRNESS_OBJECTIVE_SCALE))
    assert expected_coeff > 0
    assert all(coeff == expected_coeff for _, coeff in fairness_terms)
    assert {name for name, _ in fairness_terms} == {
        "weekend_fair_dev_e0_D1",
        "weekend_fair_dev_e1_D1",
    }


def test_weekend_fairness_skips_departments_without_weekend_slots() -> None:
    context = _make_context([date(2025, 1, 6)], fairness_weight=2.0)

    artifacts = build_model(context)
    proto = artifacts.model.Proto()

    fairness_terms = _objective_fairness_coefficients(proto)
    assert fairness_terms == []

