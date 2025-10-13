from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from loader.calendar import build_calendar
from ortools.sat.python import cp_model

from src.model import (
    CROSS_ASSIGNMENT_OBJECTIVE_SCALE,
    DUE_HOUR_OBJECTIVE_SCALE,
    ModelContext,
    build_model,
)


SUMMARY_COLUMNS = [
    "employee_id",
    "window_start_date",
    "window_end_date",
    "hours_worked_h",
    "absence_hours_h",
    "hours_with_leaves_h",
    "night_shifts_count",
    "rest11_exceptions_count",
]


def _make_history_summary(
    hours_with_leaves: float,
    month_start: str,
    *,
    window_days: int = 15,
) -> pd.DataFrame:
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    return pd.DataFrame(
        {
            "employee_id": ["E1"],
            "window_start_date": [pd.Timestamp(month_start)],
            "window_end_date": [
                pd.Timestamp(month_start) + pd.Timedelta(days=window_days - 1)
            ],
            "hours_worked_h": [hours_with_leaves],
            "absence_hours_h": [0.0],
            "hours_with_leaves_h": [hours_with_leaves],
            "night_shifts_count": [0],
            "rest11_exceptions_count": [0],
        }
    )


def _make_context(
    *,
    slot_specs: list[tuple[str, int]],
    due_hours: float,
    horizon_start: date,
    horizon_end: date,
    max_week_hours: float | None = None,
    max_month_hours: float | None = None,
    max_balance_delta_hours: float | None = None,
    history_entries: list[tuple[str, int]] | None = None,
    history_summary: pd.DataFrame | None = None,
    due_weight: float | None = None,
    final_balance_weight: float | None = None,
    full_day_hours_by_role: dict[str, float] | None = None,
    fallback_daily_hours: float = 7.5,
    cross_weight: float | None = None,
    employee_reparto: str = "HOME",
    slot_reparti: list[str] | None = None,
) -> ModelContext:
    employees_dict: dict[str, list] = {
        "employee_id": ["E1"],
        "role": ["INFERMIERE"],
        "ore_dovute_mese_h": [due_hours],
        "start_balance": [0.0],
    }
    employees_dict["max_week_hours_h"] = [max_week_hours]
    employees_dict["max_month_hours_h"] = [max_month_hours]
    if max_balance_delta_hours is not None:
        employees_dict["max_balance_delta_month_h"] = [max_balance_delta_hours]
    employees_dict["reparto_id"] = [employee_reparto]
    employees = pd.DataFrame(employees_dict)

    slot_ids = list(range(1, len(slot_specs) + 1))
    slot_dates = [pd.to_datetime(day).date() for day, _ in slot_specs]
    durations = [minutes for _, minutes in slot_specs]

    if slot_reparti is None:
        slot_reparti = [employee_reparto] * len(slot_specs)

    slots = pd.DataFrame(
        {
            "slot_id": slot_ids,
            "shift_code": ["M"] * len(slot_ids),
            "date": slot_dates,
            "duration_min": durations,
            "reparto_id": slot_reparti,
        }
    )

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

    eid_of = {"E1": 0}
    emp_of = {0: "E1"}
    sid_of = {slot_id: idx for idx, slot_id in enumerate(slot_ids)}
    slot_of = {idx: slot_id for slot_id, idx in sid_of.items()}

    slot_date2 = {sid_of[slot_id]: did_of[slot_date] for slot_id, slot_date in zip(slot_ids, slot_dates, strict=False)}
    slot_duration_min = {slot_id: duration for slot_id, duration in zip(slot_ids, durations, strict=False)}
    slot_reparto = {
        slot_id: str(reparto).strip().upper()
        for slot_id, reparto in zip(slot_ids, slot_reparti, strict=False)
        if str(reparto).strip()
    }

    eligible_eids = {sid_of[slot_id]: [0] for slot_id in slot_ids}

    history_df = pd.DataFrame(
        history_entries,
        columns=["data", "shift_duration_min"],
    ) if history_entries else pd.DataFrame(columns=["data", "shift_duration_min"])
    if not history_df.empty:
        history_df["employee_id"] = "E1"

    summary_df = history_summary if history_summary is not None else pd.DataFrame(columns=SUMMARY_COLUMNS)

    absences_cfg: dict[str, object] = {"count_as_worked_hours": True}
    if full_day_hours_by_role is not None:
        absences_cfg["full_day_hours_by_role_h"] = full_day_hours_by_role
    if fallback_daily_hours is not None:
        absences_cfg["fallback_contract_daily_avg_h"] = fallback_daily_hours

    defaults_cfg: dict[str, object] = {
        "contract_hours_by_role_h": {"INFERMIERE": due_hours},
        "absences": absences_cfg,
    }
    balance_cfg: dict[str, float] = {}
    if due_weight is not None:
        balance_cfg["due_hours_penalty_weight"] = due_weight
    if final_balance_weight is not None:
        balance_cfg["final_balance_penalty_weight"] = final_balance_weight
    if balance_cfg:
        defaults_cfg["balance"] = balance_cfg

    cfg = {
        "horizon": {
            "start_date": horizon_start.isoformat(),
            "end_date": horizon_end.isoformat(),
        },
        "defaults": defaults_cfg,
    }
    if cross_weight is not None:
        cfg["cross"] = {"penalty_weight": cross_weight}

    empty_df = pd.DataFrame()

    bundle = {
        "eid_of": eid_of,
        "emp_of": emp_of,
        "sid_of": sid_of,
        "slot_of": slot_of,
        "did_of": did_of,
        "date_of": date_of,
        "num_employees": 1,
        "num_slots": len(slot_ids),
        "num_days": len(did_of),
        "eligible_eids": eligible_eids,
        "slot_date2": slot_date2,
        "slot_duration_min": slot_duration_min,
        "history_month_to_date": summary_df,
        "slot_reparto": slot_reparto,
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


def test_weekly_hours_cap_blocks_overassignment() -> None:
    context = _make_context(
        slot_specs=[("2025-01-06", 360), ("2025-01-07", 360)],
        due_hours=168,
        horizon_start=date(2025, 1, 6),
        horizon_end=date(2025, 1, 7),
        max_week_hours=10,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_weekly_hours_cap_includes_history_before_horizon() -> None:
    context = _make_context(
        slot_specs=[("2025-01-08", 300)],
        due_hours=168,
        horizon_start=date(2025, 1, 8),
        horizon_end=date(2025, 1, 9),
        max_week_hours=10,
        history_entries=[("2025-01-06", 360)],
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_monthly_hours_cap_blocks_overassignment() -> None:
    context = _make_context(
        slot_specs=[("2025-01-02", 360), ("2025-01-03", 360), ("2025-01-04", 360)],
        due_hours=168,
        horizon_start=date(2025, 1, 2),
        horizon_end=date(2025, 1, 4),
        max_month_hours=10,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)
    model.Add(artifacts.assign_vars[(0, 2)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_monthly_due_slack_reflects_difference() -> None:
    context = _make_context(
        slot_specs=[("2025-01-02", 180), ("2025-01-03", 120)],
        due_hours=10,
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 31),
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    assert artifacts.monthly_hour_under_slack
    key = next(iter(artifacts.monthly_hour_under_slack.keys()))
    under_slack = artifacts.monthly_hour_under_slack[key]
    over_slack = artifacts.monthly_hour_over_slack[key]
    balance = artifacts.monthly_hour_balance[key]

    assert solver.Value(under_slack) == 300
    assert solver.Value(over_slack) == 0
    assert solver.Value(balance) == -300


def test_monthly_due_includes_history_summary() -> None:
    summary = _make_history_summary(hours_with_leaves=6.0, month_start="2025-01-01")
    context = _make_context(
        slot_specs=[("2025-01-16", 240), ("2025-01-17", 240)],
        due_hours=20,
        horizon_start=date(2025, 1, 16),
        horizon_end=date(2025, 1, 31),
        history_summary=summary,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    assert artifacts.monthly_hour_under_slack
    key = next(iter(artifacts.monthly_hour_under_slack.keys()))
    under_slack = artifacts.monthly_hour_under_slack[key]
    over_slack = artifacts.monthly_hour_over_slack[key]
    balance = artifacts.monthly_hour_balance[key]

    # History contributes 360 minutes, plan 480 minutes => deficit 360 minutes
    assert solver.Value(under_slack) == 360
    assert solver.Value(over_slack) == 0
    assert solver.Value(balance) == -360


def test_monthly_balance_delta_blocks_excess_variation() -> None:
    context = _make_context(
        slot_specs=[("2025-01-05", 600), ("2025-01-06", 600)],
        due_hours=10,
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 31),
        max_balance_delta_hours=6.0,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_monthly_balance_delta_allows_within_limit() -> None:
    context = _make_context(
        slot_specs=[("2025-01-05", 600), ("2025-01-06", 600)],
        due_hours=10,
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 31),
        max_balance_delta_hours=12.0,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_monthly_due_skipped_for_partial_month_without_history() -> None:
    context = _make_context(
        slot_specs=[("2025-01-05", 180)],
        due_hours=10,
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 15),
    )

    artifacts = build_model(context)

    assert not artifacts.monthly_hour_under_slack


def test_monthly_due_skipped_when_history_missing_for_early_gap() -> None:
    context = _make_context(
        slot_specs=[("2025-01-20", 180)],
        due_hours=10,
        horizon_start=date(2025, 1, 16),
        horizon_end=date(2025, 1, 31),
    )

    artifacts = build_model(context)

    assert not artifacts.monthly_hour_under_slack


def test_monthly_due_skipped_when_history_not_long_enough() -> None:
    summary = _make_history_summary(
        hours_with_leaves=5.0,
        month_start="2025-01-01",
        window_days=10,
    )
    context = _make_context(
        slot_specs=[("2025-01-20", 180)],
        due_hours=10,
        horizon_start=date(2025, 1, 16),
        horizon_end=date(2025, 1, 31),
        history_summary=summary,
    )

    artifacts = build_model(context)

    assert not artifacts.monthly_hour_under_slack


def test_due_hour_objective_prefers_matching_due_hours() -> None:
    context = _make_context(
        slot_specs=[("2025-01-02", 480), ("2025-01-03", 120)],
        due_hours=10.0,
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 31),
        due_weight=3.0,
        full_day_hours_by_role={"INFERMIERE": 7.5},
    )

    artifacts = build_model(context)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)

    assert status == cp_model.OPTIMAL

    under_eff = next(iter(artifacts.monthly_hour_under_effective.values()))
    first_slot = artifacts.assign_vars[(0, 0)]
    second_slot = artifacts.assign_vars[(0, 1)]

    assert solver.Value(first_slot) == 1
    assert solver.Value(second_slot) == 1
    assert solver.Value(under_eff) == 0
    assert all(solver.Value(var) == 0 for var in artifacts.monthly_hour_over_effective.values())

    model_proto = artifacts.model.Proto()
    target_index = next(
        idx
        for idx, variable in enumerate(model_proto.variables)
        if variable.name == under_eff.Name()
    )
    coeff_value = None
    for var_ref, coeff in zip(model_proto.objective.vars, model_proto.objective.coeffs, strict=False):
        if var_ref == target_index:
            coeff_value = coeff
            break

    expected_coeff = int(
        round((3.0 * 1.5 / (7.5 * 60.0)) * DUE_HOUR_OBJECTIVE_SCALE)
    )
    assert coeff_value == expected_coeff


def test_cross_assignment_penalty_counts_cross_slots() -> None:
    cross_weight = 2.5
    context = _make_context(
        slot_specs=[("2025-01-06", 480)],
        due_hours=0,
        horizon_start=date(2025, 1, 6),
        horizon_end=date(2025, 1, 6),
        cross_weight=cross_weight,
        employee_reparto="HOME",
        slot_reparti=["OTHER"],
    )

    artifacts = build_model(context)
    artifacts.model.Add(artifacts.assign_vars[(0, 0)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)

    assert status == cp_model.OPTIMAL

    expected = int(round(cross_weight * CROSS_ASSIGNMENT_OBJECTIVE_SCALE))
    assert solver.ObjectiveValue() == pytest.approx(expected)


def test_final_balance_variables_require_full_month() -> None:
    partial_context = _make_context(
        slot_specs=[("2025-01-15", 480)],
        due_hours=10.0,
        horizon_start=date(2025, 1, 10),
        horizon_end=date(2025, 1, 31),
    )

    partial_artifacts = build_model(partial_context)
    assert partial_artifacts.final_hour_balance == {}
    assert partial_artifacts.final_hour_balance_abs == {}

    full_context = _make_context(
        slot_specs=[("2025-01-15", 480)],
        due_hours=10.0,
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 31),
    )

    full_artifacts = build_model(full_context)
    assert (0, "final") in full_artifacts.final_hour_balance
    assert (0, "final") in full_artifacts.final_hour_balance_abs


def test_final_balance_objective_scales_with_role_minutes() -> None:
    weight = 4.0
    context = _make_context(
        slot_specs=[("2025-01-15", 480)],
        due_hours=10.0,
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 31),
        final_balance_weight=weight,
        full_day_hours_by_role={"INFERMIERE": 8.0},
    )

    artifacts = build_model(context)
    assert len(artifacts.final_hour_balance_abs) == 1
    final_abs = next(iter(artifacts.final_hour_balance_abs.values()))

    model_proto = artifacts.model.Proto()
    target_index = next(
        idx
        for idx, variable in enumerate(model_proto.variables)
        if variable.name == final_abs.Name()
    )

    coeff_value = None
    for var_ref, coeff in zip(model_proto.objective.vars, model_proto.objective.coeffs, strict=False):
        if var_ref == target_index:
            coeff_value = coeff
            break

    expected_coeff = int(
        round((weight * 1.5 / (8.0 * 60.0)) * DUE_HOUR_OBJECTIVE_SCALE)
    )
    assert coeff_value == expected_coeff
