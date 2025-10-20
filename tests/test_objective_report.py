from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

from src.model import build_model
from src.objective_report import compute_objective_breakdown, write_objective_breakdown_report
from tests.test_model_states import _make_basic_context


def test_objective_breakdown_and_report(tmp_path: Path) -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    slots = pd.DataFrame(
        {
            "slot_id": [1],
            "shift_code": ["M"],
            "date": [pd.Timestamp("2025-01-01")],
        }
    )
    pre_df = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "data": ["2025-01-01"],
            "state_code": ["M"],
        }
    )

    context = _make_basic_context(
        leaves,
        slots=slots,
        calendar_dates=[pd.Timestamp("2025-01-01")],
        cfg_extra={"preassignments": {"change_penalty_weight": 5.0}},
        preassignments_df=pre_df,
        preassignment_pairs=[(0, 0, "M")],
    )

    artifacts = build_model(context)
    state_var = artifacts.state_vars[(0, 0, "M")]
    artifacts.model.Add(state_var == 0)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    breakdown = compute_objective_breakdown(solver, artifacts)
    assert math.isclose(breakdown.total_objective, solver.ObjectiveValue())
    assert math.isclose(breakdown.total_contribution, breakdown.total_objective)

    assert breakdown.rows, "Expected at least one component in breakdown"
    preassign_row = next(
        row for row in breakdown.rows if row.component == "preassegnazioni"
    )
    assert math.isclose(preassign_row.violations, 1.0)
    assert math.isclose(preassign_row.contribution, breakdown.total_objective)
    assert preassign_row.violations_normalized is None
    assert breakdown.total_violations_normalized is None

    report_path = tmp_path / "objective_report.txt"
    written_path = write_objective_breakdown_report(breakdown, report_path)
    content = written_path.read_text(encoding="utf-8")
    assert "preassegnazioni" in content
    assert "Totale violazioni" in content
