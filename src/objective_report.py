from __future__ import annotations

"""Utility per calcolare e salvare il breakdown della funzione obiettivo."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from ortools.sat.python import cp_model

from .model import ModelArtifacts


@dataclass(frozen=True)
class ObjectiveBreakdownRow:
    component: str
    contribution: float
    contribution_pct: float
    violations: float
    violation_pct: float


@dataclass(frozen=True)
class ObjectiveBreakdown:
    total_objective: float
    total_contribution: float
    total_violations: float
    rows: tuple[ObjectiveBreakdownRow, ...]


def compute_objective_breakdown(
    solver: cp_model.CpSolver, artifacts: ModelArtifacts
) -> ObjectiveBreakdown:
    """Calcola il contributo di ciascun componente dell'obiettivo."""

    component_totals: "OrderedDict[str, dict[str, float]]" = OrderedDict()
    records = getattr(artifacts, "objective_terms", ())

    for record in records:
        var_value = float(solver.Value(record.var))
        if record.is_complement:
            violation = 1.0 - var_value
        else:
            violation = var_value
        if violation < 0:
            violation = 0.0

        contribution = violation * float(record.coeff)
        bucket = component_totals.setdefault(
            record.component, {"contribution": 0.0, "violations": 0.0}
        )
        bucket["contribution"] += contribution
        bucket["violations"] += violation

    total_objective = float(solver.ObjectiveValue())
    total_contribution = sum(item["contribution"] for item in component_totals.values())
    total_violations = sum(item["violations"] for item in component_totals.values())

    rows: list[ObjectiveBreakdownRow] = []
    for component, data in component_totals.items():
        contribution = data["contribution"]
        violations = data["violations"]
        contribution_pct = (contribution / total_objective * 100.0) if total_objective else 0.0
        violation_pct = (violations / total_violations * 100.0) if total_violations else 0.0
        rows.append(
            ObjectiveBreakdownRow(
                component=component,
                contribution=contribution,
                contribution_pct=contribution_pct,
                violations=violations,
                violation_pct=violation_pct,
            )
        )

    return ObjectiveBreakdown(
        total_objective=total_objective,
        total_contribution=total_contribution,
        total_violations=total_violations,
        rows=tuple(rows),
    )


def write_objective_breakdown_report(
    breakdown: ObjectiveBreakdown, path: Path | str
) -> Path:
    """Scrive il breakdown su un file di testo."""

    target_path = Path(path)

    lines: list[str] = [
        "Breakdown funzione obiettivo",
        f"Valore totale funzione obiettivo: {breakdown.total_objective:.6f}",
        f"Somma contributi calcolati: {breakdown.total_contribution:.6f}",
    ]

    delta = breakdown.total_objective - breakdown.total_contribution
    if abs(delta) > 1e-6:
        lines.append(f"Nota: differenza residua = {delta:.6f}")

    lines.append("")
    lines.append("Dettaglio per componente:")

    if not breakdown.rows:
        lines.append("(nessun termine registrato)")
    else:
        for row in breakdown.rows:
            lines.append(
                "- "
                f"{row.component}: contributo={row.contribution:.6f} "
                f"({row.contribution_pct:.2f}%), violazioni={row.violations:.6f} "
                f"({row.violation_pct:.2f}%)"
            )

    lines.append("")
    lines.append(
        f"Totale violazioni (non pesate): {breakdown.total_violations:.6f}"
    )

    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


__all__ = [
    "ObjectiveBreakdown",
    "ObjectiveBreakdownRow",
    "compute_objective_breakdown",
    "write_objective_breakdown_report",
]
