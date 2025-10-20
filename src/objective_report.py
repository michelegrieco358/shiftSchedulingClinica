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
    violations_normalized: float | None = None
    violations_normalized_pct: float | None = None


@dataclass(frozen=True)
class ObjectiveBreakdown:
    total_objective: float
    total_contribution: float
    total_violations: float
    rows: tuple[ObjectiveBreakdownRow, ...]
    total_violations_normalized: float | None = None


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
            record.component,
            {
                "contribution": 0.0,
                "violations": 0.0,
                "normalized": 0.0,
                "normalized_count": 0.0,
            },
        )
        bucket["contribution"] += contribution
        bucket["violations"] += violation
        scale = record.unit_scale
        if scale is not None and scale > 0:
            bucket["normalized"] += violation / scale
            bucket["normalized_count"] += 1.0

    total_objective = float(solver.ObjectiveValue())
    total_contribution = sum(item["contribution"] for item in component_totals.values())
    total_violations = sum(item["violations"] for item in component_totals.values())
    has_normalized = any(
        item.get("normalized_count", 0.0) > 0.0 for item in component_totals.values()
    )
    total_normalized = (
        sum(
            item.get("normalized", 0.0)
            for item in component_totals.values()
            if item.get("normalized_count", 0.0) > 0.0
        )
        if has_normalized
        else 0.0
    )

    value_for_pct: dict[str, float] = {}
    total_for_pct = 0.0
    for component, data in component_totals.items():
        if data.get("normalized_count", 0.0) > 0.0:
            value = data.get("normalized", 0.0)
        else:
            value = data["violations"]
        value_for_pct[component] = value
        total_for_pct += value

    rows: list[ObjectiveBreakdownRow] = []
    for component, data in component_totals.items():
        contribution = data["contribution"]
        violations = data["violations"]
        contribution_pct = (contribution / total_objective * 100.0) if total_objective else 0.0
        violation_pct = (
            (value_for_pct.get(component, 0.0) / total_for_pct * 100.0) if total_for_pct else 0.0
        )
        normalized = None
        normalized_pct = None
        if data.get("normalized_count", 0.0) > 0.0:
            normalized = data.get("normalized", 0.0)
            normalized_pct = (normalized / total_normalized * 100.0) if total_normalized else 0.0
        rows.append(
            ObjectiveBreakdownRow(
                component=component,
                contribution=contribution,
                contribution_pct=contribution_pct,
                violations=violations,
                violation_pct=violation_pct,
                violations_normalized=normalized,
                violations_normalized_pct=normalized_pct,
            )
        )

    return ObjectiveBreakdown(
        total_objective=total_objective,
        total_contribution=total_contribution,
        total_violations=total_violations,
        rows=tuple(rows),
        total_violations_normalized=total_normalized if has_normalized else None,
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
            if row.violations_normalized is not None:
                entry = (
                    f"- {row.component}: contributo={row.contribution:.6f} "
                    f"({row.contribution_pct:.2f}%), violazioni_equivalenti={row.violations_normalized:.6f} "
                    f"({row.violation_pct:.2f}%)"
                )
                entry += f", violazioni_raw={row.violations:.6f}"
                if row.violations_normalized_pct is not None:
                    entry += f" (quota turni: {row.violations_normalized_pct:.2f}%)"
            else:
                entry = (
                    f"- {row.component}: contributo={row.contribution:.6f} "
                    f"({row.contribution_pct:.2f}%), violazioni={row.violations:.6f} "
                    f"({row.violation_pct:.2f}%)"
                )
            lines.append(entry)

    lines.append("")
    lines.append(
        f"Totale violazioni (non pesate): {breakdown.total_violations:.6f}"
    )
    if breakdown.total_violations_normalized is not None:
        lines.append(
            f"Totale violazioni equivalenti (turni medi): {breakdown.total_violations_normalized:.6f}"
        )

    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


__all__ = [
    "ObjectiveBreakdown",
    "ObjectiveBreakdownRow",
    "compute_objective_breakdown",
    "write_objective_breakdown_report",
]
