"""Esegue il caricamento dei dati e risolve il modello CP-SAT con logging del gap."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings
from typing import Iterable, Mapping, Optional

from ortools.sat.python import cp_model
import pandas as pd

from src.objective_report import (
    compute_objective_breakdown,
    write_objective_breakdown_report,
)
from src.solver import build_solver_from_sources


class GapLoggingCallback(cp_model.CpSolverSolutionCallback):
    """Stampa il gap corrente solo quando viene trovata una nuova soluzione."""

    def __init__(self) -> None:
        super().__init__()
        self._last_objective: Optional[float] = None

    def OnSolutionCallback(self) -> None:  # pragma: no cover - runtime callback
        objective = self.ObjectiveValue()
        bound = self.BestObjectiveBound()

        if self._last_objective is not None and objective == self._last_objective:
            return

        self._last_objective = objective
        if objective == bound:
            gap = 0.0
        elif objective != 0:
            gap = abs(objective - bound) / abs(objective)
        else:
            gap = float("inf")

        print(
            f"Nuova soluzione: objective={objective:.6g}  bound={bound:.6g}  gap={gap:.4%}"
        )


def _build_state_table(
    solver: cp_model.CpSolver,
    artifacts: ModelArtifacts,
    bundle: Mapping[str, object],
) -> pd.DataFrame:
    emp_of: Mapping[int, str] = bundle.get("emp_of", {})  # type: ignore[assignment]
    date_of: Mapping[int, object] = bundle.get("date_of", {})  # type: ignore[assignment]

    if not emp_of or not date_of:
        raise RuntimeError("Bundle privo delle mappe necessarie (emp_of/date_of).")

    state_codes: Iterable[str] = artifacts.state_codes

    rows: list[dict[str, object]] = []
    for emp_idx in sorted(emp_of.keys()):
        employee_id = emp_of[emp_idx]
        for day_idx in sorted(date_of.keys()):
            day = date_of[day_idx]
            state_value = ""
            for code in state_codes:
                var = artifacts.state_vars.get((emp_idx, day_idx, code))
                if var is None:
                    continue
                if solver.Value(var):
                    state_value = code
                    break
            rows.append(
                {
                    "employee_id": employee_id,
                    "date": str(day),
                    "state": state_value,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["date", "employee_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Risolvi il modello CP-SAT caricando dati e config specificati."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Percorso al file di configurazione YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory dei CSV di input (default: data).",
    )
    args = parser.parse_args()

    # Ignora l'avviso informativo sui locks mancanti, utile in ambiente POC.
    warnings.filterwarnings(
        "ignore",
        message=r"locks\.csv non trovato: caricati 0 record da locks\.csv",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"locks\.csv: caricati \d+ record",
        category=UserWarning,
    )

    model, artifacts, context, bundle = build_solver_from_sources(
        args.config,
        args.data_dir,
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600
    callback = GapLoggingCallback()

    status = solver.SolveWithSolutionCallback(model, callback)

    print("Solver status:", solver.StatusName(status))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return

    print("Objective value:", solver.ObjectiveValue())
    print("Assegnazioni candidate:", len(artifacts.assign_vars))
    print("Dipendenti considerati:", len(context.employees))

    breakdown = compute_objective_breakdown(solver, artifacts)
    report_path = Path("objective_breakdown.txt")
    write_objective_breakdown_report(breakdown, report_path)
    print(f"Report funzione obiettivo salvato in: {report_path}")

    try:
        state_df = _build_state_table(solver, artifacts, bundle)
    except RuntimeError as exc:  # pragma: no cover - bundle inconsistente
        print(f"Impossibile generare il CSV delle assegnazioni: {exc}")
    else:
        states_path = Path("states.csv")
        state_df.to_csv(states_path, index=False)
        print(f"Assegnazioni (stati giornalieri) esportate in: {states_path}")


if __name__ == "__main__":
    main()
