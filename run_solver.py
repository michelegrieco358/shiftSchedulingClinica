"""Esegue il caricamento dei dati e risolve il modello CP-SAT con logging del gap."""

from __future__ import annotations

import warnings
from typing import Optional

from ortools.sat.python import cp_model

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


def main() -> None:
    # Ignora l'avviso informativo sui locks mancanti, utile in ambiente POC.
    warnings.filterwarnings(
        "ignore",
        message=r"locks\.csv non trovato: caricati 0 record da locks\.csv",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"locks\.csv: caricati 0 record",
        category=UserWarning,
    )

    model, artifacts, context, bundle = build_solver_from_sources("config.yaml", "data")

    solver = cp_model.CpSolver()
    callback = GapLoggingCallback()

    status = solver.SolveWithSolutionCallback(model, callback)

    print("Solver status:", solver.StatusName(status))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return

    print("Objective value:", solver.ObjectiveValue())
    print("Assegnazioni candidate:", len(artifacts.assign_vars))
    print("Dipendenti considerati:", len(context.employees))


if __name__ == "__main__":
    main()
