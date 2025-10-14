"""Esegue il caricamento dei dati e risolve il modello CP-SAT."""

from __future__ import annotations

from ortools.sat.python import cp_model

from src.solver import build_solver_from_sources


def main() -> None:
    model, artifacts, context, bundle = build_solver_from_sources("config.yaml", "data")

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    print("Solver status:", solver.StatusName(status))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return

    print("Objective value:", solver.ObjectiveValue())
    print("Assegnazioni candidate:", len(artifacts.assign_vars))
    print("Dipendenti considerati:", len(context.employees))


if __name__ == "__main__":
    main()

