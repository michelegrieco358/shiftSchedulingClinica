from __future__ import annotations

"""Utility di alto livello per caricare i dati, costruire il modello CP-SAT
 e applicare i vincoli principali."""

from typing import Mapping, Tuple

from ortools.sat.python import cp_model

from .load_data import load_context
from .model import ModelArtifacts, ModelContext, add_coverage_constraints, build_model


class SolverBuildError(Exception):
    """Errore durante la costruzione del modello di scheduling."""


def build_solver_from_sources(
    cfg_path: str,
    data_dir: str,
) -> Tuple[cp_model.CpModel, ModelArtifacts, ModelContext, Mapping[str, object]]:
    """Carica i dati, costruisce il bundle e restituisce (model, artifacts, context, bundle)."""
    try:
        context, data, bundle = load_context(cfg_path, data_dir)
    except Exception as exc:  # pragma: no cover - superficie ridotta
        raise SolverBuildError(str(exc)) from exc

    artifacts = build_model(context)
    add_coverage_constraints(context, artifacts)

    return artifacts.model, artifacts, context, bundle


__all__ = [
    "SolverBuildError",
    "build_solver_from_sources",
]

