from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from loader import LoadedData, LoaderError as _CoreLoaderError, load_all as _core_load_all
from loader.config import load_config as _core_load_config

from .model import ModelContext, build_context_from_data
from .preprocessing import build_all as _build_bundle


class LoaderError(Exception):
    """High-level loader wrapper error."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the YAML config using the original loader."""
    config_path = Path(path)
    try:
        return _core_load_config(str(config_path))
    except _CoreLoaderError as exc:  # pragma: no cover - delegated
        raise LoaderError(str(exc)) from exc


_ALIAS_MAP = {
    "employees": "employees_df",
    "shift_slots": "shift_slots_df",
    "shift_role_eligibility": "eligibility_df",
    "role_dept_pools": "role_dept_pools_df",
    "locks": "locks_df",
    "availability": "availability_df",
    "slot_requirements": "slot_requirements_df",
    "month_plan": "month_plan_df",
    "history": "history_df",
    "leaves": "leaves_df",
    "leaves_days": "leaves_days_df",
    "gaps": "gap_pairs_df",
    "preassignments": "preassignments_df",
}


@contextlib.contextmanager
def _temporary_config(cfg: Mapping[str, Any]) -> Path:
    tmp_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(dict(cfg), tmp_file, allow_unicode=True, sort_keys=False)
        tmp_file.flush()
        tmp_path = Path(tmp_file.name)
    finally:
        tmp_file.close()
    try:
        yield tmp_path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)


def _call_core_loader(config_path: Path, data_dir: Path) -> LoadedData:
    try:
        return _core_load_all(str(config_path), str(data_dir))
    except _CoreLoaderError as exc:  # pragma: no cover - delegated
        raise LoaderError(str(exc)) from exc


def _to_dataframe_dict(loaded: LoadedData | Mapping[str, Any]) -> dict[str, Any]:
    if is_dataclass(loaded):
        data = {field.name: getattr(loaded, field.name) for field in fields(loaded)}
    elif isinstance(loaded, Mapping):
        data = dict(loaded)
    else:  # pragma: no cover - defensive guard
        raise LoaderError("Unexpected structure returned by load_all")

    for alias, target in _ALIAS_MAP.items():
        if alias not in data and target in data:
            data[alias] = data[target]

    if "absences" not in data:
        abs_alias = _build_absences_alias(data)
        if abs_alias is not None:
            data["absences"] = abs_alias

    return data


def _build_absences_alias(data: Mapping[str, Any]):
    for key in ("absences", "leaves_df", "leaves_days_df"):
        frame = data.get(key)
        if frame is None or getattr(frame, "empty", False):
            continue
        if {"employee_id", "date"}.issubset(frame.columns):
            alias = frame.loc[:, [col for col in frame.columns if col in {"employee_id", "date", "kind", "tipo", "tipo_set"}]].copy()
            alias = alias.rename(columns={"tipo": "kind", "tipo_set": "kind"})
            if "kind" not in alias.columns:
                alias["kind"] = "full_day"
            return alias.drop_duplicates().reset_index(drop=True)
    return None


def load_all_data(
    cfg: Mapping[str, Any] | str | Path,
    data_dir: str | Path,
    *,
    as_dict: bool = True,
) -> dict[str, Any] | LoadedData:
    """Load all scheduling dataframes using the core loader."""

    data_path = Path(data_dir)
    if isinstance(cfg, Mapping):
        with _temporary_config(cfg) as tmp_cfg:
            loaded = _call_core_loader(tmp_cfg, data_path)
    else:
        loaded = _call_core_loader(Path(cfg), data_path)

    if as_dict:
        return _to_dataframe_dict(loaded)
    return loaded


def load_context(
    cfg: Mapping[str, Any] | str | Path,
    data_dir: str | Path,
) -> tuple[ModelContext, dict[str, Any], Mapping[str, object]]:
    """
    Carica tutti i dati, esegue il preprocessing e costruisce un ModelContext pronto per il solver.
    Restituisce (context, dataset_dict, bundle_preprocessing).
    """
    data = load_all_data(cfg, data_dir, as_dict=True)
    cfg_dict = data.get("cfg")
    if cfg_dict is None:
        raise LoaderError("Il caricamento dati non ha restituito la configurazione 'cfg'.")

    bundle = _build_bundle(data, cfg_dict)
    context = build_context_from_data(data, bundle)
    return context, data, bundle

