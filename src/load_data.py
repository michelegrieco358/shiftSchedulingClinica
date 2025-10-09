from __future__ import annotations
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

class LoaderError(Exception):
    pass


def load_config(path: str | Path) -> dict:
    """Legge il file di configurazione YAML e restituisce un dizionario.
    Non effettua validazioni: i controlli restano ai loader specifici."""
    p = Path(path)
    if not p.exists():
        raise LoaderError(f"File di configurazione non trovato: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        raise LoaderError(f"Errore durante la lettura del config: {e}")

    return cfg


from loader import LoadedData, load_all  # importa il loader originale


def _ensure_config_path(cfg: dict | str | Path) -> tuple[Path, bool]:
    """Restituisce un percorso a un file di configurazione YAML.

    Se ``cfg`` è un dizionario viene creato un file temporaneo che verrà
    eliminato dal chiamante; la funzione restituisce anche un flag ``cleanup``
    che indica se il file va rimosso.
    """

    if isinstance(cfg, (str, Path)):
        config_path = Path(cfg)
        if not config_path.exists():
            raise LoaderError(f"File di configurazione non trovato: {config_path}")
        return config_path, False

    if isinstance(cfg, dict):
        try:
            with NamedTemporaryFile(
                "w", suffix=".yaml", encoding="utf-8", delete=False
            ) as tmp:
                yaml.safe_dump(cfg, tmp)
            return Path(tmp.name), True
        except Exception as exc:  # pragma: no cover - errore raro di I/O
            raise LoaderError(
                f"Impossibile creare un file temporaneo per la configurazione: {exc}"
            ) from exc

    raise LoaderError("cfg deve essere un percorso al file YAML o un dizionario")


def load_all_data(cfg: dict | str | Path, data_dir: str | Path) -> dict[str, Any]:
    """Carica tutti i DataFrame del progetto usando i loader originali."""

    config_path, cleanup = _ensure_config_path(cfg)

    try:
        loaded = load_all(str(config_path), str(data_dir))
    except Exception as exc:  # pragma: no cover - demandato ai test specifici
        raise LoaderError(f"load_all ha fallito: {exc}") from exc
    finally:
        if cleanup:
            try:
                config_path.unlink()
            except FileNotFoundError:
                pass

    if isinstance(loaded, LoadedData):
        # ``dataclasses.asdict`` effettua una deepcopy ricorsiva e con DataFrame di
        # grandi dimensioni può diventare estremamente costosa in termini di
        # memoria. Creiamo invece un dizionario superficiale mantenendo i
        # riferimenti originali.
        return {
            field: getattr(loaded, field)
            for field in loaded.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(loaded, dict):
        return loaded

    raise LoaderError(
        "load_all non ha restituito né LoadedData né un dizionario di DataFrame"
    )



