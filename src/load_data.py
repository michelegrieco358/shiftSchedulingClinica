from __future__ import annotations
from pathlib import Path
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


from loader import load_all  # importa il loader originale

def load_all_data(cfg: dict, data_dir: str) -> dict:
    """Carica tutti i DataFrame del progetto usando i loader originali."""
    dfs = load_all(cfg, data_dir)
    if not isinstance(dfs, dict):
        raise LoaderError("load_all non ha restituito un dizionario di DataFrame")
    return dfs



