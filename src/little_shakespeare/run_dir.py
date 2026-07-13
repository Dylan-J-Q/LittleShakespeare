"""Run-directory conventions and id resolution.

Single source of truth for the ``models/<id>/`` layout, deduplicating the
id-scanning logic that previously lived in both ``main.py`` and ``inference.py``.
Using :class:`pathlib.Path` throughout also fixes the mixed-separator paths that
``f"models/{id}"`` produced on Windows.
"""
from pathlib import Path
from typing import List, Optional

MODELS_ROOT = Path("models")
VOCABS_ROOT = Path("vocabs")

CHECKPOINT_FILENAME = "model.pt"
CONFIG_FILENAME = "config.json"
LOG_FILENAME = "training.log"
CSV_LOG_FILENAME = "training_log.csv"


def _numeric_ids() -> List[int]:
    if not MODELS_ROOT.exists():
        return []
    return sorted(int(p.name) for p in MODELS_ROOT.iterdir() if p.is_dir() and p.name.isdigit())


def get_next_model_id() -> int:
    """Next unused integer id for a new run directory (0 if none exist)."""
    ids = _numeric_ids()
    return ids[-1] + 1 if ids else 0


def resolve_model_id(index: Optional[int] = None) -> Optional[int]:
    """Return ``index`` if given, else the highest existing id, else ``None``."""
    if index is not None:
        return index
    ids = _numeric_ids()
    return ids[-1] if ids else None


def model_dir(model_id: int) -> Path:
    return MODELS_ROOT / str(model_id)


def checkpoint_path(model_id: int) -> Path:
    return model_dir(model_id) / CHECKPOINT_FILENAME


def config_path(model_id: int) -> Path:
    return model_dir(model_id) / CONFIG_FILENAME
