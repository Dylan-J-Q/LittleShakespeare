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
BENCHMARKS_ROOT = Path("benchmarks")

CHECKPOINT_FILENAME = "model.pt"
CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.json"
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


def metrics_path(model_id: int) -> Path:
    return model_dir(model_id) / METRICS_FILENAME


def vocab_path(data_path: str, num_merges: int) -> Path:
    """vocabs/<dataset-stem>/<num_merges>.vocab — namespaced by which text
    file produced it. Keyed on num_merges alone (the old flat vocabs/<n>.vocab
    layout), swapping PreprocessingConfig.data_path to a different file would
    silently reuse a vocab trained on the previous dataset's statistics.
    Uses the filename stem, not the raw path, so this stays one clean
    directory level rather than mirroring data_path's own subdirectories."""
    dataset_name = Path(data_path).stem
    return VOCABS_ROOT / dataset_name / f"{num_merges}.vocab"
