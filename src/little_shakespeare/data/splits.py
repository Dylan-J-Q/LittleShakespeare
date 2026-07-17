"""Deterministic train/val/test split over raw text, by config-driven fractions.

Splitting happens before tokenization (see CLAUDE.md's documented pipeline),
so this is a single source of truth both ``training.pipeline`` and
``eval.evaluate`` depend on, rather than two copies of the same slicing math
that could silently drift apart.
"""
from typing import Tuple

from little_shakespeare.config import PreprocessingConfig


def split_text(raw_text: str, config: PreprocessingConfig) -> Tuple[str, str, str]:
    """Return (train_text, val_text, test_text) per config.train_fraction/val_fraction.

    test_fraction is whatever's left — not stored separately, so it can't
    drift out of sync with the other two.
    """
    assert 0 < config.train_fraction < 1, f"train_fraction must be in (0, 1), got {config.train_fraction}"
    assert 0 < config.val_fraction < 1, f"val_fraction must be in (0, 1), got {config.val_fraction}"
    assert config.train_fraction + config.val_fraction < 1, (
        f"train_fraction ({config.train_fraction}) + val_fraction ({config.val_fraction}) "
        "must leave a positive share for test"
    )

    train_size = int(len(raw_text) * config.train_fraction)
    val_size = int(len(raw_text) * config.val_fraction)

    train_text = raw_text[:train_size]
    val_text = raw_text[train_size:train_size + val_size]
    test_text = raw_text[train_size + val_size:]
    return train_text, val_text, test_text
