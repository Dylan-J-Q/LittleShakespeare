"""Tests for data/splits.py's config-driven train/val/test split."""
import pytest

from little_shakespeare.config import PreprocessingConfig
from little_shakespeare.data.splits import split_text


def test_default_fractions_give_80_10_10():
    text = "x" * 1000
    train, val, test = split_text(text, PreprocessingConfig())
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_custom_fractions_are_honored():
    text = "x" * 1000
    config = PreprocessingConfig(train_fraction=0.6, val_fraction=0.2)
    train, val, test = split_text(text, config)
    assert len(train) == 600
    assert len(val) == 200
    assert len(test) == 200


def test_splits_are_contiguous_and_lossless():
    text = "the quick brown fox jumps over the lazy dog " * 30
    config = PreprocessingConfig(train_fraction=0.7, val_fraction=0.15)
    train, val, test = split_text(text, config)
    assert train + val + test == text


def test_fractions_summing_to_one_or_more_is_rejected():
    with pytest.raises(AssertionError):
        split_text("x" * 100, PreprocessingConfig(train_fraction=0.8, val_fraction=0.3))
