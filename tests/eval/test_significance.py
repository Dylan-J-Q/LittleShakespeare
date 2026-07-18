"""Tests for eval/significance.py's bootstrap confidence intervals."""
import math

import pytest

from little_shakespeare.eval.perplexity import NLLStats
from little_shakespeare.eval.significance import paired_bootstrap_difference_ci


def _uniform_blocks(n_blocks: int, nll_per_block: float, tokens_per_block: int = 10,
                     chars_per_block: int = 12) -> list:
    return [NLLStats(nll_per_block, tokens_per_block, chars_per_block) for _ in range(n_blocks)]


def test_identical_models_have_ci_including_zero():
    blocks_a = _uniform_blocks(80, nll_per_block=5.0)
    blocks_b = _uniform_blocks(80, nll_per_block=5.0)
    diff, lower, upper = paired_bootstrap_difference_ci(blocks_a, blocks_b, n_resamples=200)
    assert diff == 0.0
    assert lower <= 0.0 <= upper


def test_clearly_different_models_have_ci_excluding_zero():
    blocks_a = _uniform_blocks(200, nll_per_block=3.0)   # much lower loss, no variance
    blocks_b = _uniform_blocks(200, nll_per_block=8.0)   # much higher loss, no variance
    diff, lower, upper = paired_bootstrap_difference_ci(blocks_a, blocks_b, n_resamples=200)
    assert diff < 0  # A has lower bpc than B
    assert upper < 0  # CI entirely below zero -> a real, distinguishable difference


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap_difference_ci(_uniform_blocks(10, 5.0), _uniform_blocks(11, 5.0))


def test_same_seed_is_reproducible():
    blocks_a = _uniform_blocks(30, nll_per_block=4.0)
    blocks_b = _uniform_blocks(30, nll_per_block=5.0)
    result1 = paired_bootstrap_difference_ci(blocks_a, blocks_b, n_resamples=100, seed=42)
    result2 = paired_bootstrap_difference_ci(blocks_a, blocks_b, n_resamples=100, seed=42)
    assert result1 == result2
