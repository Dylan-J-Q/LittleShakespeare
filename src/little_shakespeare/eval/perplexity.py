"""Perplexity and bits-per-character evaluation metrics.

Both are derived from a single token-weighted negative-log-likelihood
accumulation over a full ``DataLoader`` (see :func:`accumulate_nll`) rather
than an average of per-batch means — averaging batch means is only equal to
the true corpus-level average when every batch has an identical token count,
which a final partial batch violates.
"""
import math
from typing import List, NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from little_shakespeare.data.tokenizer import BaseTokenizer
from little_shakespeare.model.transformer import TransformerModel


class NLLStats(NamedTuple):
    """Raw accumulated totals; nothing derived yet. One instance can cover
    an entire loader (grand total) or a single example — same shape either
    way, since the sum of many per-example NLLStats is just their component-
    wise sum (see aggregate_nll_stats)."""
    total_nll: float     # summed negative log-likelihood, in nats
    total_tokens: int
    total_chars: int


def aggregate_nll_stats(stats: List[NLLStats]) -> NLLStats:
    """Component-wise sum of several NLLStats into one grand total."""
    return NLLStats(
        total_nll=sum(s.total_nll for s in stats),
        total_tokens=sum(s.total_tokens for s in stats),
        total_chars=sum(s.total_chars for s in stats),
    )


@torch.no_grad()
def accumulate_nll_per_example(model: TransformerModel, loader: DataLoader,
                                tokenizer: BaseTokenizer, device: str) -> List[NLLStats]:
    """One NLLStats per example (per dataset block), not per batch.

    This is the granularity statistical significance testing needs — each
    block is the natural independent evaluation unit; batches are just a
    GPU-efficiency grouping with no statistical meaning of their own.
    accumulate_nll's grand total is exactly this list, summed.
    """
    model.eval()
    per_example = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        token_nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='none'
        ).reshape(y.shape)  # (batch, seq) — back to per-position, undoing the flatten

        for row, targets in zip(token_nll, y):
            chars = len(tokenizer.decode(targets.tolist()))
            per_example.append(NLLStats(row.sum().item(), targets.numel(), chars))

    return per_example


def accumulate_nll(model: TransformerModel, loader: DataLoader,
                    tokenizer: BaseTokenizer, device: str) -> NLLStats:
    """Sum NLL, token count, and character count over an entire loader.

    Character counts come from decoding each batch's target tokens back to
    text — reusing the tokenizer's existing ``decode`` rather than tracking
    character offsets through the dataset separately.
    """
    return aggregate_nll_stats(accumulate_nll_per_example(model, loader, tokenizer, device))


def perplexity(stats: NLLStats) -> float:
    """exp(average NLL per token) — the model's effective branching factor."""
    return math.exp(stats.total_nll / stats.total_tokens)


def bits_per_char(stats: NLLStats) -> float:
    """Average bits needed per character — comparable across tokenizers."""
    total_bits = stats.total_nll / math.log(2)
    return total_bits / stats.total_chars


def metrics_report(stats: NLLStats) -> dict:
    """The loss/perplexity/bpc/token/char bundle metrics.json and compare.py
    both need — one place computing it so the two never quietly diverge."""
    return {
        "loss": stats.total_nll / stats.total_tokens,
        "perplexity": perplexity(stats),
        "bpc": bits_per_char(stats),
        "tokens": stats.total_tokens,
        "chars": stats.total_chars,
    }
