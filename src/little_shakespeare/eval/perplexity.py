"""Perplexity and bits-per-character evaluation metrics.

Both are derived from a single token-weighted negative-log-likelihood
accumulation over a full ``DataLoader`` (see :func:`accumulate_nll`) rather
than an average of per-batch means — averaging batch means is only equal to
the true corpus-level average when every batch has an identical token count,
which a final partial batch violates.
"""
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from little_shakespeare.data.tokenizer import BaseTokenizer
from little_shakespeare.model.transformer import TransformerModel


class NLLStats(NamedTuple):
    """Raw accumulated totals a loader pass produces; nothing derived yet."""
    total_nll: float     # summed negative log-likelihood, in nats
    total_tokens: int
    total_chars: int


@torch.no_grad()
def accumulate_nll(model: TransformerModel, loader: DataLoader,
                    tokenizer: BaseTokenizer, device: str) -> NLLStats:
    """Sum NLL, token count, and character count over an entire loader.

    Character counts come from decoding each batch's target tokens back to
    text — reusing the tokenizer's existing ``decode`` rather than tracking
    character offsets through the dataset separately.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_chars = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        batch_nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='sum'
        )
        total_nll += batch_nll.item()
        total_tokens += y.numel()
        total_chars += sum(len(tokenizer.decode(seq.tolist())) for seq in y)

    return NLLStats(total_nll, total_tokens, total_chars)


def perplexity(stats: NLLStats) -> float:
    """exp(average NLL per token) — the model's effective branching factor."""
    return math.exp(stats.total_nll / stats.total_tokens)


def bits_per_char(stats: NLLStats) -> float:
    """Average bits needed per character — comparable across tokenizers."""
    total_bits = stats.total_nll / math.log(2)
    return total_bits / stats.total_chars
