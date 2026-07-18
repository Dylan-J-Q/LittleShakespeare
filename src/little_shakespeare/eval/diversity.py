"""Distinct-n: a diversity/repetition diagnostic for generated text.

Operates on whitespace-split words rather than raw tokens, deliberately —
computing this on BPE tokens would make it sensitive to num_merges the same
way raw perplexity is, defeating the point of a metric meant to sit
alongside tokenizer-invariant comparisons. See Lesson 2 (teaching/lessons)
for why this exists alongside perplexity/bpc rather than instead of them:
it catches repetition under free-running generation, a failure mode
teacher-forced perplexity cannot see.
"""
from typing import Dict


def distinct_n(text: str, n: int) -> float:
    """Fraction of a text's word n-grams that are unique (Li et al. 2016).

    Returns 0.0 for text shorter than n words (nothing to compute).
    """
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def diversity_report(text: str) -> Dict[str, float]:
    """distinct-1 and distinct-2 together — the pair standardly reported."""
    return {
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
    }
