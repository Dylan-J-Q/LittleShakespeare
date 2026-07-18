"""Verbatim-memorization checks: is a generation reciting training text?

Small corpus + many training epochs (LittleShakespeare trains up to 300) is
exactly the regime where a model can memorize rather than generalize — see
Lesson 2. Word-level n-grams, not tokens, for the same tokenizer-invariance
reason as eval/diversity.py.
"""
from typing import FrozenSet, NamedTuple, Optional


class MemorizationReport(NamedTuple):
    overlap_fraction: float          # share of the generation's n-grams found in training text
    longest_match_words: int         # length (in words) of the longest verbatim match
    longest_match_text: Optional[str]


def training_ngrams(training_text: str, n: int = 8) -> FrozenSet[tuple]:
    """Precompute once per training corpus, reuse across many generations."""
    words = training_text.split()
    return frozenset(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def overlap_fraction(generated_text: str, reference_ngrams: FrozenSet[tuple], n: int = 8) -> float:
    """Share of the generation's n-grams that appear verbatim in reference_ngrams."""
    words = generated_text.split()
    if len(words) < n:
        return 0.0
    gen_ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    matches = sum(1 for g in gen_ngrams if g in reference_ngrams)
    return matches / len(gen_ngrams)


def longest_verbatim_match(generated_text: str, training_text: str, max_n: int = 25) -> tuple:
    """Longest run of consecutive words in generated_text that also appears,
    in that exact order, somewhere in training_text.

    A direct answer to "did it just copy a chunk of Shakespeare" — more
    interpretable than a fixed-n overlap fraction alone. O(max_n) n-gram-set
    builds over the training text; fine at this corpus's scale. (Match length
    is monotonic in n, so binary search would cut this to O(log max_n) if
    the corpus ever grows enough to matter.)
    """
    gen_words = generated_text.split()
    for n in range(min(max_n, len(gen_words)), 0, -1):
        candidates = training_ngrams(training_text, n)
        for i in range(len(gen_words) - n + 1):
            gram = tuple(gen_words[i:i + n])
            if gram in candidates:
                return n, " ".join(gram)
    return 0, None


def check_memorization(generated_text: str, training_text: str, n: int = 8) -> MemorizationReport:
    """Convenience wrapper combining the fixed-n overlap fraction and the
    longest-match search into one report."""
    frac = overlap_fraction(generated_text, training_ngrams(training_text, n), n)
    match_len, match_text = longest_verbatim_match(generated_text, training_text)
    return MemorizationReport(frac, match_len, match_text)
