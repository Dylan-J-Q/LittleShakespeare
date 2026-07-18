"""Bootstrap resampling: is a metric difference real, or noise from a small
held-out set?

Resamples over per-example NLLStats (see eval.perplexity.accumulate_nll_per_example)
rather than per-batch — each dataset block is the natural independent unit.
Not a new way to measure quality; a check on whether the metrics you already
have are trustworthy at the sample size you have them at. See Lesson 2.
"""
import random
from typing import Callable, List, Tuple

from little_shakespeare.eval.perplexity import NLLStats, aggregate_nll_stats, bits_per_char

MetricFn = Callable[[NLLStats], float]


def bootstrap_metric_ci(per_example_stats: List[NLLStats], metric_fn: MetricFn = bits_per_char,
                         n_resamples: int = 1000, confidence: float = 0.95,
                         seed: int = 0) -> Tuple[float, float, float]:
    """(point_estimate, lower, upper) for metric_fn over per_example_stats."""
    point = metric_fn(aggregate_nll_stats(per_example_stats))
    resampled = _resample_metric(per_example_stats, metric_fn, n_resamples, seed)
    lower, upper = _interval(resampled, confidence)
    return point, lower, upper


def paired_bootstrap_difference_ci(stats_a: List[NLLStats], stats_b: List[NLLStats],
                                    metric_fn: MetricFn = bits_per_char, n_resamples: int = 1000,
                                    confidence: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    """(point_diff, lower, upper) for metric_fn(A) - metric_fn(B).

    Resamples the SAME block indices for both models each round (paired),
    controlling for block-level difficulty variation rather than treating A
    and B as independent samples — a harder block inflates both models'
    loss together, and pairing cancels that shared noise out of the diff.
    """
    if len(stats_a) != len(stats_b):
        raise ValueError("stats_a and stats_b must cover the same blocks (equal length)")

    n = len(stats_a)
    point_diff = metric_fn(aggregate_nll_stats(stats_a)) - metric_fn(aggregate_nll_stats(stats_b))

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sample_a = [stats_a[i] for i in idx]
        sample_b = [stats_b[i] for i in idx]
        diffs.append(metric_fn(aggregate_nll_stats(sample_a)) - metric_fn(aggregate_nll_stats(sample_b)))

    lower, upper = _interval(sorted(diffs), confidence)
    return point_diff, lower, upper


def _resample_metric(stats: List[NLLStats], metric_fn: MetricFn, n_resamples: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    n = len(stats)
    values = []
    for _ in range(n_resamples):
        sample = [stats[rng.randrange(n)] for _ in range(n)]
        values.append(metric_fn(aggregate_nll_stats(sample)))
    values.sort()
    return values


def _interval(sorted_values: List[float], confidence: float) -> Tuple[float, float]:
    n = len(sorted_values)
    alpha = (1 - confidence) / 2
    lower = sorted_values[int(alpha * n)]
    upper = sorted_values[min(int((1 - alpha) * n), n - 1)]
    return lower, upper
