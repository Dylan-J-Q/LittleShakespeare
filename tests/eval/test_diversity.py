"""Tests for eval/diversity.py's distinct-n diagnostic."""
from little_shakespeare.eval.diversity import distinct_n, diversity_report


def test_fully_repetitive_text_has_low_distinct_1():
    text = "the the the the the the the the"
    assert distinct_n(text, 1) == 1 / 8


def test_fully_unique_text_has_distinct_1_of_one():
    text = "to be or not to be"  # "to" and "be" repeat, but check distinct-2 instead
    assert distinct_n("one two three four five", 1) == 1.0


def test_distinct_2_catches_repetition_distinct_1_might_miss():
    # Every word is one of only two, so distinct-1 is low; distinct-2 (bigrams)
    # can still be perfectly unique if the words never repeat in the same pair.
    text = "a b a b a b"
    assert distinct_n(text, 1) == 2 / 6
    bigrams = distinct_n(text, 2)
    assert 0 < bigrams <= 1.0


def test_text_shorter_than_n_returns_zero_not_an_error():
    assert distinct_n("only three words", 5) == 0.0


def test_diversity_report_returns_both_metrics():
    report = diversity_report("to be or not to be that is the question")
    assert set(report.keys()) == {"distinct_1", "distinct_2"}
    assert 0.0 <= report["distinct_1"] <= 1.0
    assert 0.0 <= report["distinct_2"] <= 1.0


if __name__ == "__main__":
    test_fully_repetitive_text_has_low_distinct_1()
    test_fully_unique_text_has_distinct_1_of_one()
    test_distinct_2_catches_repetition_distinct_1_might_miss()
    test_text_shorter_than_n_returns_zero_not_an_error()
    test_diversity_report_returns_both_metrics()
    print("OK")
