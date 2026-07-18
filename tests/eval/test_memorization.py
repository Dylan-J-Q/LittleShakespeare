"""Tests for eval/memorization.py's verbatim-overlap checks."""
from little_shakespeare.eval.memorization import (
    check_memorization,
    longest_verbatim_match,
    overlap_fraction,
    training_ngrams,
)

TRAINING_TEXT = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune"
)


def test_verbatim_copy_has_full_overlap():
    generated = "to be or not to be that is the question whether"
    ngrams = training_ngrams(TRAINING_TEXT, n=4)
    assert overlap_fraction(generated, ngrams, n=4) == 1.0


def test_novel_text_has_zero_overlap():
    generated = "purple elephants dance beneath the shimmering violet moon tonight"
    ngrams = training_ngrams(TRAINING_TEXT, n=4)
    assert overlap_fraction(generated, ngrams, n=4) == 0.0


def test_longest_verbatim_match_finds_the_exact_copied_span():
    generated = "and then he said to be or not to be that is the question and left"
    length, text = longest_verbatim_match(generated, TRAINING_TEXT)
    assert length == 10  # "to be or not to be that is the question" — the longest run present verbatim
    assert text == "to be or not to be that is the question"


def test_no_match_returns_zero_and_none():
    length, text = longest_verbatim_match("completely unrelated novel content here", TRAINING_TEXT)
    assert length == 0
    assert text is None


def test_check_memorization_combines_both_signals():
    generated = "to be or not to be that is the question, indeed."
    report = check_memorization(generated, TRAINING_TEXT, n=4)
    assert report.overlap_fraction > 0
    assert report.longest_match_words > 0
    assert report.longest_match_text is not None


if __name__ == "__main__":
    test_verbatim_copy_has_full_overlap()
    test_novel_text_has_zero_overlap()
    test_longest_verbatim_match_finds_the_exact_copied_span()
    test_no_match_returns_zero_and_none()
    test_check_memorization_combines_both_signals()
    print("OK")
