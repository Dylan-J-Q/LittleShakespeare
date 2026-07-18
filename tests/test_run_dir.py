"""Tests for run_dir.py's path-resolution logic."""
from little_shakespeare.run_dir import VOCABS_ROOT, vocab_path


def test_vocab_path_uses_the_filename_stem_not_the_full_path():
    path = vocab_path("data/LittleShakespeare.txt", 256)
    assert path == VOCABS_ROOT / "LittleShakespeare" / "256.vocab"


def test_different_datasets_get_different_vocab_paths():
    a = vocab_path("data/LittleShakespeare.txt", 256)
    b = vocab_path("data/LittleShakespeareExtended.txt", 256)
    assert a != b
    assert a.parent != b.parent  # different dataset -> different subdirectory


def test_different_num_merges_get_different_vocab_paths_within_the_same_dataset():
    a = vocab_path("data/LittleShakespeare.txt", 256)
    b = vocab_path("data/LittleShakespeare.txt", 512)
    assert a != b
    assert a.parent == b.parent  # same dataset -> same subdirectory


if __name__ == "__main__":
    test_vocab_path_uses_the_filename_stem_not_the_full_path()
    test_different_datasets_get_different_vocab_paths()
    test_different_num_merges_get_different_vocab_paths_within_the_same_dataset()
    print("OK")
