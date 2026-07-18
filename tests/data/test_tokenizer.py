"""Minimal guard for the BPE tokenizer."""
import os

from little_shakespeare.config import PreprocessingConfig
from little_shakespeare.data.tokenizer import BPETokenizer
from little_shakespeare.run_dir import vocab_path as resolve_vocab_path


def test_bpe():
    text = "the theatre saw the theme of the throne. " * 50
    cfg = PreprocessingConfig(num_merges=40)
    vocab_path = str(resolve_vocab_path(cfg.data_path, cfg.num_merges))
    if os.path.exists(vocab_path):          # force fresh training
        os.remove(vocab_path)
    try:
        tok = BPETokenizer(text, cfg)

        # 1. Roundtrip: encode then decode must return the original text.
        assert tok.decode(tok.encode(text)) == text, "roundtrip failed"

        # 2. Merges must be real substrings, not integer-id soup like '6867'.
        #    (This is the bug that was fixed: keys must only contain source chars.)
        chars = set(text)
        assert all(set(k) <= chars for k in tok.vocab), "vocab has non-source chars"
        assert any(len(k) >= 3 for k in tok.vocab), "no multi-char merges happened"

        # 3. BPE must actually compress vs. raw characters.
        assert len(tok.encode(text)) < len(text), "no compression"
    finally:
        if os.path.exists(vocab_path):
            os.remove(vocab_path)


if __name__ == "__main__":
    test_bpe()
    print("OK")
