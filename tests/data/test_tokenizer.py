"""Minimal guard for the BPE tokenizer."""
from little_shakespeare import run_dir
from little_shakespeare.config import PreprocessingConfig
from little_shakespeare.data.tokenizer import BPETokenizer


def test_bpe(tmp_path, monkeypatch):
    # Isolate vocab caching under tmp_path — writing to the real vocabs/
    # tree would risk clobbering a genuine cached vocab trained with the
    # same num_merges against the default corpus.
    monkeypatch.setattr(run_dir, "VOCABS_ROOT", tmp_path)

    text = "the theatre saw the theme of the throne. " * 50
    cfg = PreprocessingConfig(num_merges=40)
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
