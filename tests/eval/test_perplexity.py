"""Tests for eval/perplexity.py.

Pure-math checks against hand-constructed NLLStats (no model/GPU needed),
plus one integration check that accumulate_nll's totals actually match what
a tiny real model/loader produce.
"""
import math

from little_shakespeare.config import ModelConfig, PreprocessingConfig
from little_shakespeare.data.dataset import ShakespeareDataset
from little_shakespeare.data.tokenizer import CharTokenizer
from little_shakespeare.eval.perplexity import NLLStats, accumulate_nll, bits_per_char, perplexity
from little_shakespeare.model.transformer import TransformerModel
from torch.utils.data import DataLoader


def test_perplexity_of_uniform_distribution_equals_branching_factor():
    # A model that's exactly as uncertain as a uniform guess over K options
    # has per-token NLL = log(K) nats — perplexity should recover K exactly.
    k = 12
    stats = NLLStats(total_nll=100 * math.log(k), total_tokens=100, total_chars=100)
    assert math.isclose(perplexity(stats), k, rel_tol=1e-9)


def test_perplexity_matches_exp_of_average_nll():
    stats = NLLStats(total_nll=340.0, total_tokens=200, total_chars=800)
    assert math.isclose(perplexity(stats), math.exp(340.0 / 200), rel_tol=1e-9)


def test_bits_per_char_divides_by_characters_not_tokens():
    stats = NLLStats(total_nll=100.0, total_tokens=50, total_chars=250)
    expected = (100.0 / math.log(2)) / 250
    assert math.isclose(bits_per_char(stats), expected, rel_tol=1e-9)
    # Sanity: bpc must differ from the (wrong) per-token bits figure whenever
    # characters-per-token != 1, which is exactly why bpc needs its own divisor.
    per_token_bits = (100.0 / math.log(2)) / 50
    assert not math.isclose(bits_per_char(stats), per_token_bits)


def test_accumulate_nll_totals_match_a_tiny_real_model_and_loader():
    text = "to be or not to be that is the question " * 20
    preprocessing_config = PreprocessingConfig(block_size=8)
    tokenizer = CharTokenizer(text, preprocessing_config)
    dataset = ShakespeareDataset(text, tokenizer, preprocessing_config)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model_config = ModelConfig(
        embedding_dim=8, d_model=8, num_heads=2, ff_hidden_dim=16,
        num_layers=1, dropout_rate=0.0, max_pos_encoding_len=32,
    )
    model = TransformerModel(vocab_size=tokenizer.get_vocab_size(), config=model_config)

    stats = accumulate_nll(model, loader, tokenizer, device="cpu")

    expected_tokens = sum(y.numel() for _, y in loader)
    assert stats.total_tokens == expected_tokens
    assert stats.total_chars > 0
    assert math.isfinite(stats.total_nll)
    assert stats.total_nll > 0  # NLL of an untrained model is never exactly 0

    # And the two derived metrics should be finite, sane numbers off the same stats.
    assert math.isfinite(perplexity(stats))
    assert math.isfinite(bits_per_char(stats))


if __name__ == "__main__":
    test_perplexity_of_uniform_distribution_equals_branching_factor()
    test_perplexity_matches_exp_of_average_nll()
    test_bits_per_char_divides_by_characters_not_tokens()
    test_accumulate_nll_totals_match_a_tiny_real_model_and_loader()
    print("OK")
