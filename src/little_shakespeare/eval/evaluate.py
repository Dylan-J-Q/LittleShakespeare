"""Evaluate a trained checkpoint's perplexity/bpc independently of training.

Usage:
    python -m little_shakespeare.eval.evaluate --index 3 --split test
    python -m little_shakespeare.eval.evaluate                    # latest model, all splits
"""
import argparse

from torch.utils.data import DataLoader

from little_shakespeare.checkpoint import load_run
from little_shakespeare.data.dataset import ShakespeareDataset
from little_shakespeare.eval.perplexity import accumulate_nll, bits_per_char, perplexity
from little_shakespeare.run_dir import resolve_model_id

SPLIT_NAMES = ("train", "val", "test")


def evaluate_checkpoint(model_id: int, splits=SPLIT_NAMES, batch_size: int = 32) -> dict:
    """Run perplexity/bpc for one checkpoint over the requested splits.

    ``load_run`` rebuilds the tokenizer/split from the run's own saved
    config.json rather than the caller's current config.py defaults — a
    checkpoint trained with num_merges=512 must be evaluated with that same
    vocab, not whatever the default happens to be today.
    """
    run = load_run(model_id)
    text_by_split = dict(zip(SPLIT_NAMES, (run.train_text, run.val_text, run.test_text)))

    results = {}
    for split in splits:
        dataset = ShakespeareDataset(text_by_split[split], run.tokenizer, run.preprocessing_config)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        stats = accumulate_nll(run.model, loader, run.tokenizer, run.device)
        results[split] = {
            "loss": stats.total_nll / stats.total_tokens,
            "perplexity": perplexity(stats),
            "bpc": bits_per_char(stats),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint's perplexity/bpc.")
    parser.add_argument("--index", type=int, default=None, help="Model id (default: latest)")
    parser.add_argument("--split", choices=[*SPLIT_NAMES, "all"], default="all")
    args = parser.parse_args()

    model_id = resolve_model_id(args.index)
    if model_id is None:
        raise SystemExit("No trained models found in models/.")

    splits = SPLIT_NAMES if args.split == "all" else (args.split,)
    results = evaluate_checkpoint(model_id, splits=splits)

    print(f"Model {model_id}:")
    for split, metrics in results.items():
        print(
            f"  {split:>5} | loss={metrics['loss']:.4f} | "
            f"perplexity={metrics['perplexity']:.3f} | bpc={metrics['bpc']:.4f}"
        )


if __name__ == "__main__":
    main()
