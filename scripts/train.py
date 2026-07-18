"""Entry point: train a model into a new models/<id>/ run directory.

    python scripts/train.py --note "doubled batch size to test VRAM headroom"
"""
import argparse

from little_shakespeare.training.pipeline import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model into a new models/<id>/ run directory.")
    parser.add_argument("--note", type=str, default=None,
                         help="Freeform hypothesis for this run, written into metrics.json.")
    args = parser.parse_args()

    run_training(note=args.note)
