"""Scan every models/<id>/ run and write benchmarks/leaderboard.{md,csv}.

Sorted by val_bpc — the only column that's fair across runs with different
num_merges. Deliberately fast: reads config.json/metrics.json/training_log.csv
off disk only, never loads a checkpoint or runs the model (that's compare.py's
job). Runs missing data show "—" rather than a silently backfilled/estimated
value — see Lesson 2 in teaching/lessons for why that's the honest choice.

Every field metrics.json records is surfaced here — val AND test splits,
epoch/early-stopping info, training time, git provenance — not just val
perplexity/bpc. Runs without a metrics.json (pre-Wave-2, or mid-Wave-2 runs
like an early one that only got as far as the CSV bpc column) show "—" for
whatever genuinely isn't recorded anywhere, rather than guessing.

Usage: python -m little_shakespeare.eval.leaderboard
"""
import csv
import json
import math
from pathlib import Path
from typing import Optional

from little_shakespeare.run_dir import BENCHMARKS_ROOT, CONFIG_FILENAME, CSV_LOG_FILENAME, METRICS_FILENAME, MODELS_ROOT

MISSING = "—"
COLUMNS = [
    "model_id", "num_merges", "d_model", "num_layers",
    "val_loss", "val_perplexity", "val_bpc", "val_tokens", "val_chars",
    "test_loss", "test_perplexity", "test_bpc", "test_tokens", "test_chars",
    "final_epoch", "early_stopped", "train_time_min", "git_commit", "timestamp", "note",
]


def _split_columns(split_name: str, split: dict) -> dict:
    return {
        f"{split_name}_loss": f"{split['loss']:.4f}",
        f"{split_name}_perplexity": f"{split['perplexity']:.2f}",
        f"{split_name}_bpc": f"{split['bpc']:.4f}",
        f"{split_name}_tokens": split["tokens"],
        f"{split_name}_chars": split["chars"],
    }


def _row_from_metrics_json(metrics_path: Path) -> dict:
    """Authoritative source: written by pipeline.py, val/test re-evaluated on
    the actual best checkpoint (not just the last logged epoch)."""
    metrics = json.loads(metrics_path.read_text())
    row = {}
    row.update(_split_columns("val", metrics["val"]))
    row.update(_split_columns("test", metrics["test"]))
    row["final_epoch"] = metrics["final_epoch"]
    row["early_stopped"] = metrics["early_stopped"]
    row["train_time_min"] = f"{metrics['total_train_time_seconds'] / 60:.1f}"
    row["git_commit"] = metrics.get("git_commit") or MISSING
    row["timestamp"] = metrics.get("timestamp") or MISSING
    row["note"] = metrics.get("note") or MISSING
    return row


def _row_from_csv(csv_path: Path) -> dict:
    """Fallback for runs predating metrics.json. Uses the MINIMUM-val_loss
    row, not the last one — early stopping's patience window means the last
    logged epoch usually isn't the epoch that was actually checkpointed.
    Only val_loss/val_perplexity/val_bpc/final_epoch are recoverable this way
    (final_epoch is literally the row's own epoch number, not an estimate);
    everything metrics.json-only (test split, timing, git commit) stays "—" —
    the CSV never recorded it, there's nothing to read."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    best = min(rows, key=lambda r: float(r["val_loss"]))
    val_loss = float(best["val_loss"])
    val_perplexity = float(best["val_perplexity"]) if "val_perplexity" in best else math.exp(val_loss)
    val_bpc = f"{float(best['val_bpc']):.4f}" if "val_bpc" in best else MISSING

    return {
        "val_loss": f"{val_loss:.4f}",
        "val_perplexity": f"{val_perplexity:.2f}",
        "val_bpc": val_bpc,
        "final_epoch": best["epoch"],
    }


def _row_for_run(model_dir: Path) -> dict:
    row = {col: MISSING for col in COLUMNS}
    row["model_id"] = model_dir.name

    config_path = model_dir / CONFIG_FILENAME
    if config_path.exists():
        config = json.loads(config_path.read_text())
        row["num_merges"] = config["preprocessing_config"]["num_merges"]
        row["d_model"] = config["model_config"]["d_model"]
        row["num_layers"] = config["model_config"]["num_layers"]

    metrics_path = model_dir / METRICS_FILENAME
    csv_path = model_dir / CSV_LOG_FILENAME
    if metrics_path.exists():
        row.update(_row_from_metrics_json(metrics_path))
    elif csv_path.exists():
        row.update(_row_from_csv(csv_path))

    return row


def _sort_key(row: dict):
    if row["val_bpc"] == MISSING:
        return (1, 0.0)
    return (0, float(row["val_bpc"]))


def build_leaderboard(models_root: Optional[Path] = None) -> list:
    models_root = models_root or MODELS_ROOT
    if not models_root.exists():
        return []

    run_dirs = sorted(
        (p for p in models_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    rows = [_row_for_run(d) for d in run_dirs]
    rows.sort(key=_sort_key)
    return rows


def write_markdown(rows: list, path: Path) -> None:
    lines = [
        "# Model Leaderboard",
        "",
        "Sorted by validation bits-per-character (bpc) — the one metric that stays "
        f"comparable across different `num_merges` tokenizer configs. `{MISSING}` marks "
        "data that was never recorded for that run (pre-metrics.json runs in particular); "
        "nothing here is retroactively estimated.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in COLUMNS) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    rows = build_leaderboard()
    BENCHMARKS_ROOT.mkdir(exist_ok=True)
    write_markdown(rows, BENCHMARKS_ROOT / "leaderboard.md")
    write_csv(rows, BENCHMARKS_ROOT / "leaderboard.csv")
    print(f"Wrote {BENCHMARKS_ROOT / 'leaderboard.md'} and leaderboard.csv ({len(rows)} runs)")


if __name__ == "__main__":
    main()
