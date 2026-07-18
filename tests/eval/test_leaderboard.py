"""Tests for eval/leaderboard.py's row-building and sorting logic.

Builds fake models/<id>/ trees under tmp_path — pure filesystem/logic,
no model loading, matching leaderboard.py's own "never load a checkpoint" rule.
"""
import json

from little_shakespeare.eval.leaderboard import MISSING, build_leaderboard


def _write_config(run_dir, num_merges=256, d_model=512, num_layers=6):
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "preprocessing_config": {"num_merges": num_merges},
        "model_config": {"d_model": d_model, "num_layers": num_layers},
    }
    (run_dir / "config.json").write_text(json.dumps(config))


def _write_metrics(run_dir, val_loss=3.7, val_perplexity=40.4, val_bpc=2.7, note=None,
                    test_loss=3.75, test_perplexity=42.5, test_bpc=2.75,
                    final_epoch=23, early_stopped=True, total_train_time_seconds=70.2,
                    git_commit="abc1234"):
    metrics = {
        "note": note,
        "timestamp": "2026-07-18T12:00:00",
        "git_commit": git_commit,
        "final_epoch": final_epoch,
        "early_stopped": early_stopped,
        "best_val_loss": val_loss,
        "total_train_time_seconds": total_train_time_seconds,
        "val": {"loss": val_loss, "perplexity": val_perplexity, "bpc": val_bpc, "tokens": 88784, "chars": 111529},
        "test": {"loss": test_loss, "perplexity": test_perplexity, "bpc": test_bpc, "tokens": 88480, "chars": 111529},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def _write_csv(run_dir, rows, header):
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    (run_dir / "training_log.csv").write_text("\n".join(lines))


def test_run_with_metrics_json_uses_its_values(tmp_path):
    models_root = tmp_path / "models"
    run_dir = models_root / "0"
    _write_config(run_dir)
    _write_metrics(run_dir, val_loss=3.5, val_perplexity=33.1, val_bpc=2.5, note="baseline")

    rows = build_leaderboard(models_root)
    assert len(rows) == 1
    assert rows[0]["val_bpc"] == "2.5000"
    assert rows[0]["val_perplexity"] == "33.10"
    assert rows[0]["note"] == "baseline"


def test_metrics_json_row_includes_test_split_and_run_metadata(tmp_path):
    models_root = tmp_path / "models"
    run_dir = models_root / "0"
    _write_config(run_dir)
    _write_metrics(run_dir, test_loss=3.9, test_perplexity=49.4, test_bpc=2.9,
                    final_epoch=23, early_stopped=True, total_train_time_seconds=120.0,
                    git_commit="deadbee")

    rows = build_leaderboard(models_root)
    row = rows[0]
    assert row["test_loss"] == "3.9000"
    assert row["test_perplexity"] == "49.40"
    assert row["test_bpc"] == "2.9000"
    assert row["final_epoch"] == 23
    assert row["early_stopped"] is True
    assert row["train_time_min"] == "2.0"  # 120s -> 2.0 min
    assert row["git_commit"] == "deadbee"


def test_csv_fallback_recovers_final_epoch_but_not_test_or_timing(tmp_path):
    models_root = tmp_path / "models"
    run_dir = models_root / "0"
    _write_config(run_dir)
    _write_csv(run_dir, [(1, 3.9, 4.0), (2, 3.5, 3.7)], header=["epoch", "train_loss", "val_loss"])

    rows = build_leaderboard(models_root)
    row = rows[0]
    assert row["final_epoch"] == "2"       # min-val_loss row's own epoch number, read verbatim
    assert row["test_bpc"] == MISSING       # never recorded for this era of run
    assert row["train_time_min"] == MISSING
    assert row["git_commit"] == MISSING


def test_old_csv_only_run_derives_perplexity_but_not_bpc(tmp_path):
    models_root = tmp_path / "models"
    run_dir = models_root / "0"
    _write_config(run_dir)
    _write_csv(run_dir, [(1, 3.9, 4.0), (2, 3.5, 3.7)], header=["epoch", "train_loss", "val_loss"])

    rows = build_leaderboard(models_root)
    assert rows[0]["val_loss"] == "3.7000"  # min val_loss row, not the last one
    assert rows[0]["val_bpc"] == MISSING     # never recorded for this era of run


def test_new_csv_without_metrics_json_uses_its_own_bpc_column(tmp_path):
    models_root = tmp_path / "models"
    run_dir = models_root / "0"
    _write_config(run_dir)
    _write_csv(
        run_dir,
        [(1, 4.5, 4.3, 73.7, 3.1), (2, 4.0, 3.9, 49.4, 2.8)],
        header=["epoch", "train_loss", "val_loss", "val_perplexity", "val_bpc"],
    )

    rows = build_leaderboard(models_root)
    assert rows[0]["val_bpc"] == "2.8000"  # best (min val_loss) epoch's own recorded bpc


def test_run_with_neither_file_is_all_missing_but_does_not_crash(tmp_path):
    models_root = tmp_path / "models"
    (models_root / "0").mkdir(parents=True)

    rows = build_leaderboard(models_root)
    assert rows[0]["val_bpc"] == MISSING
    assert rows[0]["num_merges"] == MISSING


def test_sorted_ascending_by_bpc_with_missing_last(tmp_path):
    models_root = tmp_path / "models"
    for model_id, bpc in [("0", 3.0), ("1", 2.5), ("2", None)]:
        run_dir = models_root / model_id
        _write_config(run_dir)
        if bpc is not None:
            _write_metrics(run_dir, val_bpc=bpc)

    rows = build_leaderboard(models_root)
    assert [r["model_id"] for r in rows] == ["1", "0", "2"]


def test_empty_models_root_returns_empty_list(tmp_path):
    assert build_leaderboard(tmp_path / "does_not_exist") == []


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_run_with_metrics_json_uses_its_values(Path(d) / "a")
    print("OK (run via pytest for full coverage)")
