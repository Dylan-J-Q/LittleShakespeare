"""Sample-quality A/B comparison across checkpoints.

Deliberately never computes a "winner" — that would hand over the judgment
CLAUDE.md's methodology reserves for the user. This produces controlled raw
material instead: fixed prompts (eval/prompts.txt), a fixed seed re-applied
per prompt so every compared checkpoint sees the identical random sampling
trajectory, and context metrics (perplexity/bpc/diversity/memorization/
significance) shown alongside — never collapsed into a verdict.

Usage:
    python -m little_shakespeare.eval.compare --ids 1,2,4
    python -m little_shakespeare.eval.compare --ids 1,4 --seed 0 --temperature 0.8
"""
import argparse
import itertools
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from little_shakespeare.checkpoint import load_model
from little_shakespeare.config import GenerationConfig, PreprocessingConfig
from little_shakespeare.data.dataset import ShakespeareDataset
from little_shakespeare.data.splits import split_text
from little_shakespeare.data.tokenizer import BPETokenizer
from little_shakespeare.eval.diversity import diversity_report
from little_shakespeare.eval.memorization import check_memorization
from little_shakespeare.eval.perplexity import aggregate_nll_stats, accumulate_nll_per_example, metrics_report
from little_shakespeare.eval.significance import paired_bootstrap_difference_ci
from little_shakespeare.inference.generator import TextGenerator
from little_shakespeare.run_dir import BENCHMARKS_ROOT, checkpoint_path, config_path

PROMPTS_PATH = Path(__file__).parent / "prompts.txt"
COMPARE_OUTPUT_ROOT = BENCHMARKS_ROOT / "compare"


def load_prompts() -> List[str]:
    lines = PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


class Run:
    """Everything needed to generate from, and evaluate, one checkpoint."""

    def __init__(self, model_id: int):
        with open(config_path(model_id)) as f:
            saved_config = json.load(f)
        self.preprocessing_config = PreprocessingConfig(**saved_config["preprocessing_config"])

        with open(self.preprocessing_config.data_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        self.train_text, self.val_text, _ = split_text(raw_text, self.preprocessing_config)

        self.tokenizer = BPETokenizer(raw_text, self.preprocessing_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, _ = load_model(
            str(checkpoint_path(model_id)), device=self.device, vocab_size=self.tokenizer.get_vocab_size()
        )
        self.generator = TextGenerator(
            self.model, self.tokenizer, device=self.device, block_size=self.preprocessing_config.block_size
        )

        val_dataset = ShakespeareDataset(self.val_text, self.tokenizer, self.preprocessing_config)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.per_example_stats = accumulate_nll_per_example(self.model, val_loader, self.tokenizer, self.device)
        self.context = metrics_report(aggregate_nll_stats(self.per_example_stats))


def load_runs(ids: List[int]) -> Dict[int, Run]:
    runs = {}
    for model_id in ids:
        try:
            runs[model_id] = Run(model_id)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Skipping model {model_id}: could not load ({e})")
    return runs


def pairwise_significance(runs: Dict[int, Run], n_resamples: int, seed: int) -> List[dict]:
    results = []
    for id_a, id_b in itertools.combinations(sorted(runs), 2):
        run_a, run_b = runs[id_a], runs[id_b]
        if len(run_a.per_example_stats) != len(run_b.per_example_stats):
            continue  # different val split sizes (e.g. different train/val/test fractions) — not comparable
        diff, lower, upper = paired_bootstrap_difference_ci(
            run_a.per_example_stats, run_b.per_example_stats, n_resamples=n_resamples, seed=seed
        )
        results.append({
            "pair": f"{id_a} vs {id_b}", "diff": diff, "lower": lower, "upper": upper,
            "distinguishable": lower > 0 or upper < 0,
        })
    return results


def run_comparison(ids: List[int], gen_config: GenerationConfig, seed: int, n_resamples: int) -> str:
    prompts = load_prompts()
    runs = load_runs(ids)
    if not runs:
        return "No checkpoints could be loaded."

    lines = [f"# Comparison: models {', '.join(str(i) for i in runs)}", ""]
    lines.append(
        f"seed={seed} | temperature={gen_config.temperature} | top_k={gen_config.top_k} | "
        f"top_p={gen_config.top_p} | max_length={gen_config.max_length}"
    )
    lines.append("")

    lines.append("## Context metrics (val set)")
    lines.append("")
    lines.append("| model | val_loss | val_perplexity | val_bpc |")
    lines.append("|---|---|---|---|")
    for model_id, run in runs.items():
        c = run.context
        lines.append(f"| {model_id} | {c['loss']:.4f} | {c['perplexity']:.2f} | {c['bpc']:.4f} |")
    lines.append("")

    sig_results = pairwise_significance(runs, n_resamples, seed)
    if sig_results:
        lines.append(f"## Pairwise bpc significance (paired bootstrap, {n_resamples} resamples, 95% CI)")
        lines.append("")
        lines.append("| pair | Δ bpc (A − B) | 95% CI | distinguishable from noise? |")
        lines.append("|---|---|---|---|")
        for r in sig_results:
            verdict = "yes" if r["distinguishable"] else "no"
            lines.append(f"| {r['pair']} | {r['diff']:+.5f} | [{r['lower']:+.5f}, {r['upper']:+.5f}] | {verdict} |")
        lines.append("")
        lines.append(
            "_A gap that isn't distinguishable from noise doesn't mean the models are identical — "
            "it means this val set is too small to tell them apart on bpc alone. Not a verdict either way._"
        )
        lines.append("")

    for prompt_idx, prompt in enumerate(prompts):
        lines.append(f"## Prompt: \"{prompt}\"")
        lines.append("")
        for model_id, run in runs.items():
            torch.manual_seed(seed + prompt_idx)  # identical random trajectory across models, per prompt
            generated = run.generator.generate(prompt, config=gen_config)
            diversity = diversity_report(generated)
            memorization = check_memorization(generated, run.train_text)

            lines.append(f"### model {model_id}")
            lines.append("")
            lines.append(generated)
            lines.append("")
            lines.append(
                f"- distinct-1: {diversity['distinct_1']:.3f} | distinct-2: {diversity['distinct_2']:.3f}"
            )
            lines.append(
                f"- memorization: overlap={memorization.overlap_fraction:.3f} | "
                f"longest verbatim match: {memorization.longest_match_words} words"
                + (f" (\"{memorization.longest_match_text}\")" if memorization.longest_match_text else "")
            )
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sample-quality A/B comparison across checkpoints.")
    parser.add_argument("--ids", type=str, required=True, help="Comma-separated model ids, e.g. 1,2,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_resamples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    args = parser.parse_args()

    ids = [int(i) for i in args.ids.split(",")]
    gen_config = GenerationConfig()
    overrides = {k: v for k, v in vars(args).items()
                 if k in ("temperature", "top_k", "top_p", "max_length") and v is not None}
    gen_config = replace(gen_config, **overrides)

    report = run_comparison(ids, gen_config, seed=args.seed, n_resamples=args.n_resamples)

    COMPARE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = COMPARE_OUTPUT_ROOT / f"{timestamp}_ids-{'-'.join(str(i) for i in ids)}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
