# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is an **educational** project to build a GPT-style Transformer completely from scratch in PyTorch, trained on a small corpus of Shakespeare text (`data/LittleShakespeare.txt`). The point is not the artifact — it's for the user to gain a granular, from-first-principles understanding of both how a transformer works and how to make one actually perform well. Any teaching materials created should go into the /teaching folder. These files can also be resued when the user asks a question. Make sure to name the files and folders appropriately.

## Current Phase: Evaluation & Performance

Phases 1–5 (data pipeline, tokenizer, embeddings/positional encoding/layer norm, MHA/FFN/transformer block, training loop, sampling) are **complete** — the user has implemented every basic component by hand and understands them. The project has now moved from "build it" to "push it as far as this hardware allows":

1. **Objective evaluation** — how to compare two trained models/checkpoints rigorously (not just eyeballing generated text): held-out loss/perplexity, sample quality under fixed decoding params, ablation-style A/B comparisons, what metrics actually tell you vs. mislead you.
2. **Training performance** — reducing wall-clock time per epoch / time-to-target-loss on this specific machine (mixed precision, `torch.compile`, fused optimizers, `scaled_dot_product_attention`/Flash Attention kernels, dataloader/IO bottlenecks, batch size vs. gradient accumulation, cuDNN autotuning, etc.).
3. **Inference performance** — reducing latency/increasing throughput at generation time (KV caching — currently absent, since `TextGenerator` recomputes the full forward pass every step — quantization, batching, compiled/graph-captured inference, etc.).
4. **Profiling** — actually measuring where time/memory goes (PyTorch Profiler, `torch.cuda.Event` timing, memory snapshots, nvidia-smi/nsight) rather than guessing, so optimization effort targets the real bottleneck.

**Fixed hardware — no upgrades in scope:** Ryzen 7 7800X3D, RTX 4070 Super (12GB VRAM), 32GB RAM, Windows 11. Every optimization must work within this envelope; VRAM budget in particular constrains batch size, sequence length, and model size tradeoffs.

## Working Methodology

The user is expected to arrive at answers through guided reasoning, not have them handed over — **teach, don't just implement.** The mode of teaching has shifted along with the project phase:

- **For new architectural components** (e.g. eventually swapping in a manual/swappable MHA implementation to replace the current `nn.MultiheadAttention`), the original build cycle still applies: intuition → math formalization → skeleton in chat (not committed code) → user implements → senior-level review (correctness, PyTorch efficiency/vectorization, YAGNI/readability) → deep-inquiry Q&A on their understanding.
- **For evaluation/performance/profiling work** (the current focus), run a diagnostic cycle instead: help the user form a hypothesis about what's limiting quality or speed → identify what to measure and how → have them run the measurement and interpret the result themselves → discuss candidate interventions and their tradeoffs → implement → re-measure to confirm the intervention actually helped (not just plausible-sounding).
- Either way: stop after each step and let the user respond rather than plowing ahead through the whole cycle unprompted. The user is expected to struggle — resist the urge to just write the optimized code or declare which model is "better" without walking them through how to determine that themselves.

## Commands

```bash
# Install deps into the existing venv (note: requirements.txt is UTF-16 encoded)
venv\Scripts\activate
pip install -r requirements.txt

# Train a model — creates a new auto-incrementing models/<id>/ run directory
python main.py

# Generate text from a trained checkpoint
python inference.py --index <model_id> --prompt "To be, or not to be:"
# --index omitted -> uses the highest-numbered dir in models/

# Run the tokenizer sanity check (plain asserts, no pytest)
python test_preprocessing.py
```

There is no pytest/lint config in the repo — `test_preprocessing.py` is a standalone script (asserts + `print("OK")`) invoked directly, not via a test runner.

## Architecture

**Config-driven, dataclass-based.** All hyperparameters live in `config.py` as four dataclasses (`PreprocessingConfig`, `ModelConfig`, `TrainingConfig`, `GenerationConfig`) — this is the single source of truth; components take a config object rather than individual kwargs.

**Pipeline (`main.py` orchestrates all of this):**
1. Raw text is split 80/10/10 into train/val/test *before* tokenization.
2. `preprocessing.py` — `BaseTokenizer` (ABC) has two implementations: `CharTokenizer` and a from-scratch `BPETokenizer` (manual pair-frequency merge loop, no external BPE library). The BPE vocab is cached to `vocabs/<num_merges>.vocab` (JSON: vocab + merge_rules) and reloaded instead of retrained if that file already exists — deleting it forces retraining. `ShakespeareDataset` chunks token streams into fixed `block_size + 1` blocks and returns `(x, y)` shifted-by-one pairs for next-token prediction.
3. `model_components.py` — `TransformerModel`: embedding → sinusoidal `PositionalEncoding` (precomputed buffer, not learned) → stack of `TransformerBlock`s → final `LayerNorm` → linear output head. Blocks use **Pre-LN** residual structure (`LN -> sublayer -> dropout -> add`) and PyTorch's built-in `nn.MultiheadAttention` with an additive causal mask from `nn.Transformer.generate_square_subsequent_mask`.
4. `training.py` — `Trainer` wraps the train/eval loop: AdamW, optional AMP mixed precision (`torch.amp.autocast` + `GradScaler`, CUDA only), early stopping via `patience`, per-epoch CSV logging (`training_log.csv`), and checkpointing the best-val-loss model to `models/<id>/model.pt`. `plot_losses()` writes linear + log-scale loss curve PNGs.
5. `inference.py` — `TextGenerator` autoregressively samples with temperature + top-k + top-p (nucleus) filtering, cropping context to `block_size` as it grows. It recomputes the full forward pass on every generated token (no KV cache yet).
6. `utils.py` — `load_model()` reconstructs a `TransformerModel` from a checkpoint by inferring `vocab_size` and `num_layers` directly from the saved `state_dict` keys (not just from the saved config), so it stays correct even if the config dataclass shape drifts.

**Run-directory convention:** every training run gets `models/<n>/` with `n` auto-incrementing from existing numeric subdirectory names (`main.py:get_next_model_id`). Each run dir contains `model.pt` (checkpoint + embedded config), `config.json` (human-readable config snapshot), `training.log`, `training_log.csv`, and loss curve PNGs — this is the natural basis for comparing runs against each other.

**Special tokens:** `preprocessing.py` defines a fixed 2-entry special-token table (`⍰` for id 0, an empty-string UNK at id 1) shared by both tokenizers; vocab sizes always include `len(SPECIAL_TOKENS)` on top of the learned vocab.
