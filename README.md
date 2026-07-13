# LittleShakespeare - Pushed to its limits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.12.10](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31213/)

## Demo
Prompt: To be, or not to be:

holy mon.

SISABELLA:
So with were what Ifear.

ANGELO:
Ha good ct me finness that the wdyself.

ANGELO:
Whad to mustingman?

ISABELO:
SABELLA:
I hat.

ISABELLA:
So strum, my lords it dection.

ISABELLA:
Hay, sir.

ISABELLA:
Yought son
Thout ailkillain, you to your grave you your prant.

ISABELLA:
O, my lord.

ISABELLA:
I'll not your your good my lor,
ABELLA:
Ifear meancan it is your gray you your shand
That t.

ISABELLA:
ABELLA:
How?

ISABELLA:
O, Iness your your your it, thenerttainst good osonours to the father,
Witherrongumost be how,
Aladyryety,
To have you make it aba ed, as your hand
What you ves de't.

ISABELLA:
SABELLA:
O, that, by your your sa scuch so you dis'ty lord.

ISABELLA:
How it. Hay, sir; and band
ABELLA:
O, sir, th hall you leser blook liveven spper we thinke
Hows of the t's de donden swhat you leseech you
To have may father sper sheak you,
Nor it. Thanceit ink your your have sa mintter give your your sper
As ince to bbtrich your for your y,
Whent what I ds dench ark for your your your desscrany litlong as you ve.

LABELO:
ABELO:
GELLA:
How cantain,
As you hatold a genten, ter for for you.

ANGELO:
SABELO:
GELO:
Yous I much are a day.

ISABELLA:
Oratess, I'ter prove! it is your so lest
By you are you have madodisdy
BELO:
As could you not you speavengaintend
That you your fore you comen you,
Have you sotopartenty dogoo ttrand, I have sotwor your prave.

ISABELO:
I have no more to you me, sir, sir.

ANGELO:
ISABELLA:
We you your your hat.

ISABELLA:
ABELO:
We pitippivoy.

ISABELLA:
ABELO:
Who you your gention.

ANGELO:
We your your by you se; I have you your your gray.

ANGELLA:
O, sir, werviva wor to--morter you your your it will I
Wither your your your your your your gent.

ISABELO:
I know you
Whater to good you your it not be preving.

ISABELLA:
I'll you your headenator your sudence?

ISABELLA:
Ory you your dices; and you have not have ave, you.

ANGELO:
Thance to your sewere come, madeep

## Why This Project
This project is an attempt to optimise a small transformer trained on Shakespeare's works within strict hardware requirements. The hardware constraints are that the whole model must be trained and run inference on a Ryzen 7800x3d CPU, NVIDIA 4070 super GPU (12GB of VRAM) and 32GB of RAM. The goals of this project are: Increase generation accuracy; improve generation speed; and to learn how transformers and the systems around them work in detail. Implemented mostly from first principles using components from pytorch.

## Architecture
The architecture of the transformer:

```mermaid
flowchart TD
    In["Token IDs (batch, seq_len)"] --> Emb[Embedding Layer]
    Emb --> Pos["+ Sinusoidal Positional Encoding"]
    Pos --> Blocks["Transformer Block × 6"]
    Blocks --> LN[Final LayerNorm]
    LN --> Out[Output Linear Layer]
    Out --> Logits["Logits (batch, seq_len, vocab_size)"]

```

The transformer blocks:

```mermaid
flowchart TD
    X[Block Input] --> LN1[LayerNorm 1]
    LN1 --> MHA["Multi-Head Attention (causal mask)"]
    MHA --> Add1((+))
    X --> Add1
    Add1 --> LN2[LayerNorm 2]
    LN2 --> FFN["Feed Forward: Linear → GELU → Linear"]
    FFN --> Add2((+))
    Add1 --> Add2
    Add2 --> Y[Block Output]

```

## Repo Structure

```
LittleShakespeare/
├── data/
│   └── LittleShakespeare.txt        # raw corpus; split 80/10/10 into train/val/test at runtime
├── models/
│   └── <n>/                         # one directory per training run, auto-incrementing
│       ├── model.pt                 # checkpoint (weights + embedded config)
│       ├── config.json              # hyperparameter snapshot for this run
│       ├── training.log             # human-readable training log
│       ├── training_log.csv         # epoch, train_loss, val_loss
│       └── loss_curves_*.png        # linear + log-scale loss curves
├── vocabs/
│   └── <num_merges>.vocab           # cached BPE vocab + merge rules
├── scripts/
│   ├── train.py                     # training entry point (thin CLI)
│   └── generate.py                  # text-generation entry point (thin CLI)
├── src/little_shakespeare/          # the importable package
│   ├── config.py                    # every hyperparameter, as 4 dataclasses
│   ├── run_dir.py                   # run-directory conventions + model-id resolution
│   ├── checkpoint.py                # checkpoint save/load
│   ├── data/
│   │   ├── tokenizer.py             # tokenizers (char-level + from-scratch BPE)
│   │   └── dataset.py               # block chunking into (x, y) next-token pairs
│   ├── model/
│   │   └── transformer.py           # the transformer architecture
│   ├── training/
│   │   ├── trainer.py               # training/eval loop, early stopping
│   │   ├── reporting.py             # CSV logging + loss-curve plotting
│   │   └── pipeline.py              # end-to-end run_training() wiring
│   ├── inference/
│   │   └── generator.py             # autoregressive text generation
│   ├── eval/                        # (greenfield) evaluation harness — see Roadmap
│   └── benchmark/                   # (greenfield) profiling & benchmarking — see Roadmap
├── tests/
│   └── data/
│       └── test_tokenizer.py        # tokenizer sanity checks (pytest)
└── pyproject.toml                   # packaging + dependencies
```

The package installs editable (`pip install -e .`), so `little_shakespeare.*` imports resolve regardless of working directory. CI is planned — see Roadmap.

## Installation
This project installs as an editable package. Install the CUDA build of PyTorch **first** (the pinned `+cuXXX` wheels aren't on PyPI's default index), then do the editable install:

```bash
# 1. CUDA build of PyTorch from PyTorch's own index (developed against cu126)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. Editable install — pulls matplotlib and makes the little_shakespeare package importable
pip install -e ".[dev]"
```

For a CPU-only or different-CUDA setup, pick the matching `torch` command from https://pytorch.org/get-started/locally/ in step 1.

## Usage

Train a new model — creates a new auto-incrementing `models/<id>/` run directory:
```bash
python scripts/train.py
```

Generate text from a trained checkpoint:
```bash
python scripts/generate.py --index <model_id> --prompt "To be, or not to be:"
```
Omit `--index` to use the highest-numbered (most recent) run in `models/`.

## Benchmarks

No formal evaluation harness exists yet — building one (held-out perplexity/bits-per-character, sample-quality comparisons under fixed decoding parameters, and training/inference tokens-per-sec) is the current focus of the project. In the meantime, here's where the two existing runs actually landed, read straight from their `training_log.csv`:

| Run | Tokenizer merges | d_model | Layers | Heads | Best Val Loss | Naive Val Perplexity¹ |
|---|---|---|---|---|---|---|
| 0 | 256 | 512 | 6 | 16 | 3.7505 (epoch 27) | ~42.5 |
| 1 | 256 | 512 | 6 | 16 | 3.7640 (epoch 24) | ~43.1 |

¹ `exp(val_loss)` — a naive perplexity, not yet the token-weighted version the evaluation harness will compute. Treat as directional, not final.

## Testing & CI

Run the test suite:
```bash
pytest
```
The suite is currently minimal (a tokenizer sanity check), and there's no CI pipeline yet — more tests and a GitHub Actions workflow (lint, type-check, test on every push) are planned next.

## Roadmap

Phases 1–5 are complete — every core component (data pipeline, tokenizer, embeddings/positional encoding, multi-head attention/feed-forward/transformer block, training loop, autoregressive sampling) was implemented from scratch.

**Current phase: Evaluation & Performance** — pushing this model as far as the hardware above allows:
- [ ] Objective evaluation — perplexity/bits-per-character, sample-quality A/B comparisons, a cross-run leaderboard
- [ ] Training performance — profiling, mixed precision tuning, fused optimizer, attention kernel choice
- [ ] Inference performance — KV caching (currently absent — every generated token recomputes the full forward pass), throughput benchmarking
- [ ] Profiling-driven optimization throughout, instead of guessing

Also planned: a `src/`-based package layout, a real pytest suite, CI, and eventually fine-tuning experiments.

## License
MIT license as seen in [License](/LICENSE)