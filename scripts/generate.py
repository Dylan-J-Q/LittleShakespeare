"""Entry point: generate text from a trained checkpoint.

    python scripts/generate.py --index 0 --prompt "To be, or not to be:"

--index omitted -> uses the highest-numbered dir in models/.
"""
import argparse
import json
from pathlib import Path

from little_shakespeare.config import GenerationConfig, PreprocessingConfig
from little_shakespeare.checkpoint import load_model
from little_shakespeare.data.tokenizer import BPETokenizer
from little_shakespeare.inference.generator import TextGenerator
from little_shakespeare.run_dir import resolve_model_id, checkpoint_path, config_path, VOCABS_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run inference on the model.")
    parser.add_argument("--index", type=int, help="Model index to use.")
    parser.add_argument("--prompt", type=str, help="Prompt for generation.")
    args = parser.parse_args()

    gen_config = GenerationConfig()
    preprocessing_config = PreprocessingConfig()
    print(f"Using device: {gen_config.device}")

    model_index = resolve_model_id(args.index)

    if model_index is not None:
        ckpt_path = checkpoint_path(model_index)
    else:
        ckpt_path = Path("best_model.pt")

    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        return

    model, model_config, training_config = load_model(checkpoint_path=str(ckpt_path), device=gen_config.device)

    if model_index is not None:
        with open(config_path(model_index), "r") as f:
            config_data = json.load(f)
        num_merges = config_data["preprocessing_config"]["num_merges"]
        vocab_path = VOCABS_ROOT / f"{num_merges}.vocab"
    else:
        vocab_path = Path("tokenizer.vocab")

    tokenizer = BPETokenizer("", preprocessing_config)
    tokenizer.load_vocab(str(vocab_path))
    generator = TextGenerator(model, tokenizer, device=gen_config.device, block_size=preprocessing_config.block_size)

    prompt = args.prompt if args.prompt is not None else "To be, or not to be:"
    print(f"Prompt: {prompt}")
    print("-" * 30)

    result = generator.generate(prompt, config=gen_config)
    print(result)


if __name__ == "__main__":
    main()
