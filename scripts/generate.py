"""Entry point: generate text from a trained checkpoint.

    python scripts/generate.py --index 0 --prompt "To be, or not to be:"

--index omitted -> uses the highest-numbered dir in models/.
"""
import argparse
import json

from little_shakespeare.config import GenerationConfig, PreprocessingConfig
from little_shakespeare.checkpoint import load_model
from little_shakespeare.data.tokenizer import BPETokenizer
from little_shakespeare.inference.generator import TextGenerator
from little_shakespeare.run_dir import resolve_model_id, checkpoint_path, config_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on the model.")
    parser.add_argument("--index", type=int, help="Model index to use.")
    parser.add_argument("--prompt", type=str, help="Prompt for generation.")
    parser.add_argument("--temperature", type=float, help="Override GenerationConfig.temperature.")
    parser.add_argument("--top_k", type=int, help="Override GenerationConfig.top_k.")
    parser.add_argument("--top_p", type=float, help="Override GenerationConfig.top_p.")
    parser.add_argument("--max_length", type=int, help="Override GenerationConfig.max_length.")
    args = parser.parse_args()

    gen_config = GenerationConfig()
    if args.temperature is not None:
        gen_config.temperature = args.temperature
    if args.top_k is not None:
        gen_config.top_k = args.top_k
    if args.top_p is not None:
        gen_config.top_p = args.top_p
    if args.max_length is not None:
        gen_config.max_length = args.max_length
    print(f"Using device: {gen_config.device}")

    model_index = resolve_model_id(args.index)
    if model_index is None:
        print("No trained models found in models/. Run `python scripts/train.py` first.")
        return

    ckpt_path = checkpoint_path(model_index)
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        return

    model, model_config, training_config = load_model(checkpoint_path=str(ckpt_path), device=gen_config.device)

    with open(config_path(model_index), "r") as f:
        config_data = json.load(f)
    # This run's OWN saved preprocessing config, not today's config.py
    # defaults — a run trained with a non-default data_path/num_merges/
    # block_size must be evaluated with that same setup, not a guess.
    preprocessing_config = PreprocessingConfig(**config_data["preprocessing_config"])

    tokenizer = BPETokenizer.from_vocab_file(preprocessing_config)
    generator = TextGenerator(model, tokenizer, device=gen_config.device, block_size=preprocessing_config.block_size)

    prompt = args.prompt if args.prompt is not None else "To be, or not to be:"
    print(f"Prompt: {prompt}")
    print("-" * 30)

    result = generator.generate(prompt, config=gen_config)
    print(result)


if __name__ == "__main__":
    main()
