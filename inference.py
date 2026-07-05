import torch
import torch.nn.functional as F
import os
import argparse
import json
from model_components import TransformerModel
from preprocessing import BPETokenizer
from utils import load_model
from config import TrainingConfig, GenerationConfig, PreprocessingConfig

torch.serialization.add_safe_globals([TrainingConfig, GenerationConfig])

class TextGenerator:
    def __init__(self, model: TransformerModel, tokenizer, device: str = "cpu", block_size: int = None):
        """
        Args:
            model: The trained Transformer model.
            tokenizer: The tokenizer used during preprocessing.
            device: The device to run the model on ('cpu' or 'cuda').
            block_size: Max context length the model was trained on; older tokens are
                cropped from the context so generation never runs on unseen positions.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.model.to(self.device)
        self.model.eval()  # disable dropout for generation

    def generate(self, prompt: str, config: GenerationConfig = None) -> str:
        """
        Generates text from a given prompt using various sampling strategies.
        """
        if config is None:
            config = GenerationConfig()

        # Convert prompt to token IDs
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)

        generated_ids = input_ids
        num_new_tokens = max(config.max_length - generated_ids.shape[1], 0)

        for _ in range(num_new_tokens):
            context = generated_ids
            if self.block_size is not None and context.shape[1] > self.block_size:
                context = context[:, -self.block_size:]

            # Get logits from the model
            with torch.no_grad():
                outputs = self.model(context)
                # We only need the logits for the last token
                logits = outputs[:, -1, :] # Shape: (batch_size, vocab_size)

            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature
            
            # Choose sampling strategy
            if config.top_k > 0:
                logits = self._top_k_filter(logits, config.top_k)
            if config.top_p > 0:
                logits = self._top_p_filter(logits, config.top_p)

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return self.tokenizer.decode(generated_ids[0].tolist())

    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        # Keep only the top k logits, set the rest to -infinity
        k = min(k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability greater than p
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the mask to keep the first token that exceeds p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter the mask back to the original order
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Run inference on the model.")
    parser.add_argument("--index", type=int, help="Model index to use.")
    parser.add_argument("--prompt", type=str, help="Prompt for generation.")
    args = parser.parse_args()

    gen_config = GenerationConfig()
    preprocessing_config = PreprocessingConfig()
    print(f"Using device: {gen_config.device}")

    model_index = args.index
    if model_index is None:
        indices = []
        if os.path.exists("models"):
            for entry in os.listdir("models"):
                if entry.isdigit():
                    indices.append(int(entry))
        
        if indices:
            model_index = max(indices)
        else:
            model_index = None

    if model_index is not None:
        checkpoint_path = os.path.join("models", str(model_index), "model.pt")
    else:
        checkpoint_path = "best_model.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
    else:
        model, model_config, training_config = load_model(checkpoint_path=checkpoint_path, device=gen_config.device)
        
        if model_index is not None:
            config_path = os.path.join("models", str(model_index), "config.json")
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            num_merges = config_data["preprocessing_config"]["num_merges"]
            vocab_path = os.path.join("vocabs", f"{num_merges}.vocab")
        else:
            vocab_path = "tokenizer.vocab"
        
        tokenizer = BPETokenizer("", preprocessing_config)
        tokenizer.load_vocab(vocab_path)
        generator = TextGenerator(model, tokenizer, device=gen_config.device, block_size=preprocessing_config.block_size)
        
        prompt = args.prompt if args.prompt is not None else "To be, or not to be:"
        print(f"Prompt: {prompt}")
        print("-" * 30)
        
        result = generator.generate(prompt, config=gen_config)
        print(result)
