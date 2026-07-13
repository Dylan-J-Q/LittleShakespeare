import torch
import torch.nn.functional as F
from typing import Optional

from little_shakespeare.config import GenerationConfig
from little_shakespeare.model.transformer import TransformerModel


class TextGenerator:
    def __init__(self, model: TransformerModel, tokenizer, device: str = "cpu", block_size: Optional[int] = None):
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

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
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
