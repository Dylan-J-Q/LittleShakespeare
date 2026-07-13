import torch
from torch.utils.data import Dataset
from typing import List
from little_shakespeare.config import PreprocessingConfig
from little_shakespeare.data.tokenizer import BaseTokenizer

class ShakespeareDataset(Dataset):
    """
    Improved Dataset using 'Block' logic to ensure the model
    sees sequences of data rather than isolated characters.
    """
    def __init__(self, raw_text: str, tokenizer: BaseTokenizer, config: PreprocessingConfig, logger=None):
        self.logger = logger
        self.tokenizer = tokenizer
        self.block_size = config.block_size
        self.data = self._preprocess(raw_text)
        if self.logger:
            self.logger.info(f"Dataset initialized with {len(self.data)} blocks.")

    def _preprocess(self, text: str) -> List[torch.Tensor]:
        full_sequence = self.tokenizer.encode(text)
        blocks = []
        for i in range(0, len(full_sequence) - self.block_size, self.block_size):
            chunk = full_sequence[i : i + self.block_size + 1]
            if len(chunk) == self.block_size + 1:
                blocks.append(torch.tensor(chunk, dtype=torch.long))

        return blocks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
