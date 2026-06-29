import torch 
from torch.utils.data import Dataset, DataLoader
from typing import List
from abc import ABC, abstractmethod

UNK_TOKEN = "�"
SPECIAL_TOKENS = {
    0: "⍰",
    1: UNK_TOKEN,
}
UNK_INDEX = [key for key, value in SPECIAL_TOKENS.items() if value == UNK_TOKEN][0]

DEFAULT_BLOCK_SIZE = 512


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers. 
    """
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    def get_vocab_size(self) -> int:
        return len(self.vocab)
    


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.
    """
    def __init__(self, text: str):
        super().__init__()
        self.vocab = {char: idx + len(SPECIAL_TOKENS) for idx, char in enumerate(sorted(set(text)))}
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        return [self.vocab[char] if char in self.vocab else UNK_INDEX for char in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.inverse_vocab[token] if token in self.inverse_vocab else SPECIAL_TOKENS[UNK_INDEX] for token in tokens])



class ShakespeareDataset(Dataset):
    """
    Improved Dataset using 'Block' logic to ensure the model 
    sees sequences of data rather than isolated characters.
    """
    def __init__(self, raw_text: str, tokenizer: BaseTokenizer, block_size: int = DEFAULT_BLOCK_SIZE):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self._preprocess(raw_text)

    def _preprocess(self, text: str) -> List[List[int]]:
        full_sequence = self.tokenizer.encode(text)
        blocks = []
        for i in range(0, len(full_sequence) - self.block_size, self.block_size):
            chunk = full_sequence[i : i + self.block_size]
            if len(chunk) == self.block_size: 
                blocks.append(chunk)
        
        return blocks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.long)
