import torch
import json
import os
from torch.utils.data import Dataset
from typing import List
from abc import ABC, abstractmethod
from collections import Counter
from config import PreprocessingConfig

UNK_TOKEN = ""
SPECIAL_TOKENS = {
    0: "⍰",
    1: UNK_TOKEN,
}
UNK_INDEX = [key for key, value in SPECIAL_TOKENS.items() if value == UNK_TOKEN][0]

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
        # Ensure the vocabulary size includes all special tokens to avoid index out of range errors
        return len(self.vocab) + len(SPECIAL_TOKENS)
    



class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.
    """
    def __init__(self, text: str, config: PreprocessingConfig):
        super().__init__()
        self.vocab = {char: idx + len(SPECIAL_TOKENS) for idx, char in enumerate(sorted(set(text)))}
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        return [self.vocab[char] if char in self.vocab else UNK_INDEX for char in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.inverse_vocab[token] if token in self.inverse_vocab else SPECIAL_TOKENS[UNK_INDEX] for token in tokens])
    
 
class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding tokenizer.
    """
    def __init__(self, text: str, config: PreprocessingConfig, logger=None):
        super().__init__()
        self.logger = logger
        self.merge_rules = []
        vocab_path = os.path.join("vocabs", f"{config.num_merges}.vocab")
        
        if os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            # Initialize with characters
            tokens = list(text)
            vocab = {char: i + len(SPECIAL_TOKENS) for i, char in enumerate(sorted(set(text)))}
            self.vocab = vocab
            self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}
            
            # Simple BPE training
            if self.logger:
                self.logger.info(f"Starting BPE training with {config.num_merges} merges.")
            for _ in range(config.num_merges):
                pairs = Counter()
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i+1])] += 1
                
                if not pairs:
                    break
                
                # Find most frequent pair
                best_pair = max(pairs, key=pairs.get)
                if pairs[best_pair] < 2:
                    break
                
                # Merge best_pair
                new_token = f"{best_pair[0]}{best_pair[1]}"
                new_idx = len(self.vocab) + len(SPECIAL_TOKENS)
                self.vocab[new_token] = new_idx
                self.inverse_vocab[new_idx] = new_token
                self.merge_rules.append(best_pair)
                
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(new_token)  # keep the token stream as strings, like encode()
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            # Save the vocab after training
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            self.save_vocab(vocab_path)
            if self.logger:
                self.logger.info("BPE training complete and vocab saved.")

    def encode(self, text: str) -> List[int]:
        tokens = list(text)
        for str1, str2 in self.merge_rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == str1 and tokens[i+1] == str2:
                    new_tokens.append(str1 + str2)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return [self.vocab.get(t, UNK_INDEX) for t in tokens]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.inverse_vocab.get(token, SPECIAL_TOKENS[UNK_INDEX]) for token in tokens])

    def save_vocab(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "vocab": self.vocab,
                "merge_rules": self.merge_rules
            }, f, ensure_ascii=False)

    def load_vocab(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merge_rules = data["merge_rules"]
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}

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