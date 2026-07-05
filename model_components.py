import math
import torch
import torch.nn as nn
from config import ModelConfig

class PositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim

        # Create a matrix of shape (max_pos_encoding_len, embedding_dim) for positional encodings
        pe = torch.zeros(config.max_pos_encoding_len, config.embedding_dim)
        position = torch.arange(0, config.max_pos_encoding_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embedding_dim, 2).float() * (-math.log(10000.0) / config.embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embedding_dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ff_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ff_hidden_dim, config.d_model)
        )
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x, mask=None):
        identity = x
        x = self.ln1(x)  # Residual connection
        x, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
        x = self.dropout(x)
        x = x + identity

        identity = x
        x = self.ln2(x)  # Residual connection
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + identity

        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.pos_encoding = PositionalEncoding(config)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) 
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)  # final norm for Pre-LN residual stream
        self.output_layer = nn.Linear(config.d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        seq_len = x.size(1)
        if mask is None:
            # Additive causal mask (0 / -inf), the form nn.MultiheadAttention's attn_mask expects.
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device, dtype=x.dtype)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        return self.output_layer(x)
