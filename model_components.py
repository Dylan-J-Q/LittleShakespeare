import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Create a matrix of shape (max_len, embedding_dim) for positional encodings
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embedding_dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
    

class LayerNormalisation(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.q_kv_projection = nn.Linear(d_model, d_model * 3) 
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        qkv = self.q_kv_projection(x)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2) 
        v = v.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)

        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))

        attention_probabilities = F.softmax(attention_weights, dim=-1)

        out = torch.matmul(attention_probabilities, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_projection(out)
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor, activation_function: callable = F.relu) -> torch.Tensor:
        return self.linear2(activation_function(self.linear1(x)))
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_hidden_dim: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln1 = LayerNormalisation(d_model)
        self.ffn = FeedForwardNetwork(d_model, ff_hidden_dim)
        self.ln2 = LayerNormalisation(d_model)

    def forward(self, x, mask=None):
        identity = x
        x = self.ln1(x)  # Residual connection
        x = self.mha(x, mask=mask)
        x = x + identity

        identity = x
        x = self.ln2(x)  # Residual connection
        x = self.ffn(x)
        x = x + identity

        return x
    

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, ff_hidden_dim: int, num_layers: int):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, ff_hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        batch_size, seq_len, _ = x.shape
        if mask is None:
            # Create a causal mask: lower triangular matrix of 1s, upper triangular of 0s
            # Shape: (1, 1, seq_len, seq_len) for broadcasting
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        return self.output_layer(x)
