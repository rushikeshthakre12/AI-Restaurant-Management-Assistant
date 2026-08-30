"""
Practical 7 (CO5): Sinusoidal Positional Encoding.

Self-attention has no built-in notion of word order (it treats the input as
a set, not a sequence) -- positional encoding injects that order back in by
adding a fixed sinusoidal pattern to each token's embedding, using a
different frequency per dimension.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions -> sine
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions -> cosine
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, embed_dim), not a trainable parameter

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, embed_dim = 1, 6, 8
    x = torch.zeros(batch, seq_len, embed_dim)  # zero input so output == the encoding itself

    pos_enc = PositionalEncoding(embed_dim)
    encoded = pos_enc(x)

    print(f"Positional encoding for a {seq_len}-token sequence, embed_dim={embed_dim}:")
    print(encoded[0].detach().numpy().round(3))
    print("\nNote each position gets a unique pattern across dimensions, "
          "and the pattern for position p is fixed (not learned).")
