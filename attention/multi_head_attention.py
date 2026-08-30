"""
Practical 6 (CO4) continued: Multi-Head Attention -- run several
scaled-dot-product-attention "heads" in parallel on different learned
projections of Q/K/V, then concatenate and project back down. Each head can
learn to attend to a different kind of relationship in the sequence.
"""
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from attention.self_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x, batch_size):
        # (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self._split_heads(self.W_q(x), batch_size)
        K = self._split_heads(self.W_k(x), batch_size)
        V = self._split_heads(self.W_v(x), batch_size)

        head_outputs, head_weights = scaled_dot_product_attention(Q, K, V)

        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        concat = head_outputs.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.W_o(concat)
        return output, head_weights


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, embed_dim, num_heads = 1, 5, 8, 2
    x = torch.randn(batch, seq_len, embed_dim)

    mha = MultiHeadAttention(embed_dim, num_heads)
    output, head_weights = mha(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}  (same as input -- ready to feed the next layer)")
    print(f"Per-head attention weights shape: {tuple(head_weights.shape)}  (batch, heads, seq_len, seq_len)")
    for h in range(num_heads):
        print(f"Head {h} attention row 0: {head_weights[0, h, 0].detach().numpy().round(3)}")
