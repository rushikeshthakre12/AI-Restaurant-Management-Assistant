"""
Practical 6 (CO4): Self-Attention and Scaled Dot-Product Attention,
implemented directly from the formula (no nn.MultiheadAttention shortcut)
so Query/Key/Value and the softmax(QK^T / sqrt(d_k))V computation are all
visible.
"""
import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


class SelfAttention(nn.Module):
    """Single-head self-attention: learns its own Q, K, V projections from
    the same input sequence (hence 'self')."""
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, embed_dim = 1, 5, 8
    x = torch.randn(batch, seq_len, embed_dim)

    self_attn = SelfAttention(embed_dim)
    output, weights = self_attn(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Attention weight matrix shape: {tuple(weights.shape)}  (each row sums to 1)")
    print(f"Row sums (should all be ~1.0): {weights.sum(dim=-1).detach().numpy().round(4)}")
