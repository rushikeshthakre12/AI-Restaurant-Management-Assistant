"""
Practical 8 (CO5): A small Transformer encoder block built from the pieces
in attention/ -- Multi-Head Attention + Positional Encoding + LayerNorm +
a feed-forward sublayer with residual connections, matching the standard
Transformer encoder layer diagram.
"""
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from attention.multi_head_attention import MultiHeadAttention
from attention.positional_encoding import PositionalEncoding
from attention.layer_norm import LayerNorm


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    x -> MultiHeadAttention -> Add & LayerNorm -> FeedForward -> Add & LayerNorm
    Residual ("Add") connections let gradients flow through the block even
    when a sublayer's output is close to zero early in training.
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + attn_out)          # residual connection + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)            # residual connection + LayerNorm
        return x, attn_weights


class MiniTransformerClassifier(nn.Module):
    """
    A minimal but genuine Transformer: embedding -> positional encoding ->
    N encoder blocks -> mean-pool -> linear classifier head.
    Used here for intent classification as a second, deep-learning-based
    classifier to compare against the TF-IDF + Logistic Regression baseline
    in ml/train_intent.py.
    """
    def __init__(self, vocab_size, num_classes, embed_dim=32, num_heads=4, ff_dim=64, num_layers=2, max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        h = self.embedding(x)
        h = self.pos_encoding(h)
        for block in self.blocks:
            h, _ = block(h)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size, num_classes, seq_len, batch = 200, 5, 10, 4
    x = torch.randint(1, vocab_size, (batch, seq_len))

    model = MiniTransformerClassifier(vocab_size, num_classes)
    logits = model(x)
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)}  (batch, num_classes)")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
