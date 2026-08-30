"""
Practical 7 (CO5) continued: Layer Normalization, plus a short comparison
against Batch Normalization to explain why Transformers use LayerNorm.

Batch Normalization normalizes each *feature* across the batch dimension --
it needs a reasonably large batch to estimate stable statistics, and its
behavior differs between training and inference (running averages).

Layer Normalization normalizes across the *feature* dimension for each
individual sample independently -- this works with batch size 1 (important
for variable-length sequences and autoregressive decoding) and behaves
identically in training and inference, which is why Transformers use it
instead of BatchNorm.

"Internal Covariate Shift" is the term for the problem both aim to fix:
as parameters update during training, the distribution of each layer's
inputs keeps shifting, which slows convergence. Normalizing re-centers and
re-scales those inputs at every layer so downstream layers see a more
stable input distribution.
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Manual implementation matching nn.LayerNorm's math, for the viva."""
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))   # learned scale
        self.beta = nn.Parameter(torch.zeros(embed_dim))   # learned shift
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8) * 5 + 3  # arbitrary scale/shift to make normalization visible

    manual_ln = LayerNorm(embed_dim=8)
    builtin_ln = nn.LayerNorm(8)
    # copy builtin's (default) gamma=1, beta=0 so the two are directly comparable
    manual_out = manual_ln(x)
    builtin_out = builtin_ln(x)

    print("Input mean/std per sample (should NOT be 0/1):")
    print(x[0].mean(dim=-1).detach().numpy().round(3), x[0].std(dim=-1).detach().numpy().round(3))

    print("\nManual LayerNorm output mean/std per sample (should be ~0/~1):")
    print(manual_out[0].mean(dim=-1).detach().numpy().round(4), manual_out[0].std(dim=-1).detach().numpy().round(4))

    max_diff = (manual_out - builtin_out).abs().max().item()
    print(f"\nMax difference vs torch.nn.LayerNorm: {max_diff:.6f} (should be ~0, confirms correctness)")
