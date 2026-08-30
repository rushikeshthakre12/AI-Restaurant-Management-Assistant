"""
Practical 5 (CO3): Bahdanau-style additive attention added to the
Encoder-Decoder from deep_learning/encoder_decoder.py.

Demonstrates: attention scores, attention weights (softmax over scores),
and the resulting context vector, so you can show *why* attention helps the
decoder focus on the relevant input word at each generation step.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.preprocessing import tokenize
from deep_learning.encoder_decoder import Encoder, build_vocab, encode, PAIRS, SOS, EOS


class AdditiveAttention(nn.Module):
    """score(h_dec, h_enc_i) = v^T tanh(W [h_dec ; h_enc_i])"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, batch, hidden) -> (batch, 1, hidden)
        decoder_hidden = decoder_hidden.permute(1, 0, 2)
        seq_len = encoder_outputs.size(1)
        decoder_hidden_expanded = decoder_hidden.repeat(1, seq_len, 1)

        energy = torch.tanh(self.W(torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)             # raw attention scores, shape (batch, seq_len)
        weights = F.softmax(scores, dim=1)              # attention weights (sum to 1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # weighted sum
        return context, weights, scores


def demo():
    word2idx, idx2word = build_vocab(PAIRS)
    encoder = Encoder(vocab_size=len(word2idx))
    attention = AdditiveAttention(hidden_dim=64)

    test_sentence = "book a table for four people at 8 pm"
    tokens = [t for t in tokenize(test_sentence) if t in word2idx] or ["book"]
    src_ids = torch.tensor([encode(tokens, word2idx)])

    encoder_outputs, hidden, cell = encoder(src_ids)   # (1, seq_len, hidden)

    # simulate one decoder step's hidden state and inspect what it attends to
    fake_decoder_hidden = hidden  # (1, 1, hidden)
    context, weights, scores = attention(fake_decoder_hidden, encoder_outputs)

    print(f"Input tokens: {tokens}")
    print(f"Raw attention scores: {scores.detach().numpy().round(3)}")
    print(f"Attention weights (softmax, sum={weights.sum().item():.3f}): {weights.detach().numpy().round(3)}")
    print(f"Context vector shape: {tuple(context.shape)}")
    top_idx = weights.argmax(dim=1).item()
    print(f"Model is focusing most on token: '{tokens[top_idx]}'")


if __name__ == "__main__":
    demo()
