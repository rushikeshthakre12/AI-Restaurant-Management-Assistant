"""
Practical 4 (CO2): LSTM Encoder-Decoder sequence-to-sequence model.

Task: map a short customer request to a short canonical intent phrase, e.g.
  "book a table for four"       -> "table booking request"
  "cancel my reservation"       -> "cancel booking request"
This is a toy but real seq2seq task -- small enough to train in seconds,
big enough to show encoder hidden/cell state -> decoder generation clearly.
"""
import sys
from pathlib import Path
import random

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.preprocessing import tokenize

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("book a table for four", "table booking request"),
    ("reserve a table for two", "table booking request"),
    ("i want to book a table", "table booking request"),
    ("cancel my reservation", "cancel booking request"),
    ("cancel my table booking", "cancel booking request"),
    ("i want two paneer pizzas", "place order request"),
    ("order one chicken biryani", "place order request"),
    ("i want to order food", "place order request"),
    ("show me the menu", "menu query request"),
    ("what dishes do you have", "menu query request"),
    ("suggest something spicy", "recommendation request"),
    ("recommend a good dish", "recommendation request"),
    ("cancel my order", "cancel order request"),
    ("i want to cancel my order", "cancel order request"),
    ("what are your timings", "hours query request"),
    ("when do you open", "hours query request"),
]

SOS, EOS, PAD = "<sos>", "<eos>", "<pad>"


def build_vocab(pairs):
    tokens = set()
    for src, tgt in pairs:
        tokens |= set(tokenize(src)) | set(tokenize(tgt))
    vocab = [PAD, SOS, EOS] + sorted(tokens)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def encode(tokens, word2idx):
    # Unknown tokens fall back to <pad>'s id rather than crashing -- this is
    # a tiny fixed-vocabulary demo model, not an open-vocabulary system.
    return [word2idx.get(t, word2idx[PAD]) for t in tokens]


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell  # outputs feed the attention module in attention/attention.py


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.fc(output)
        return logits, hidden, cell


def train_seq2seq(epochs=300, lr=0.01):
    word2idx, idx2word = build_vocab(PAIRS)
    vocab_size = len(word2idx)
    encoder = Encoder(vocab_size)
    decoder = Decoder(vocab_size)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD])

    for epoch in range(epochs):
        total_loss = 0.0
        for src_text, tgt_text in PAIRS:
            src_ids = torch.tensor([encode(tokenize(src_text), word2idx)])
            tgt_ids = [word2idx[SOS]] + encode(tokenize(tgt_text), word2idx) + [word2idx[EOS]]
            tgt_ids = torch.tensor([tgt_ids])

            optimizer.zero_grad()
            _, hidden, cell = encoder(src_ids)
            decoder_input = tgt_ids[:, :-1]
            target = tgt_ids[:, 1:]
            logits, _, _ = decoder(decoder_input, hidden, cell)
            loss = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch+1}/{epochs}  avg_loss={total_loss/len(PAIRS):.4f}")

    return encoder, decoder, word2idx, idx2word


def generate(encoder, decoder, word2idx, idx2word, src_text, max_len=6):
    encoder.eval()
    decoder.eval()
    src_ids = torch.tensor([encode(tokenize(src_text), word2idx)])
    with torch.no_grad():
        _, hidden, cell = encoder(src_ids)
        input_id = torch.tensor([[word2idx[SOS]]])
        output_words = []
        for _ in range(max_len):
            logits, hidden, cell = decoder(input_id, hidden, cell)
            next_id = logits[0, -1].argmax().item()
            if idx2word[next_id] == EOS:
                break
            output_words.append(idx2word[next_id])
            input_id = torch.tensor([[next_id]])
    return " ".join(output_words)


if __name__ == "__main__":
    encoder, decoder, word2idx, idx2word = train_seq2seq()
    torch.save(encoder.state_dict(), MODEL_DIR / "seq2seq_encoder.pt")
    torch.save(decoder.state_dict(), MODEL_DIR / "seq2seq_decoder.pt")

    for test in ["book a table for two", "i want to order pizza", "cancel my table"]:
        print(f"{test!r} -> {generate(encoder, decoder, word2idx, idx2word, test)!r}")
