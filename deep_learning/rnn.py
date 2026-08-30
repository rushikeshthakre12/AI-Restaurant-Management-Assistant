"""
Practical 3 (CO2): RNN / LSTM / GRU for next-word prediction, trained on
data/restaurant_corpus.txt using PyTorch. Small word-level language model --
the point is to demonstrate the syllabus concept (hidden state passing
through a recurrent cell) with a model you can actually explain line by
line, not to build a production text generator.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.preprocessing import tokenize

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "restaurant_corpus.txt"
MODEL_DIR = BASE_DIR / "ml" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 3
EMBED_DIM = 32
HIDDEN_DIM = 64


def build_vocab():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    all_tokens = []
    for line in lines:
        all_tokens.extend(tokenize(line))
    vocab = sorted(set(all_tokens))
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}  # 0 reserved for padding
    word2idx["<pad>"] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, lines


def build_training_pairs(word2idx, lines, seq_len=SEQ_LEN):
    """Slide a window of seq_len tokens across each line; target = next token."""
    X, y = [], []
    for line in lines:
        tokens = tokenize(line)
        ids = [word2idx[t] for t in tokens if t in word2idx]
        for i in range(len(ids) - seq_len):
            X.append(ids[i:i + seq_len])
            y.append(ids[i + seq_len])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class NextWordRNN(nn.Module):
    """cell_type in {'rnn', 'lstm', 'gru'} — same wrapper, different recurrent cell."""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, cell_type="lstm"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        cell_map = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self.rnn = cell_map[cell_type](embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.cell_type = cell_type

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        last_output = output[:, -1, :]  # take the final timestep's hidden state
        logits = self.fc(last_output)
        return logits


def train_model(cell_type: str, epochs: int = 200, lr: float = 0.01):
    word2idx, idx2word, lines = build_vocab()
    X, y = build_training_pairs(word2idx, lines)
    model = NextWordRNN(vocab_size=len(word2idx), cell_type=cell_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            print(f"[{cell_type}] epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  train_acc={acc:.4f}")

    return model, word2idx, idx2word, losses[-1]


def predict_next_word(model, word2idx, idx2word, seed_text: str, seq_len=SEQ_LEN):
    model.eval()
    tokens = tokenize(seed_text)[-seq_len:]
    ids = [word2idx.get(t, 0) for t in tokens]
    while len(ids) < seq_len:
        ids = [0] + ids
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long))
        pred_id = logits.argmax(dim=1).item()
    return idx2word.get(pred_id, "<unk>")


if __name__ == "__main__":
    for cell in ["rnn", "lstm", "gru"]:
        model, word2idx, idx2word, final_loss = train_model(cell)
        torch.save(model.state_dict(), MODEL_DIR / f"nextword_{cell}.pt")
        test_prompt = "i want to"
        prediction = predict_next_word(model, word2idx, idx2word, test_prompt)
        print(f"{cell.upper()}: final_loss={final_loss:.4f}  '{test_prompt}' -> '{prediction}'\n")
