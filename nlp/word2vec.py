"""
Practical 2 (CO1): Word2Vec CBOW and Skip-gram, trained on
data/restaurant_corpus.txt using Gensim. Small corpus by design, so both
models train in a few seconds and results are easy to explain in a viva.
"""
import sys
from pathlib import Path
from gensim.models import Word2Vec

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.preprocessing import tokenize

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "restaurant_corpus.txt"
MODEL_DIR = BASE_DIR / "ml" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_sentences() -> list[list[str]]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return [tokenize(line) for line in lines]


def train_word2vec(sg: int, vector_size: int = 50, window: int = 4, min_count: int = 1, epochs: int = 100) -> Word2Vec:
    """sg=0 -> CBOW, sg=1 -> Skip-gram (Gensim's own convention)."""
    sentences = load_sentences()
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        seed=42,
    )
    return model


def demo():
    print("Training CBOW model...")
    cbow = train_word2vec(sg=0)
    cbow.save(str(MODEL_DIR / "word2vec_cbow.model"))

    print("Training Skip-gram model...")
    skipgram = train_word2vec(sg=1)
    skipgram.save(str(MODEL_DIR / "word2vec_skipgram.model"))

    test_pairs = [("pizza", "pasta"), ("biryani", "rice"), ("booking", "reservation")]
    query_words = ["pizza", "biryani", "spicy", "booking"]

    for name, model in [("CBOW", cbow), ("Skip-gram", skipgram)]:
        print(f"\n--- {name} ---")
        vocab = set(model.wv.index_to_key)
        print(f"Vocabulary size: {len(vocab)}")
        for w1, w2 in test_pairs:
            if w1 in vocab and w2 in vocab:
                sim = model.wv.similarity(w1, w2)
                print(f"similarity({w1}, {w2}) = {sim:.4f}")
        for w in query_words:
            if w in vocab:
                most_similar = model.wv.most_similar(w, topn=3)
                print(f"most_similar('{w}') = {most_similar}")


if __name__ == "__main__":
    demo()
