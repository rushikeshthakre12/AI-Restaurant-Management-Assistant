"""
Practical 1 (CO1): classic NLP pipeline — tokenization, stopword removal,
lemmatization, POS tagging. Pure NLTK, no external model download required.
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation other than currency symbols we need for NER."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9₹.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return word_tokenize(clean_text(text))


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS]


def _wordnet_pos(tag: str) -> str:
    """Map a Penn Treebank POS tag to the WordNet POS tag lemmatize() expects."""
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return "n"


def lemmatize(tokens: list[str]) -> list[str]:
    tagged = pos_tag(tokens)
    return [_LEMMATIZER.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]


def pos_tags(tokens: list[str]) -> list[tuple[str, str]]:
    return pos_tag(tokens)


def preprocess(text: str, drop_stopwords: bool = True) -> dict:
    """Full pipeline in one call. Returns every intermediate stage so the
    result can be shown step-by-step in a viva demo."""
    tokens = tokenize(text)
    tagged = pos_tags(tokens)
    filtered = remove_stopwords(tokens) if drop_stopwords else tokens
    lemmas = lemmatize(filtered)
    return {
        "original": text,
        "cleaned": clean_text(text),
        "tokens": tokens,
        "pos_tags": tagged,
        "tokens_no_stopwords": filtered,
        "lemmas": lemmas,
    }


if __name__ == "__main__":
    sample = "I want to book a table for 4 people tomorrow at 8 PM."
    result = preprocess(sample)
    for k, v in result.items():
        print(f"{k}: {v}")
