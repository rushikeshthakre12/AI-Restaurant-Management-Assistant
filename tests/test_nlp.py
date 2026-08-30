import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from nlp.preprocessing import preprocess, tokenize, remove_stopwords, lemmatize
from nlp.ner import extract_entities, extract_people_count, extract_times


def test_tokenize_basic():
    tokens = tokenize("Book a table for 4 people!")
    assert "book" in tokens
    assert "table" in tokens


def test_remove_stopwords():
    tokens = ["i", "want", "to", "book", "a", "table"]
    filtered = remove_stopwords(tokens)
    assert "i" not in filtered
    assert "book" in filtered


def test_lemmatize_running_to_run():
    lemmas = lemmatize(["running", "tables"])
    assert "run" in lemmas
    assert "table" in lemmas


def test_preprocess_returns_all_stages():
    result = preprocess("I want to book a table")
    assert set(result.keys()) == {"original", "cleaned", "tokens", "pos_tags", "tokens_no_stopwords", "lemmas"}


def test_ner_extracts_people_count():
    assert extract_people_count("book a table for 4 people") == 4


def test_ner_extracts_time():
    times = extract_times("book a table at 8 pm")
    assert any("8" in t for t in times)


def test_ner_full_entities():
    entities = extract_entities("book a table for 6 people on saturday at 7:30 pm")
    assert entities["NUMBER_OF_PEOPLE"] == 6
    assert "saturday" in entities["DATE"]
    assert any("7:30" in t for t in entities["TIME"])
