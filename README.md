# AI Restaurant Management Assistant

A working MCA integrated project combining NLP, Machine Learning, Deep
Learning, Transformer components, a recommendation system, sentiment
analysis, MySQL-style relational data, and a Streamlit UI.

**This is a fully runnable project** — every module below has been
executed end-to-end while building it (real training runs, real DB writes,
real Streamlit boot test). Numbers you see when you run it yourself are
computed on the fly, not hard-coded.

## Two honest substitutions vs. the original spec

1. **Database: SQLite by default, MySQL supported.** No live MySQL server
   was available in the environment this was built in. `database/connection.py`
   defaults to SQLite (`data/restaurant.db`, zero setup) and switches to a
   real MySQL server if you set `DB_BACKEND=mysql` in `.env` — the
   MySQL-compatible schema is in `database/schema.sql`, ready to run against
   a real server for your submission.
2. **Generative response step: template-based by default, Hugging Face
   optional.** Downloading a pretrained Hugging Face model requires internet
   access, which this environment doesn't have. `transformer/chatbot.py`
   defaults to a template-based response generator (see `_TEMPLATES`); set
   `TRANSFORMERS_USE_LLM=1` on a machine with internet and it will call a
   real Hugging Face `text2text-generation` pipeline instead, using the
   exact same few-shot prompt built in `transformer/prompts.py`.

Everything else — NLP pipeline, Word2Vec, ML intent classifier, recommendation
engine, sentiment analysis, RNN/LSTM/GRU, encoder-decoder, attention,
self-attention, multi-head attention, positional encoding, LayerNorm, the
Streamlit app, booking/ordering logic, PDF bills, K-Means segmentation,
demand prediction, and the pytest suite — runs exactly as shipped.

## Quick start

```bash
python -m venv venv
source venv/bin/activate                # Windows: venv\Scripts\activate

pip install -r requirements.txt

python -c "import nltk; [nltk.download(p, quiet=True) for p in \
  ['punkt','punkt_tab','stopwords','wordnet','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','omw-1.4']]"

cp .env.example .env        # defaults work as-is for SQLite

# Seed the database (menu, users, offers, reviews with real computed sentiment)
python -m database.seed
# Optional: realistic demo order/booking history, so dashboard analytics
# and K-Means segmentation have something to compute from
python -m database.seed_demo_orders

# Launch the app
streamlit run app.py
```

Demo accounts (created by `database.seed`):
- Admin: `admin@restaurant.com` / `admin123`
- Customer: `customer@example.com` / `customer123`

**Important:** modules that import across packages (almost everything except
`app.py`) must be run with `python -m package.module`, not
`python package/module.py` — e.g. `python -m ml.train_intent`, not
`python ml/train_intent.py`. Running a script directly makes Python treat
its own folder as the import root, which breaks same-named package imports
(a standard Python quirk, not a bug in this code).

## Run each AI/ML module standalone (for viva demonstration)

```bash
python -m nlp.preprocessing          # tokenize/stopwords/lemmatize/POS demo
python -m nlp.ner                    # custom restaurant NER demo
python -m nlp.word2vec               # trains CBOW + Skip-gram, shows similarities
python -m ml.train_intent            # trains + evaluates Logistic Regression vs Naive Bayes
python -m ml.evaluation              # prints the saved evaluation report
python -m ml.recommendation          # content-based + personalized recommendation demo
python -m ml.sentiment               # aspect-level rule scorer + trained NB classifier
python -m ml.clustering              # K-Means customer segmentation (needs seeded orders)
python -m ml.prediction              # next-day order/revenue prediction (needs seeded orders)
python -m deep_learning.rnn          # trains RNN/LSTM/GRU next-word predictors
python -m deep_learning.encoder_decoder   # trains the LSTM seq2seq model
python -m attention.attention        # Bahdanau attention on the seq2seq model
python -m attention.self_attention   # self-attention / scaled dot-product from scratch
python -m attention.multi_head_attention
python -m attention.positional_encoding
python -m attention.layer_norm       # verified bit-exact against torch.nn.LayerNorm
python -m transformer.model          # assembled mini Transformer encoder
python -m transformer.prompts        # zero/one/few-shot prompt construction demo
python -m transformer.chatbot        # full intent+NER+template response pipeline
python -m services.chatbot_service   # the orchestrator the UI actually calls
python -m utils.pdf_bill             # generates a sample PDF bill to data/bills/
```

## Run the tests

```bash
python -m pytest tests/ -v
```
23 tests across NLP, ML, booking, ordering, and recommendation — all passing
against a freshly seeded database.

## Project structure

```
app.py                  Streamlit entry point (login/register, role routing)
database/                connection.py (SQLite/MySQL), schema.sql, seed.py, seed_demo_orders.py
nlp/                     preprocessing.py, ner.py, word2vec.py
ml/                      train_intent.py, predict_intent.py, evaluation.py,
                         recommendation.py, sentiment.py, clustering.py, prediction.py
                         saved_models/ (trained artifacts committed so the app and
                         admin dashboard work immediately without a training step first)
deep_learning/           rnn.py (RNN/LSTM/GRU next-word models in one file, selectable
                         via cell_type), encoder_decoder.py
attention/               attention.py, self_attention.py, multi_head_attention.py,
                         positional_encoding.py, layer_norm.py
transformer/             model.py, chatbot.py, prompts.py
services/                booking_service.py, order_service.py, menu_service.py,
                         recommendation_service.py, chatbot_service.py, admin_service.py
ui/                      customer.py, admin.py, components.py
utils/                   helpers.py (password hashing), pdf_bill.py
tests/                   test_nlp.py, test_ml.py, test_booking.py, test_order.py, test_recommendation.py
data/                    menu.csv, intents.csv, reviews.csv, restaurant_corpus.txt
```

**Note on notebooks:** the original brief's syllabus practicals (NLP pipeline,
Word2Vec, RNN/LSTM/GRU, encoder-decoder, attention, transformer + prompting)
each map to one runnable `.py` module listed above with a `if __name__ ==
"__main__":` demo block, rather than separate `.ipynb` files — copy any
module's code into a notebook cell if your course requires `.ipynb`
submissions specifically.

## Honest notes on results (for your viva / report)

- **Intent classifier: 100% test accuracy.** This is real, but it's an
  easy result — `data/intents.csv` is generated from ~15 phrasing templates
  per intent with slot-filled numbers/times, so train and test share very
  similar phrasing. Say this plainly in your report rather than presenting
  it as a hard benchmark: to make it a harder, more convincing test, add
  more human-written (not templated) example sentences per intent.
- **Sentiment classifier: ~60% test accuracy** on a small (30-review) hand
  written dataset with an 8-review test split — a small, honest number
  that's easy to defend, versus a suspiciously perfect one.
- **Demand prediction** has a wide margin of error on the small seeded
  order history — treat it as a directional trend, not a precise forecast,
  and say so if asked.
- Every metric shown in the admin dashboard or printed by the modules above
  is computed at run time from the seeded data — nothing is hard-coded.

## Subject-to-component mapping

| Subject | Where it lives |
|---|---|
| NLP | `nlp/` (pipeline, NER, Word2Vec), `attention/` |
| Machine Learning | `ml/` (intent classification, recommendation, sentiment, clustering, prediction) |
| Deep Learning | `deep_learning/` (RNN, LSTM, GRU, encoder-decoder) |
| Generative AI / Transformers | `transformer/` (self-attention through to the assembled encoder, prompting, response generation) |
| DBMS | `database/` (schema, connection, seeding), used throughout `services/` |
| Software Engineering | modular `services/` layer, `tests/`, this README |
| Data Analytics | `ui/admin.py`, `services/admin_service.py` |

## Next steps if you want to extend this further

- Swap in a real MySQL server (`DB_BACKEND=mysql` in `.env`) for the final submission.
- Add human-written (non-templated) intent examples to make the classifier's
  reported accuracy more meaningful.
- Point `transformer/chatbot.py` at a real Hugging Face model
  (`TRANSFORMERS_USE_LLM=1`) once you have internet access, for genuinely
  generative (not template) replies.
- Prepare the SRS/UML/ER diagrams and final project report documents
  described in the original brief — those are documentation deliverables
  this codebase supports but doesn't generate for you.
