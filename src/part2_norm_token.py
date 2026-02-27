from parsivar import Normalizer, Tokenizer
from data_loader import load_hamshahri
from evaluation_utils import evaluate_and_save

DATA_DIR = "../data/hamshahri2"

df_docs, df_queries, qrels = load_hamshahri(DATA_DIR)

best_docs = df_docs["TITLE"].fillna("") + " " + df_docs["TEXT"].fillna("")

normalizer = Normalizer()
tokenizer = Tokenizer()


def normalize_and_tokenize(text):
    t = normalizer.normalize(text)
    tokens = tokenizer.tokenize_words(t)
    return " ".join(tokens)


docs_norm_tok = best_docs.apply(normalize_and_tokenize)

evaluate_and_save(
    docs_norm_tok,
    df_docs,
    df_queries,
    qrels,
    "metrics_title_text_norm_tok.csv"
)