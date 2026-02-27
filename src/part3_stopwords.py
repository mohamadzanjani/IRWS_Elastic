from collections import Counter
from data_loader import load_hamshahri
from evaluation_utils import evaluate_and_save
from parsivar import Normalizer, Tokenizer

DATA_DIR = "../data/hamshahri2"

df_docs, df_queries, qrels = load_hamshahri(DATA_DIR)

best_docs = df_docs["TITLE"].fillna("") + " " + df_docs["TEXT"].fillna("")

normalizer = Normalizer()
tokenizer = Tokenizer()


def normalize_and_tokenize(text):
    t = normalizer.normalize(text)
    tokens = tokenizer.tokenize_words(t)
    return " ".join(tokens)


docs_tokens = best_docs.apply(normalize_and_tokenize)

all_tokens = []
for text in docs_tokens:
    all_tokens.extend(text.split())

freq = Counter(all_tokens)


def remove_top_k(text, stopwords):
    return " ".join([w for w in text.split() if w not in stopwords])


for k in [100, 500, 1000]:
    stopwords = set(w for w, _ in freq.most_common(k))
    cleaned_docs = docs_tokens.apply(
        lambda t: remove_top_k(t, stopwords)
    )

    evaluate_and_save(
        cleaned_docs,
        df_docs,
        df_queries,
        qrels,
        f"metrics_stopwords_{k}.csv"
    )