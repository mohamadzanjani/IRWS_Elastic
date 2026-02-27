from data_loader import load_hamshahri
from evaluation_utils import evaluate_and_save

DATA_DIR = "../data/hamshahri2"


df_docs, df_queries, qrels = load_hamshahri(DATA_DIR)

# TITLE
evaluate_and_save(
    df_docs["TITLE"].fillna(""),
    df_docs,
    df_queries,
    qrels,
    "metrics_title.csv"
)

# TEXT
evaluate_and_save(
    df_docs["TEXT"].fillna(""),
    df_docs,
    df_queries,
    qrels,
    "metrics_text.csv"
)

# TITLE + TEXT
evaluate_and_save(
    df_docs["TITLE"].fillna("") + " " + df_docs["TEXT"].fillna(""),
    df_docs,
    df_queries,
    qrels,
    "metrics_title_text.csv"
)