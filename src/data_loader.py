import pandas as pd


def load_hamshahri(data_dir: str):
    df_docs = pd.read_csv(
        f"{data_dir}/docs.csv.gz", compression="gzip"
    )
    df_judgments = pd.read_csv(
        f"{data_dir}/judgments_dataframe.csv.gz", compression="gzip"
    )
    df_queries_fa = pd.read_csv(
        f"{data_dir}/queries_fa.csv.gz", compression="gzip"
    )

    # build qrels
    qrels = {}
    grouped = df_judgments[df_judgments["relevancy"] == 1].groupby("query_id")
    for qid, group in grouped:
        qrels[str(qid)] = {
            row.doc_id: int(row.relevancy)
            for row in group.itertuples()
        }

    return df_docs, df_queries_fa, qrels