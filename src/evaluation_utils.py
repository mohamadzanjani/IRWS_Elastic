import pandas as pd
from beir.retrieval.evaluation import EvaluateRetrieval


def get_metrics_dataframe(metrics):
    data = []
    for metric_group in metrics:
        for metric_name, score in metric_group.items():
            metric, k = metric_name.split("@")
            data.append({
                "Metric": metric,
                "k": int(k),
                "Score": score
            })

    df = pd.DataFrame(data)
    return df.pivot(index="k", columns="Metric", values="Score").reset_index()


def evaluate_and_save(
    doc_series,
    df_docs,
    df_queries,
    qrels,
    filename
):
    from ranker_tfidf import RankerTFIDF

    print(f"\nRunning experiment → {filename}")

    ranker = RankerTFIDF(doc_series.tolist())
    queries = list(df_queries.TITLE.values)

    results = ranker.batch_search(queries, k=1000)

    ranked_results = {}
    for qid, doc_scores in results.items():
        ranked_results[qid] = {
            df_docs.at[doc_id, "DOCID"]: score
            for doc_id, score in doc_scores
        }

    top_k = [1, 3, 5, 10, 100, 1000]
    metrics = EvaluateRetrieval.evaluate(qrels, ranked_results, top_k)

    metrics_df = get_metrics_dataframe(metrics)
    metrics_df.to_csv(filename, index=False)

    print(f"Saved → {filename}")
    return metrics