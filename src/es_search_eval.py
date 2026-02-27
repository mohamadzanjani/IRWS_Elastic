import time
from beir.retrieval.evaluation import EvaluateRetrieval
from es_config import get_es_client

es = get_es_client()

def search_es(index_name, queries, df_queries, df_docs, qrels, k=10):
    start = time.time()

    results = {}
    for qid, query in zip(df_queries.index, queries):
        response = es.search(
            index=index_name,
            size=k,
            query={"match": {"content": query}}
        )

        results[str(qid)] = {
            hit["_source"]["doc_id"]: hit["_score"]
            for hit in response["hits"]["hits"]
        }

    elapsed_time = time.time() - start

    metrics = EvaluateRetrieval.evaluate(qrels, results, [k])

    print("Available metric keys:", metrics[0].keys())

    ndcg = (
        metrics[0].get(f"NDCG@{k}") or
        metrics[0].get(f"NDCG_{k}") or
        metrics[0].get("ndcg") or
        0
    )

    return elapsed_time, ndcg