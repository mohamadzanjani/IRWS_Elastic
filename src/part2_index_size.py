import matplotlib.pyplot as plt
from data_loader import load_hamshahri
from es_indexer import create_index, index_documents
from es_search_eval import search_es

DATA_DIR = "../data/hamshahri2"
PERCENTS = [1, 2, 4, 8, 16, 32, 64, 100]

df_docs, df_queries, qrels = load_hamshahri(DATA_DIR)

times = []
ndcgs = []

for p in PERCENTS:
    print(f"\n=== {p}% of data ===")

    sample_size = int(len(df_docs) * p / 100)
    df_sample = df_docs.iloc[:sample_size]

    index_name = f"hamshahri_{p}"
    create_index(index_name)
    index_documents(index_name, df_sample)

    elapsed, ndcg = search_es(
        index_name,
        df_queries.TITLE.tolist(),
        df_queries,
        df_sample,
        qrels
    )

    times.append(elapsed)
    ndcgs.append(ndcg)

# ----------- Plot Response Time -----------
plt.figure()
plt.plot(PERCENTS, times, marker="o")
plt.xlabel("Index Size (%)")
plt.ylabel("Response Time (seconds)")
plt.title("Response Time vs Index Size")
plt.grid()
plt.savefig("../plots/response_time.png")

# ----------- Plot NDCG --------------------
plt.figure()
plt.plot(PERCENTS, ndcgs, marker="o")
plt.xlabel("Index Size (%)")
plt.ylabel("NDCG@10")
plt.title("NDCG@10 vs Index Size")
plt.grid()
plt.savefig("../plots/ndcg.png")

plt.show()