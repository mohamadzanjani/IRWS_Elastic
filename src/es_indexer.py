from es_config import get_es_client
from tqdm import tqdm

es = get_es_client()

def create_index(index_name):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "doc_id": {"type": "keyword"},
                "content": {"type": "text"}
            }
        }
    )


def index_documents(index_name, df_docs):
    for row in tqdm(df_docs.itertuples(), total=len(df_docs)):
        es.index(
            index=index_name,
            document={
                "doc_id": row.DOCID,
                "content": f"{row.TITLE} {row.TEXT}"
            }
        )
    es.indices.refresh(index=index_name)