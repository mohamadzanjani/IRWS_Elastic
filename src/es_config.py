from elasticsearch import Elasticsearch

ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "RaMJuzS67W1-5Q+cuR1d"

def get_es_client():
    return Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASSWORD)
    )