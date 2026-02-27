from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple


class RankerTFIDF:

    def __init__(self, docs: List[str]) -> None:
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)

    def batch_search(
        self,
        queries: List[str],
        k: int = 1000
    ) -> Dict[str, List[Tuple[int, float]]]:

        query_vecs = self.vectorizer.transform(queries)
        sim_matrix = cosine_similarity(query_vecs, self.tfidf_matrix)

        results = {}
        for i, query in enumerate(queries):
            similarities = sim_matrix[i]
            top_indices = similarities.argsort()[-k:][::-1]
            results[str(i + 1)] = [
                (int(doc_id), float(similarities[doc_id]))
                for doc_id in top_indices
            ]
        return results