import faiss
from utils import fuzzy_score

class SemanticSearcher:
    def __init__(self, embedder, faiss_index, metadata, top_k=10, score_threshold=0.5):
        self.embedder = embedder
        self.index = faiss_index
        self.metadata = metadata
        self.top_k = top_k
        self.score_threshold = score_threshold

    def search(self, query, top_k=None, score_threshold=None):
        if top_k is None: top_k = self.top_k
        if score_threshold is None: score_threshold = self.score_threshold

        q_emb = self.embedder.encode([query])
        distances, indices = self.index.search(q_emb, top_k*5)

        results = []
        seen = set()
        query_lower = query.lower()
        keywords = set(query_lower.split())

        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0: continue
            chunk = self.metadata[idx]["content"]
            key = self.metadata[idx]["chunk_id"]
            if key in seen: continue
            seen.add(key)

            sem_score = float(1 - dist)
            exact_score = 2.0 if query_lower in chunk.lower() else 0.0
            keyword_score = sum(0.5 for kw in keywords if kw in chunk.lower())
            fuzzy_val = fuzzy_score(query_lower, chunk.lower())
            final_score = sem_score + exact_score + keyword_score + fuzzy_val

            if final_score >= score_threshold:
                r = self.metadata[idx].copy()
                r["score"] = final_score
                results.append(r)

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]