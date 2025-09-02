from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, normalize: bool = True):
        embs = self.model.encode(texts, convert_to_numpy=True)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.clip(norms, 1e-10, None)
        return embs