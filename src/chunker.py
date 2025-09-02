import numpy as np
from utils import tokenize_sentences
import faiss

def chunk_text_semantic_safe(text, embedder, faiss_index=None, max_tokens=300, overlap=50, sim_threshold=0.95):
    sentences = tokenize_sentences(text)
    if not sentences:
        return [], []

    chunks, chunks_emb = [], []
    current_chunk = []

    for sent in sentences:
        current_chunk.append(sent)
        token_count = sum(len(s.split()) for s in current_chunk)

        if token_count >= max_tokens or sent == sentences[-1]:
            chunk_text = " ".join(current_chunk).strip()
            if not chunk_text:
                current_chunk = []
                continue

            emb = embedder.encode([chunk_text])[0]
            emb = np.array([emb], dtype='float32')
            faiss.normalize_L2(emb)

            # Duplicate check
            is_duplicate = False
            if faiss_index and faiss_index.ntotal > 0:
                distances, _ = faiss_index.search(emb, 1)
                if distances[0][0] < 1 - sim_threshold:
                    is_duplicate = True

            if not is_duplicate:
                chunks.append(chunk_text)
                chunks_emb.append(emb[0])
                if faiss_index is not None:
                    faiss_index.add(emb)

            # Reset chunk với overlap
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []

    return chunks, chunks_emb