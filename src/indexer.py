import os
import json
import faiss
import numpy as np

from chunker import chunk_text_semantic_safe
from utils import clean_text

def build_faiss_index(pdf_texts, embedder, index_path, json_path, max_tokens=300, overlap=50, sim_threshold=0.95):
    faiss_index = faiss.IndexFlatL2(embedder.dim)
    all_chunks = []
    all_embeddings = []

    for source_name, text in pdf_texts.items():
        text = clean_text(text)
        chunks, chunks_emb = chunk_text_semantic_safe(
            text, embedder, faiss_index=faiss_index,
            max_tokens=max_tokens, overlap=overlap, sim_threshold=sim_threshold
        )

        for i, (chunk, emb) in enumerate(zip(chunks, chunks_emb)):
            all_chunks.append({
                "content": chunk,
                "source": source_name,
                "chunk_id": f"{source_name}_{i}"
            })
            all_embeddings.append(emb)

    # Save FAISS index
    if all_embeddings:
        embeddings_array = np.array(all_embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_array)
        index = faiss.IndexFlatL2(embedder.dim)
        index.add(embeddings_array)
        faiss.write_index(index, index_path)

    # Save chunks JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    return all_chunks, faiss_index