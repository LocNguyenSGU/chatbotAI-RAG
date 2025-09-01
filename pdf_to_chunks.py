import fitz
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download("punkt", quiet=True)

# ------------------ Cấu hình ------------------
PDF_PATH = "/Users/nguyenhuuloc/Downloads/DeCuong_KLTN_STEM_AI_Kiet_Loc_HK1_2025_2026.pdf"
JSON_OUTPUT = "pdf_chunks_semantic_safe.json"
FAISS_INDEX = "pdf_safe.index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 100
OVERLAP = 3
SIM_THRESHOLD = 0.95  # Chỉ đánh dấu trùng khi similarity > threshold

# ------------------ Load embedding ------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ------------------ Chunk semantic + loại trùng ------------------
def chunk_text_semantic_safe(text, max_tokens=MAX_TOKENS, overlap=OVERLAP, sim_threshold=SIM_THRESHOLD):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [], []

    chunks = []
    chunks_emb = []
    chunk = []
    chunk_tokens = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if chunk_tokens + sent_len > max_tokens:
            if chunk:
                chunk_text = " ".join(chunk)
                emb = embed_model.encode(chunk_text)
                # Kiểm tra similarity với tất cả chunk đã lưu
                if not chunks_emb or all(cosine_similarity([emb], [ce])[0][0] < sim_threshold for ce in chunks_emb):
                    chunks.append(chunk_text)
                    chunks_emb.append(emb)
            # chunk mới với overlap
            chunk = chunk[-overlap:] if len(chunk) > overlap else chunk[:]
            chunk_tokens = sum(len(s.split()) for s in chunk)

        chunk.append(sent)
        chunk_tokens += sent_len

    # Chunk cuối
    if chunk:
        chunk_text = " ".join(chunk)
        emb = embed_model.encode(chunk_text)
        if not chunks_emb or all(cosine_similarity([emb], [ce])[0][0] < sim_threshold for ce in chunks_emb):
            chunks.append(chunk_text)
            chunks_emb.append(emb)

    return chunks, chunks_emb

# ------------------ Convert PDF sang JSON + FAISS ------------------
def pdf_to_json_and_faiss_safe(pdf_path, output_json, faiss_index):
    import faiss

    doc = fitz.open(pdf_path)
    data = []
    embeddings = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text.strip():
            continue

        chunks, chunks_emb = chunk_text_semantic_safe(text)
        for i, (chunk, emb) in enumerate(zip(chunks, chunks_emb)):
            data.append({
                "content": chunk,
                "source": os.path.basename(pdf_path),
                "page": page_num,
                "chunk_id": f"{page_num}_{i}"
            })
            embeddings.append(emb)

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Build FAISS
    if embeddings:
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype='float32'))
        faiss.write_index(index, faiss_index)

    print(f"✅ Lưu {len(data)} chunk vào {output_json} và tạo FAISS index tại {faiss_index}")

# ------------------ Main ------------------
if __name__ == "__main__":
    pdf_to_json_and_faiss_safe(PDF_PATH, JSON_OUTPUT, FAISS_INDEX)