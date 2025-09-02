import fitz
import json
import os
import re
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)

# ------------------ Cấu hình ------------------
PDF_PATH = "/Users/nguyenhuuloc/Downloads/DeCuong_KLTN_STEM_AI_Kiet_Loc_HK1_2025_2026.pdf"
JSON_OUTPUT = "pdf_chunks_semantic.json"
FAISS_INDEX = "pdf.index"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300         # số token tối đa trong 1 chunk
OVERLAP = 30             # số token chồng lấn
SIM_THRESHOLD = 0.95     # ngưỡng similarity để loại chunk trùng lặp
REMOVE_HEADER_FOOTER = True  # bật/tắt lọc header/footer

# ------------------ Load embedding ------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


# ------------------ Tiền xử lý text ------------------
def clean_page_text(text):
    # Xoá khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    # Optionally loại header/footer
    if REMOVE_HEADER_FOOTER:
        lines = text.split("\n")
        if len(lines) > 3:
            # bỏ dòng đầu và cuối nếu lặp lại thường xuyên
            text = " ".join(lines[1:-1])
    return text


# ------------------ Chunk semantic + loại trùng ------------------
def chunk_text_semantic_safe(text, max_tokens=MAX_TOKENS, overlap=OVERLAP, sim_threshold=SIM_THRESHOLD):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [], []

    chunks, chunks_emb = [], []
    current_chunk, token_count = [], 0

    for sent in sentences:
        sent_len = len(sent.split())

        # Nếu thêm câu này vượt max_tokens → đóng chunk lại
        if token_count + sent_len > max_tokens and current_chunk:
            chunk_text = " ".join(current_chunk)
            emb = embed_model.encode(chunk_text)

            # check trùng lặp bằng cosine similarity
            if not chunks_emb or all(cosine_similarity([emb], [ce])[0][0] < sim_threshold for ce in chunks_emb):
                chunks.append(chunk_text)
                chunks_emb.append(emb)

            # reset chunk với overlap
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            token_count = sum(len(s.split()) for s in current_chunk)

        # Thêm câu vào chunk hiện tại
        current_chunk.append(sent)
        token_count += sent_len

    # Chunk cuối
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        emb = embed_model.encode(chunk_text)
        if not chunks_emb or all(cosine_similarity([emb], [ce])[0][0] < sim_threshold for ce in chunks_emb):
            chunks.append(chunk_text)
            chunks_emb.append(emb)

    return chunks, chunks_emb


# ------------------ Convert PDF sang JSON + FAISS ------------------
def pdf_to_json_and_faiss_safe(pdf_path, output_json, faiss_index):
    import faiss

    doc = fitz.open(pdf_path)
    data, embeddings = [], []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text.strip():
            continue

        text = clean_page_text(text)
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

    # Build FAISS index
    if embeddings:
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype='float32'))
        faiss.write_index(index, faiss_index)

    print(f"✅ Lưu {len(data)} chunk vào {output_json} và tạo FAISS index tại {faiss_index}")


# ------------------ Main ------------------
if __name__ == "__main__":
    pdf_to_json_and_faiss_safe(PDF_PATH, JSON_OUTPUT, FAISS_INDEX)