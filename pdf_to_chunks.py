import fitz
import json
import os
import re
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import faiss

nltk.download("punkt", quiet=True)

# ------------------ Cấu hình ------------------
COURSE_FOLDER = "Course/SGK_CS_12"
PDF_FOLDER = os.path.join(COURSE_FOLDER, "pdfs")
CHUNKS_JSON_FOLDER = os.path.join(COURSE_FOLDER, "chunks_json")
FAISS_INDEX_FOLDER = os.path.join(COURSE_FOLDER, "faiss_index")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300
OVERLAP = 50
SIM_THRESHOLD = 0.95
REMOVE_HEADER_FOOTER = True

# Tạo folder nếu chưa có
os.makedirs(CHUNKS_JSON_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)

# ------------------ Load embedding ------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMB_DIM = embed_model.get_sentence_embedding_dimension()

# ------------------ Tiền xử lý text ------------------
def clean_page_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    if REMOVE_HEADER_FOOTER:
        lines = text.split("\n")
        if len(lines) > 5:
            text = " ".join(lines[1:-1])
    return text

# ------------------ Chunk semantic + duplicate check toàn khóa ------------------
def chunk_text_semantic_safe(text, embed_model, faiss_index=None, max_tokens=MAX_TOKENS, overlap=OVERLAP, sim_threshold=SIM_THRESHOLD):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [], []

    chunks, chunks_emb = [], []
    current_chunk = []

    for sent in sentences:
        current_chunk.append(sent)
        token_count = sum(len(s.split()) for s in current_chunk)

        # Tạo chunk khi đạt max_tokens hoặc câu cuối
        if token_count >= max_tokens or sent == sentences[-1]:
            chunk_text = " ".join(current_chunk).strip()
            if not chunk_text:
                current_chunk = []
                continue

            emb = embed_model.encode(chunk_text, convert_to_numpy=True)
            faiss.normalize_L2(np.array([emb], dtype='float32'))  # normalize ngay

            # Duplicate check
            is_duplicate = False
            if faiss_index and faiss_index.ntotal > 0:
                distances, _ = faiss_index.search(np.array([emb], dtype='float32'), 1)
                if distances[0][0] < 1 - sim_threshold:
                    is_duplicate = True

            if not is_duplicate:
                chunks.append(chunk_text)
                chunks_emb.append(emb)
                # Thêm luôn vào index để duplicate check cho chunks sau
                if faiss_index is not None:
                    faiss_index.add(np.array([emb], dtype='float32'))

            # Reset chunk với phần chồng lấn
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []

    return chunks, chunks_emb

# ------------------ Xử lý toàn bộ PDF ------------------
def process_all_pdfs(pdf_folder, json_folder, faiss_index_folder):
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    all_chunks_data = []
    all_embeddings = []

    # Tạo FAISS index chung
    faiss_index = faiss.IndexFlatL2(EMB_DIM)

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        lesson_name = os.path.splitext(filename)[0]

        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if not text.strip():
                continue
            text = clean_page_text(text)
            chunks, chunks_emb = chunk_text_semantic_safe(text, embed_model, faiss_index)

            for i, (chunk, emb) in enumerate(zip(chunks, chunks_emb)):
                all_chunks_data.append({
                    "content": chunk,
                    "source": filename,
                    "page": page_num,
                    "chunk_id": f"{lesson_name}_{page_num}_{i}"
                })
                all_embeddings.append(emb)

        print(f"✅ {filename}: {len(chunks)} chunk processed.")

    # Save JSON toàn khóa học
    json_path = os.path.join(json_folder, "all_chunks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks_data, f, ensure_ascii=False, indent=2)

    # Build FAISS index chung
    if all_embeddings:
        embeddings_array = np.array(all_embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_array)
        faiss_index = faiss.IndexFlatL2(EMB_DIM)
        faiss_index.add(embeddings_array)
        index_path = os.path.join(faiss_index_folder, "all_lessons.index")
        faiss.write_index(faiss_index, index_path)
        print(f"✅ FAISS index chung tạo xong: {index_path}, tổng {len(all_chunks_data)} chunk")
    else:
        print("⚠️ Không tìm thấy chunk nào để build index.")

# ------------------ Main ------------------
if __name__ == "__main__":
    process_all_pdfs(PDF_FOLDER, CHUNKS_JSON_FOLDER, FAISS_INDEX_FOLDER)