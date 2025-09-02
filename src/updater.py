import os
import json
import fitz
import re
import numpy as np
import faiss
from embedder import Embedder
from config import *

# Load embedding và FAISS index hiện tại
embedder = Embedder(EMBED_MODEL_NAME)

# Load metadata cũ
if os.path.exists(CHUNKS_JSON_PATH):
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
else:
    all_chunks = []

# Load FAISS index cũ hoặc tạo mới nếu chưa có
if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
else:
    faiss_index = faiss.IndexFlatL2(embedder.dim)

def clean_page_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    lines = text.split("\n")
    if len(lines) > 5:
        text = " ".join(lines[1:-1])
    return text

def chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    import nltk
    nltk.download("punkt", quiet=True)
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    for sent in sentences:
        current_chunk.append(sent)
        token_count = sum(len(s.split()) for s in current_chunk)
        if token_count >= max_tokens or sent == sentences[-1]:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
    return chunks

def add_new_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    lesson_name = os.path.splitext(filename)[0]

    doc = fitz.open(pdf_path)
    new_chunks = []
    new_embeddings = []

    existing_ids = {chunk['chunk_id'] for chunk in all_chunks}

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        text = clean_page_text(text)
        chunks = chunk_text(text)

        for i, chunk_text_str in enumerate(chunks):
            chunk_id = f"{lesson_name}_{page_num}_{i}"
            if chunk_id in existing_ids:
                continue  # đã tồn tại
            emb = embedder.encode([chunk_text_str])[0]
            new_chunks.append({
                "content": chunk_text_str,
                "source": filename,
                "page": page_num,
                "chunk_id": chunk_id
            })
            new_embeddings.append(emb)

    if new_chunks:
        # Append vào metadata và FAISS index
        all_chunks.extend(new_chunks)
        faiss_index.add(np.array(new_embeddings, dtype='float32'))

        # Lưu lại
        with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    print(f"✅ PDF '{filename}' processed. New chunks: {len(new_chunks)}")