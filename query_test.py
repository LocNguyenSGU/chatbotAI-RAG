import os
import re
import json
import faiss
import logging
import requests
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------ Cấu hình ------------------
API_KEY = "AIzaSyAp0SgYHRHiIcdXXDvqnAupzXfXiLBpLwg"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 15
SCORE_THRESHOLD = 0.5
KEYWORD_BOOST = 0.5
EXACT_MATCH_BOOST = 2.0
MAX_PROMPT_WORDS = 5000

COURSE_FOLDER = "Course/SGK_CS_12"
CHUNKS_JSON_FOLDER = os.path.join(COURSE_FOLDER, "chunks_json")
FAISS_INDEX_FOLDER = os.path.join(COURSE_FOLDER, "faiss_index")
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_FOLDER, "all_lessons.index")
CHUNKS_JSON_PATH = os.path.join(CHUNKS_JSON_FOLDER, "all_chunks.json")

# ------------------ Load embedding ------------------
logging.info("🚀 Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ------------------ Load JSON + FAISS index chung ------------------
logging.info("📂 Loading chunks JSON and FAISS index...")

with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    all_metadata = json.load(f)
logging.info(f"✅ Loaded JSON chunks: {len(all_metadata)} chunks")

# Clean text
def clean_text(text: str) -> str:
    text = re.sub(r'\(Ký và ghi rõ họ tên\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

metadata = [{**m, "content": clean_text(m["content"])} for m in all_metadata]

# Load FAISS index
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
logging.info(f"✅ Loaded FAISS index: {FAISS_INDEX_PATH}, total vectors: {faiss_index.ntotal}")

# ------------------ Fuzzy match ------------------
def fuzzy_score(query: str, text: str) -> float:
    return SequenceMatcher(None, query.lower(), text.lower()).ratio()

# ------------------ Search FAISS + Fuzzy ------------------
def search(query: str, top_k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD):
    query_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    all_results = []
    seen = set()

    keywords = [kw.strip().lower() for kw in re.split(r'\W+', query) if kw.strip()]
    query_lower = query.lower()

    distances, indices = faiss_index.search(query_emb, top_k * 5)

    for i, sim in zip(indices[0], distances[0]):
        if i < 0:
            continue

        chunk_metadata = metadata[i]
        chunk = chunk_metadata["content"]
        key = (chunk_metadata['source'], chunk_metadata['page'], chunk_metadata['chunk_id'])
        if key in seen:
            continue
        seen.add(key)

        # 1. Semantic score
        semantic_score = float(sim)

        # 2. Exact match boost
        exact_match_score = EXACT_MATCH_BOOST if query_lower in chunk.lower() else 0

        # 3. Keyword boost
        keyword_score = sum(KEYWORD_BOOST for kw in set(keywords) if kw in chunk.lower())

        # 4. Fuzzy score
        fuzzy_val = fuzzy_score(query_lower, chunk.lower())

        final_score = semantic_score + exact_match_score + keyword_score + fuzzy_val

        if final_score >= score_threshold:
            r = chunk_metadata.copy()
            r["score"] = final_score
            all_results.append(r)

    final_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:top_k]

    logging.info(f"Top {top_k} chunks for query: '{query}'")
    for r in final_results:
        logging.info(f"Page {r['page']} | Score {r['score']:.4f} | Snippet: {r['content'][:80]}...")

    return final_results

# ------------------ Gọi Gemini ------------------
def ask_gemini(prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": API_KEY}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    logging.info(f"📤 Sending prompt to Gemini ({len(prompt.split())} words)...")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logging.exception("Error calling Gemini API")
        return f"Lỗi: {e}"

# ------------------ RAG + Gemini ------------------
def rag_answer(query: str, top_k: int = TOP_K, max_words: int = MAX_PROMPT_WORDS):
    chunks = search(query, top_k=top_k)
    context = "\n".join([c["content"] for c in chunks])
    prompt = f"""
        Bạn là trợ lý AI. Trả lời đầy đủ, chỉ dựa vào nội dung tài liệu dưới đây.
        Không thêm thông tin ngoài tài liệu, văn phong như gia sư hỗ trợ học tập.
        
        Tài liệu:
        {context}
        
        Câu hỏi:
        {query}
        
        Trả lời:
        """
    prompt_limited = " ".join(prompt.split()[:max_words])
    answer = ask_gemini(prompt_limited)
    return answer, chunks

# ------------------ Flask API ------------------
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    answer, chunks = rag_answer(query)
    context_used = [{"page": c["page"], "score": c["score"], "snippet": c["content"][:500]} for c in chunks]

    return jsonify({
        "query": query,
        "answer": answer,
        "context_used": context_used
    })

# ------------------ Endpoint tìm chunks tương tự ------------------
@app.route("/similar", methods=["POST"])
def similar():
    data = request.json
    content = data.get("content", "").strip()
    top_k = int(data.get("top_k", 5))  # default 5

    if not content:
        return jsonify({"error": "Missing 'content'"}), 400

    # Embedding cho chunk / câu hỏi mẫu
    query_emb = embed_model.encode([content], convert_to_numpy=True, normalize_embeddings=True)

    # Search FAISS index
    distances, indices = faiss_index.search(query_emb, top_k * 5)

    results = []
    seen = set()
    for i, dist in zip(indices[0], distances[0]):
        if i < 0:
            continue
        chunk_metadata = metadata[i]
        key = (chunk_metadata['source'], chunk_metadata['page'], chunk_metadata['chunk_id'])
        if key in seen:
            continue
        seen.add(key)

        # Fuzzy score + semantic score
        fuzzy_val = SequenceMatcher(None, content.lower(), chunk_metadata['content'].lower()).ratio()
        semantic_score = float(dist)
        final_score = semantic_score + fuzzy_val  # có thể cân nhắc thêm keyword boost nếu muốn

        results.append({
            "source": chunk_metadata['source'],
            "page": chunk_metadata['page'],
            "chunk_id": chunk_metadata['chunk_id'],
            "content": chunk_metadata['content'],
            "score": final_score
        })

    # Sort theo score giảm dần và lấy top_k
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    return jsonify({
        "query_content": content,
        "top_k": top_k,
        "results": results
    })

if __name__ == "__main__":
    logging.info("🚀 Starting Flask server on port 5001")
    app.run(host="0.0.0.0", port=5001)