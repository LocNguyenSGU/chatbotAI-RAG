import re
import faiss
import json
import requests
import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import google.generativeai as palm

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ Cấu hình ------------------
API_KEY = "AIzaSyAp0SgYHRHiIcdXXDvqnAupzXfXiLBpLwg"
palm.configure(api_key=API_KEY)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

logging.info("🚀 Load embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

logging.info("📂 Load FAISS index...")
index = faiss.read_index("pdf_safe.index")

try:
    with open("pdf_chunks_semantic_safe.json", "r", encoding="utf-8") as f:
        raw_metadata = json.load(f)
except FileNotFoundError:
    logging.error("Không tìm thấy file 'metadata.json'.")
    exit()

# ------------------ Clean metadata ------------------
def clean_text(text):
    text = re.sub(r'\(Ký và ghi rõ họ tên\)', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^\w\s.,;:\-–]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

metadata = [{**m, "content": clean_text(m["content"])} for m in raw_metadata]

# ------------------ Search FAISS ------------------
def search(query, top_k=TOP_K, score_threshold=0.75):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    results = []
    seen = set()
    for idx, dist in zip(indices[0], distances[0]):
        if dist < score_threshold:
            continue
        snippet = metadata[idx]["content"][:300]
        if snippet in seen:
            continue
        seen.add(snippet)
        r = metadata[idx].copy()
        r["score"] = float(dist)
        results.append(r)
    # Log top_k selected
    logging.info(f"Top {top_k} chunks selected for query '{query}':")
    for r in results:
        logging.info(f"Page {r['page']} | Score {r['score']:.4f} | Chunk ID {r['chunk_id']}")
    return results

# ------------------ Gọi Gemini ------------------
def ask_gemini(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except (KeyError, IndexError):
        logging.warning("Không có kết quả từ Gemini.")
        return "Không có kết quả từ Gemini."
    except requests.exceptions.RequestException as e:
        logging.exception(f"Lỗi khi gọi API Gemini: {e}")
        return f"Lỗi khi gọi API Gemini: {e}"

# ------------------ RAG + Gemini ------------------
def rag_answer(query, top_k=TOP_K, max_words=1500):
    results = search(query, top_k=top_k)
    context_full = "\n".join([r["content"] for r in results])
    combined_text = f"""
Bạn là trợ lý AI. Trả lời ngắn gọn, chỉ dựa vào nội dung tài liệu dưới đây. 
Không thêm thông tin ngoài tài liệu.

Tài liệu:
{context_full}

Câu hỏi:
{query}

Trả lời:
"""
    words = combined_text.split()
    prompt_limited = " ".join(words[:max_words])
    answer = ask_gemini(prompt_limited)
    return answer, results

# ------------------ Flask API ------------------
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400

        answer, results = rag_answer(query)

        context_used = [
            {"page": r["page"], "score": r["score"], "snippet": r["content"][:500]} for r in results
        ]

        return jsonify({
            "query": query,
            "answer": answer,
            "context_used": context_used
        })

    except Exception as e:
        logging.exception("Error in /ask endpoint")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logging.info("🚀 Starting Flask server on port 5001")
    app.run(host="0.0.0.0", port=5001)