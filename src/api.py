from flask import Flask, request, jsonify
from search import SemanticSearcher
from embedder import Embedder
from config import *
import json
import faiss
import requests

# ------------------ Load embedding ------------------
embedder = Embedder(EMBED_MODEL_NAME)

# ------------------ Load metadata ------------------
with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ------------------ Load FAISS index ------------------
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# ------------------ Init searcher ------------------
searcher = SemanticSearcher(embedder, faiss_index, metadata, top_k=TOP_K, score_threshold=SCORE_THRESHOLD)

# ------------------ Flask app ------------------
app = Flask(__name__)

# ------------------ Similar endpoint ------------------
@app.route("/similar", methods=["POST"])
def similar():
    data = request.json
    content = data.get("content", "").strip()
    top_k = int(data.get("top_k", TOP_K))
    if not content:
        return jsonify({"error": "Missing 'content'"}), 400

    results = searcher.search(content, top_k=top_k)
    return jsonify({"query_content": content, "results": results})

# ------------------ Gemini API ------------------
def ask_gemini(prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": API_KEY}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Lỗi khi gọi Gemini: {e}"

# ------------------ Ask endpoint ------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    # 1. Search top-k chunks
    chunks = searcher.search(query, top_k=TOP_K)

    # 2. Build prompt
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
    # 3. Call Gemini
    answer = ask_gemini(prompt)

    # 4. Return result
    context_used = [{"page": c["page"], "score": c["score"], "snippet": c["content"][:500]} for c in chunks]
    return jsonify({"query": query, "answer": answer, "context_used": context_used})

# ------------------ Run Flask ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)