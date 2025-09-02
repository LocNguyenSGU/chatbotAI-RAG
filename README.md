# ChatbotAI-RAG

Chatbot thông minh sử dụng **Retrieval-Augmented Generation (RAG)** để trả lời câu hỏi từ PDF và cơ sở dữ liệu SQL.

---

## 🚀 Tính năng

- Tạo embedding từ PDF bằng **SentenceTransformer**, lưu trong **FAISS**.  
- Trả lời tự nhiên với **Google Gemini/OpenAI API**.  
- Chia chunk semantic, loại bỏ trùng lặp.  
- Tích hợp **LangChain SQLDatabaseChain** cho truy vấn MySQL.  
- **Flask API** với endpoint `/ask` và `/similar`.

---

## 🛠 Cài đặt

```bash
# Clone dự án
git clone <repository_url>
cd ChatbotAI-RAG

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```
## 📂 Cấu trúc mã

- **loader.py**: Đọc PDF/DOCX/DB, trả văn bản hoặc danh sách chunk.  
- **chunker.py**: Chia văn bản thành chunk, loại bỏ trùng lặp, trả về embedding.  
- **embedder.py**: Mã hóa văn bản bằng SentenceTransformer, chuẩn hóa embedding.  
- **indexer.py**: Xây dựng hoặc thêm chỉ mục FAISS cho các chunk.  
- **search.py**: Tìm kiếm top-k chunk bằng FAISS + fuzzy matching.  
- **api.py**: Flask API, cung cấp endpoint `/ask` và `/similar`.  
- **main.py**: Pipeline offline: đọc PDF → chia chunk → tạo FAISS index → lưu JSON.  