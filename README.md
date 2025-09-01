# ChatbotAI-RAG

**ChatbotAI-RAG** là dự án chatbot thông minh sử dụng **Retrieval-Augmented Generation (RAG)**, giúp trả lời câu hỏi dựa trên tài liệu PDF.

### Tính năng chính
- Tạo embedding cho từng đoạn văn bằng **SentenceTransformer**.
- Lưu trữ embedding trong **FAISS** để tìm kiếm thông tin liên quan nhanh chóng.
- Tích hợp **Google Gemini API** để tạo câu trả lời tự nhiên, chính xác và ngắn gọn dựa trên ngữ cảnh.
- Xử lý PDF thành các chunk semantic, loại bỏ nội dung trùng lặp, đảm bảo hiệu quả truy xuất.
- Cung cấp **Flask API** với endpoint `/ask` để nhận truy vấn từ người dùng và trả về câu trả lời kèm ngữ cảnh liên quan.

### Cài đặt
```bash
# Clone dự án
git clone <repository_url>
cd ChatbotAI-RAG

# Tạo môi trường ảo (Python 3.12+)
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt