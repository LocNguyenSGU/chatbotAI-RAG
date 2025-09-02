import os

COURSE_FOLDER = "../Course/SGK_CS_12"
PDF_FOLDER = os.path.join(COURSE_FOLDER, "pdfs")
CHUNKS_JSON_FOLDER = os.path.join(COURSE_FOLDER, "chunks_json")
FAISS_INDEX_FOLDER = os.path.join(COURSE_FOLDER, "faiss_index")
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_FOLDER, "all_lessons.index")
CHUNKS_JSON_PATH = os.path.join(CHUNKS_JSON_FOLDER, "all_chunks.json")
API_KEY="AIzaSyAp0SgYHRHiIcdXXDvqnAupzXfXiLBpLwg"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300
OVERLAP = 50
SIM_THRESHOLD = 0.95
REMOVE_HEADER_FOOTER = True

TOP_K = 10
SCORE_THRESHOLD = 0.5
KEYWORD_BOOST = 0.5
EXACT_MATCH_BOOST = 2.0
MAX_PROMPT_WORDS = 5000