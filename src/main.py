from loader import load_pdfs
from indexer import build_faiss_index
from embedder import Embedder
from config import *

if __name__ == "__main__":
    embedder = Embedder(EMBED_MODEL_NAME)
    pdf_texts = load_pdfs(PDF_FOLDER)
    build_faiss_index(pdf_texts, embedder, FAISS_INDEX_PATH, CHUNKS_JSON_PATH,
                      max_tokens=MAX_TOKENS, overlap=OVERLAP, sim_threshold=SIM_THRESHOLD)