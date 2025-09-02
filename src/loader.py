import fitz
import os
from utils import clean_text

def load_pdfs(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    pdf_texts = {}
    for f in pdf_files:
        path = os.path.join(pdf_folder, f)
        doc = fitz.open(path)
        text_all = []
        for page in doc:
            text_all.append(clean_text(page.get_text("text")))
        pdf_texts[f] = "\n".join(text_all)
    return pdf_texts