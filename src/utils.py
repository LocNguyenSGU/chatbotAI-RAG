import re
import nltk
from difflib import SequenceMatcher

nltk.download("punkt", quiet=True)

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fuzzy_score(query: str, text: str) -> float:
    return SequenceMatcher(None, query.lower(), text.lower()).ratio()

def tokenize_sentences(text: str):
    return nltk.sent_tokenize(text)