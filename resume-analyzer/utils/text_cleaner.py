"""Text cleaning helpers for NLP preprocessing."""

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are missing."""
    resources = {"punkt": "tokenizers/punkt", "stopwords": "corpora/stopwords"}
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_text(text: str) -> str:
    """Clean and normalize text for modeling."""
    if not text:
        return ""

    ensure_nltk_resources()

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens: List[str] = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]

    return " ".join(tokens)
