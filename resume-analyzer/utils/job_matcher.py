"""Job matching utilities for recommending roles."""

from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_cleaner import clean_text


BASE_DIR = Path(__file__).resolve().parents[1]
JOBS_PATH = BASE_DIR / "data" / "sample_jobs.csv"


@lru_cache(maxsize=1)
def load_jobs() -> pd.DataFrame:
    """Load sample jobs from CSV."""
    df = pd.read_csv(JOBS_PATH)
    df["job_title"] = df["job_title"].astype(str).str.strip()
    df["job_description"] = df["job_description"].astype(str).str.strip()
    return df


def recommend_jobs(resume_text: str, top_n: int = 5) -> pd.DataFrame:
    """Recommend top jobs based on similarity to resume text."""
    if not resume_text:
        return pd.DataFrame(columns=["job_title", "similarity_score"])

    jobs = load_jobs()
    if jobs.empty:
        return pd.DataFrame(columns=["job_title", "similarity_score"])

    cleaned_resume = clean_text(resume_text)
    cleaned_jobs = jobs["job_description"].apply(clean_text).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume] + cleaned_jobs)

    resume_vector = tfidf_matrix[0:1]
    job_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(resume_vector, job_vectors).ravel()
    jobs = jobs.copy()
    jobs["similarity_score"] = (similarities * 100).round(2)

    return jobs.sort_values("similarity_score", ascending=False).head(top_n)
