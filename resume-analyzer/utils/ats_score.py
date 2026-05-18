"""ATS scoring utilities based on cosine similarity."""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_cleaner import clean_text


def calculate_ats_score(resume_text: str, job_description: str) -> Tuple[float, List[str], List[str]]:
    """Calculate ATS score and keyword overlap between resume and job description."""
    if not resume_text or not job_description:
        return 0.0, [], []

    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_description)

    if not cleaned_resume or not cleaned_job:
        return 0.0, [], []

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    ats_score = round(similarity * 100, 2)

    feature_names = vectorizer.get_feature_names_out()
    resume_vector = tfidf_matrix[0].toarray().ravel()
    job_vector = tfidf_matrix[1].toarray().ravel()

    matching = []
    missing = []
    for idx, term in enumerate(feature_names):
        if job_vector[idx] > 0 and resume_vector[idx] > 0:
            matching.append((term, job_vector[idx]))
        elif job_vector[idx] > 0 and resume_vector[idx] == 0:
            missing.append((term, job_vector[idx]))

    matching_keywords = [term for term, _ in sorted(matching, key=lambda x: x[1], reverse=True)]
    missing_keywords = [term for term, _ in sorted(missing, key=lambda x: x[1], reverse=True)]

    return ats_score, matching_keywords, missing_keywords
