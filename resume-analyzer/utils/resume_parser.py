"""Resume parsing utilities for PDF and DOCX files."""

from pathlib import Path
from typing import Optional, Tuple

import pdfplumber
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def extract_text_from_pdf(file_obj) -> Tuple[str, Optional[str]]:
    """Extract text from a PDF file-like object.

    Returns a tuple of (text, error_message).
    """
    if file_obj is None:
        return "", "No file provided."

    try:
        with pdfplumber.open(file_obj) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:  # pragma: no cover - runtime errors
        return "", f"PDF parsing failed: {exc}"

    text = "\n".join(pages).strip()
    if not text:
        return "", "The PDF appears to be empty or scanned."

    return text, None


def extract_text_from_docx(file_obj) -> Tuple[str, Optional[str]]:
    """Extract text from a DOCX file-like object.

    Returns a tuple of (text, error_message).
    """
    if file_obj is None:
        return "", "No file provided."

    try:
        document = Document(file_obj)
        paragraphs = [para.text for para in document.paragraphs]
    except Exception as exc:  # pragma: no cover - runtime errors
        return "", f"DOCX parsing failed: {exc}"

    text = "\n".join(paragraphs).strip()
    if not text:
        return "", "The DOCX appears to be empty."

    return text, None


def extract_resume_text(uploaded_file) -> Tuple[str, Optional[str]]:
    """Extract resume text based on the uploaded file type.

    Returns a tuple of (text, error_message).
    """
    if uploaded_file is None:
        return "", "No file uploaded."

    file_name = getattr(uploaded_file, "name", "")
    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return "", "Unsupported file type. Please upload a PDF or DOCX."

    if extension == ".pdf":
        return extract_text_from_pdf(uploaded_file)

    return extract_text_from_docx(uploaded_file)
