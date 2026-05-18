"""Skill extraction utilities using a curated skills list."""

from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher


BASE_DIR = Path(__file__).resolve().parents[1]
SKILLS_PATH = BASE_DIR / "data" / "skills.csv"


@lru_cache(maxsize=1)
def load_skills() -> pd.DataFrame:
    """Load skills from the CSV file."""
    df = pd.read_csv(SKILLS_PATH)
    df["skill"] = df["skill"].astype(str).str.strip()
    df = df[df["skill"] != ""]
    return df


@lru_cache(maxsize=1)
def _get_matcher():
    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    skills = load_skills()["skill"].tolist()
    patterns = [nlp.make_doc(skill) for skill in skills]
    matcher.add("SKILLS", patterns)
    return nlp, matcher


def extract_skills(text: str) -> List[str]:
    """Extract unique skills from text using phrase matching."""
    if not text:
        return []

    nlp, matcher = _get_matcher()
    doc = nlp(text)
    matches = matcher(doc)

    found_skills = set()
    for _, start, end in matches:
        found_skills.add(doc[start:end].text.strip())

    skills_lower = {skill.lower() for skill in found_skills}
    canonical = [
        skill
        for skill in load_skills()["skill"].tolist()
        if skill.lower() in skills_lower
    ]

    return sorted(set(canonical))


def get_skill_categories(skills: List[str]) -> pd.DataFrame:
    """Return counts by category for the provided skills."""
    if not skills:
        return pd.DataFrame(columns=["category", "count"])

    df = load_skills()
    filtered = df[df["skill"].isin(skills)]
    counts = filtered.groupby("category", as_index=False).size()
    counts.columns = ["category", "count"]
    return counts
