"""Streamlit application for AI Resume Analyzer with job matching."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
import streamlit as st

from utils.ats_score import calculate_ats_score
from utils.job_matcher import recommend_jobs
from utils.resume_parser import extract_resume_text
from utils.skill_extractor import extract_skills, get_skill_categories
from utils.text_cleaner import clean_text, ensure_nltk_resources


APP_DIR = Path(__file__).resolve().parent
BANNER_PATH = APP_DIR / "assets" / "banner.png"


def configure_page() -> None:
    st.set_page_config(
        page_title="AI Resume Analyzer",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');
        :root {
            --accent: #1f6feb;
            --accent-2: #14b8a6;
            --text: #0f172a;
            --muted: #475569;
            --panel: #ffffff;
        }
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
        }
        .stApp {
            background: radial-gradient(circle at 10% 10%, #e0f2fe 0%, #f8fafc 50%, #eef2ff 100%);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        .block-container {
            padding-top: 1.5rem;
        }
        .metric-card {
            background: var(--panel);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.3);
        }
        .tag {
            display: inline-block;
            padding: 0.3rem 0.6rem;
            margin: 0.2rem;
            background: #e2e8f0;
            color: #1e293b;
            border-radius: 999px;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    if BANNER_PATH.exists():
        st.image(str(BANNER_PATH), use_column_width=True)

    st.title("AI Resume Analyzer with Job Matching Using NLP")
    st.write(
        "Upload your resume, analyze skills, compare against job descriptions, and discover "
        "the best matching roles using NLP-driven insights."
    )


def render_sidebar() -> tuple:
    st.sidebar.title("AI Resume Analyzer")
    st.sidebar.caption("Upload a resume and paste a job description to start the analysis.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload resume (PDF or DOCX)", type=["pdf", "docx"]
    )
    job_description = st.sidebar.text_area("Job description", height=220)

    st.sidebar.markdown("---")
    st.sidebar.caption("The ATS score reflects similarity between the resume and job description.")

    return uploaded_file, job_description


def create_ats_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f6feb"},
                "steps": [
                    {"range": [0, 40], "color": "#fecaca"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#bbf7d0"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320)
    return fig


def create_skills_chart(skills: list) -> go.Figure:
    if not skills:
        return go.Figure()

    df = pd.Series(skills).value_counts().head(15).reset_index()
    df.columns = ["skill", "count"]
    fig = px.bar(df, x="count", y="skill", orientation="h", color="count")
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=360)
    fig.update_coloraxes(showscale=False)
    return fig


def create_job_match_chart(jobs: pd.DataFrame) -> go.Figure:
    if jobs.empty:
        return go.Figure()

    fig = px.bar(jobs, x="similarity_score", y="job_title", orientation="h", color="similarity_score")
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=360)
    fig.update_coloraxes(showscale=False)
    return fig


def render_tags(items: list, empty_message: str) -> None:
    if not items:
        st.write(empty_message)
        return

    html = "".join([f"<span class='tag'>{item}</span>" for item in items])
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    configure_page()

    if "nltk_ready" not in st.session_state:
        ensure_nltk_resources()
        st.session_state["nltk_ready"] = True

    uploaded_file, job_description = render_sidebar()
    render_header()

    resume_text = ""
    error_message = None
    if uploaded_file:
        resume_text, error_message = extract_resume_text(uploaded_file)

    if error_message:
        st.error(error_message)
        st.stop()

    if not resume_text:
        st.info("Upload a resume to see analytics and recommendations.")
        return

    with st.expander("Resume preview", expanded=False):
        st.write(resume_text)

    skills = extract_skills(resume_text)
    job_recommendations = recommend_jobs(resume_text)

    ats_score = 0.0
    matching_keywords = []
    missing_keywords = []

    if job_description and job_description.strip():
        ats_score, matching_keywords, missing_keywords = calculate_ats_score(
            resume_text, job_description
        )
    else:
        st.warning("Add a job description to compute ATS score and missing skills.")

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp(resume_text)

    cleaned_resume = clean_text(resume_text)
    word_count = len(cleaned_resume.split())
    sentence_count = len(list(doc.sents))
    skill_count = len(skills)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Words", word_count)
    with metric_col2:
        st.metric("Sentences", sentence_count)
    with metric_col3:
        st.metric("Skills detected", skill_count)
    with metric_col4:
        st.metric("ATS score", f"{ats_score}%")

    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        st.subheader("Extracted skills")
        render_tags(skills, "No skills detected yet.")

        st.subheader("Missing skills")
        render_tags(missing_keywords[:30], "No missing skills calculated.")

        st.subheader("Matching keywords")
        render_tags(matching_keywords[:30], "No matching keywords calculated.")

    with right_col:
        st.subheader("ATS score")
        st.plotly_chart(create_ats_gauge(ats_score), use_container_width=True)

    charts_col1, charts_col2 = st.columns(2)
    with charts_col1:
        st.subheader("Skill distribution")
        st.plotly_chart(create_skills_chart(skills), use_container_width=True)

    with charts_col2:
        st.subheader("Job match scores")
        st.plotly_chart(create_job_match_chart(job_recommendations), use_container_width=True)

    st.subheader("Recommended jobs")
    if job_recommendations.empty:
        st.write("No job recommendations available.")
    else:
        st.dataframe(job_recommendations.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
