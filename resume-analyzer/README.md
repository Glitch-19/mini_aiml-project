# AI Resume Analyzer with Job Matching Using NLP

## Project overview
AI Resume Analyzer is a Streamlit application that parses resumes, extracts skills, computes ATS similarity scores against a job description, and recommends relevant roles using NLP and TF-IDF matching.

## Features
- Upload PDF or DOCX resumes
- Resume text extraction with error handling
- NLP cleaning with tokenization, stopword removal, and regex normalization
- Skill extraction using a curated skills library
- ATS score based on cosine similarity
- Missing and matching keyword insights
- Job recommendations with similarity scores
- Interactive Plotly dashboards and charts

## Folder structure
```
resume-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── skills.csv
│   └── sample_jobs.csv
├── utils/
│   ├── resume_parser.py
│   ├── text_cleaner.py
│   ├── skill_extractor.py
│   ├── ats_score.py
│   └── job_matcher.py
├── assets/
│   └── banner.png
├── uploads/
└── models/
```

## Installation
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the application
```bash
streamlit run app.py
```

## Screenshots
Capture screenshots of the dashboard after running the app and add them here.

## Technologies used
- Streamlit
- pandas, numpy, scikit-learn
- nltk, spacy
- pdfplumber, python-docx
- plotly, matplotlib

## Future improvements
- Add semantic similarity using embeddings
- Add resume section detection
- Export ATS reports as PDF
- Add multi-role comparison
