# 🎯 AI-Based Interview Performance Evaluation System

An intelligent, NLP-powered interview coach that evaluates candidate answers using semantic similarity, keyword analysis, and linguistic quality metrics — built entirely from scratch using Python.

---

## 📌 Problem Statement

Most freshers don't get feedback on how well they perform in mock interviews. This system analyzes their answers to interview questions and provides detailed, actionable scores and feedback — helping them prepare more effectively.

---

## 🚀 Features

- **Resume Skill Extractor** — Automatically extracts technical skills from a PDF resume using NLP keyword matching with word boundary detection
- **Personalized Question Generator** — Generates interview questions tailored to the candidate's skill set using a curated Q&A dataset (102 questions across 8 domains)
- **AI-Powered Answer Evaluation** — Evaluates answer relevance using Sentence Transformers (all-MiniLM-L6-v2) and cosine similarity
- **Completeness Scoring** — Checks coverage of key technical concepts using keyword analysis
- **Filler Word Detection** — Detects and counts filler words (um, like, basically, etc.) using regex pattern matching
- **Spelling Analysis** — Identifies spelling errors using pyspellchecker with a custom technical vocabulary
- **Weighted Overall Score** — Combines all metrics into a single score (Relevance 40%, Completeness 35%, Spelling 15%, Filler Penalty 10%)
- **Rule-Based Feedback Generator** — Produces specific, actionable feedback for each answer
- **Session Report** — Generates a final performance summary saved as JSON
- **Visualizations** — 4 matplotlib charts showing score breakdown, trends, summary, and filler word usage

---

## 🧠 ML/NLP Concepts Used

| Concept | Where Used |
|---|---|
| Sentence Transformers (BERT-based) | Answer relevance scoring |
| Cosine Similarity | Comparing answer embeddings |
| Keyword Extraction | Resume skill extraction |
| Regex (word boundaries) | Skill matching, filler word detection |
| Weighted Scoring | Overall score calculation |
| Rule-based NLP | Feedback generation |
| pyspellchecker | Spelling error detection |

---

## 🗂️ Project Structure
AI Powered Project/

├── data/

│   └── qa_dataset.json          # 102 curated Q&A pairs across 8 skills

├── notebooks/

│   ├── 02_resume_extractor.py   # Module 1: PDF parsing + skill extraction

│   ├── 03_question_generator.py # Module 2: Personalized question generation

│   ├── 04_answer_evaluator.py   # Module 3: Relevance + completeness scoring

│   ├── 05_grammar_filler.py     # Module 4: Spelling + filler word analysis

│   ├── 06_combined_evaluator.py # Module 5: Combined scoring + feedback

│   ├── 07_interview_session.py  # Module 6: Full interactive session

│   └── 08_visualizer.py         # Module 7: Charts and visualizations

├── resumes/

│   └── sample_resume.pdf        # Sample resume for testing

├── outputs/

│   ├── session_results.json     # Saved session results

│   ├── chart_score_breakdown.png

│   ├── chart_score_trend.png

│   ├── chart_summary.png

│   └── chart_filler_words.png

└── README.md

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
pip install pandas numpy matplotlib pypdf sentence-transformers pyspellchecker
```

### Run the interactive interview session

```bash
python notebooks/07_interview_session.py
```

### Generate visualizations (after a session)

```bash
python notebooks/08_visualizer.py
```

---

## 📊 Sample Output

QUESTION: What is the difference between a list and a tuple in Python?
OVERALL SCORE: 69.11/100
Relevance:    79.03/100

Completeness: 40.0/100

Spelling:     100/100

Filler words: 3
FEEDBACK:

Your answer is highly relevant to the question.
You covered some key points, but missed: mutable, immutable, memory.
You used 3 filler words (um, like, basically). Practice speaking concisely.
No spelling issues detected.

---

## 📈 Visualizations

Charts are automatically saved to the `outputs/` folder after each session:

| Chart | Description |
|---|---|
| `chart_score_breakdown.png` | Score breakdown per question (grouped bar chart) |
| `chart_score_trend.png` | Overall score trend across the session |
| `chart_summary.png` | Average performance summary |
| `chart_filler_words.png` | Filler word usage per question |

---

## 🔮 Future Improvements

- Add voice input support using Whisper (OpenAI) for spoken answers
- Integrate a Large Language Model (LLM) for richer, context-aware feedback
- Build a web interface using Streamlit or Flask
- Expand dataset to 500+ questions across more domains
- Add confidence scoring using answer length and vocabulary richness

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `sentence-transformers` | Semantic similarity using BERT-based model |
| `pypdf` | PDF text extraction |
| `pyspellchecker` | Spelling error detection |
| `pandas` | Data loading and handling |
| `numpy` | Numerical operations |
| `matplotlib` | Charts and visualizations |
| `re` (built-in) | Regex for skill matching and filler detection |
| `json` (built-in) | Dataset loading and result saving |

---

## 👤 Author

**Darshan Kulkarni**
Information Science Engineer | Data Science & ML Enthusiast
[LinkedIn](https://linkedin.com/in/darshan-kulkarni-ba85a92b8) | [GitHub](https://github.com/darshankulkarni8999)