import os
import re
from pypdf import PdfReader

# ---- STEP 1: Get path to resume ----
script_dir = os.path.dirname(os.path.abspath(__file__))
resume_path = os.path.join(script_dir, '..', 'resumes', 'sample_resume.pdf')


# ---- STEP 2: Extract text from PDF ----
def extract_text_from_pdf(pdf_path):
    """
    Reads a PDF file and returns all text as a single string.
    PdfReader reads page by page, so we loop through pages
    and combine the text.
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


resume_text = extract_text_from_pdf(resume_path)
print("---- RAW EXTRACTED TEXT (first 500 chars) ----")
print(resume_text[:500])


# ---- STEP 3: Define skills to look for ----
# This is our "known skills" list. We'll search resume text for these.
SKILL_KEYWORDS = [
    "Python", "Java", "SQL", "Machine Learning", "Deep Learning",
    "NumPy", "Pandas", "Scikit-learn", "TensorFlow", "PyTorch",
    "AWS", "Azure", "Cloud", "Docker", "Kubernetes",
    "OOP", "Data Structures", "Algorithms", "DBMS",
    "HTML", "CSS", "JavaScript", "React", "Flask", "Django",
    "Git", "GitHub", "Linux", "C++", "C"
]


# ---- STEP 4: Match skills against resume text (with word boundaries) ----
def extract_skills(text, skill_list):
    """
    Uses regex word boundaries (\\b) so 'C' only matches the
    standalone word 'C', not letters inside other words like 'Scikit'.

    \\b means "word boundary" - the edge between a word character
    and a non-word character (space, punctuation, start/end of string).
    """
    text_lower = text.lower()
    found_skills = []
    for skill in skill_list:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return found_skills


found = extract_skills(resume_text, SKILL_KEYWORDS)

print("\n---- SKILLS FOUND ----")
print(found)