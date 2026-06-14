import os
import json

# ---- STEP 1: Load the Q&A dataset ----
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, '..', 'data', 'qa_dataset.json')

with open(dataset_path, 'r') as f:
    qa_data = json.load(f)

print(f"Total questions in dataset: {len(qa_data)}")


# ---- STEP 2: Skills extracted from resume (from Module 2) ----
# Hardcoding for now - later we'll connect Module 2's output directly
extracted_skills = ['Python', 'Java', 'SQL', 'Machine Learning', 'Deep Learning',
                     'NumPy', 'Pandas', 'Scikit-learn', 'HTML', 'CSS', 'JavaScript', 'GitHub']


# ---- STEP 3: Map resume skills to dataset skill categories ----
# Our dataset has specific categories: Python, Java, SQL, DBMS,
# Machine Learning, Data Structures & Algorithms, Cloud, OOP
# Resume skills like "NumPy", "Pandas" should map to "Python" or "Machine Learning"
# We define a mapping so unrelated resume skills (HTML, CSS, GitHub) don't cause issues

SKILL_MAPPING = {
    "Python": "Python",
    "Java": "Java",
    "SQL": "SQL",
    "Machine Learning": "Machine Learning",
    "Deep Learning": "Machine Learning",
    "NumPy": "Python",
    "Pandas": "Python",
    "Scikit-learn": "Machine Learning",
    "AWS": "Cloud",
    "Azure": "Cloud",
    "Cloud": "Cloud",
    "OOP": "OOP",
    "Data Structures": "Data Structures & Algorithms",
    "Algorithms": "Data Structures & Algorithms",
    "DBMS": "DBMS",
}


def map_skills_to_categories(skills, mapping):
    """
    Converts resume skills (like 'NumPy') into dataset categories
    (like 'Python'). Skills not in the mapping (e.g., 'HTML', 'GitHub')
    are ignored since we don't have questions for them.
    Returns a set (no duplicates) of valid categories.
    """
    categories = set()
    for skill in skills:
        if skill in mapping:
            categories.add(mapping[skill])
    return categories


matched_categories = map_skills_to_categories(extracted_skills, SKILL_MAPPING)
print(f"\nMatched question categories: {matched_categories}")


# ---- STEP 4: Pick questions for each matched category ----
import random

def generate_questions(qa_data, categories, num_per_skill=2):
    """
    For each category (e.g., 'Python'), randomly selects
    'num_per_skill' questions from the dataset belonging
    to that category.
    """
    selected_questions = []
    for category in categories:
        # Filter dataset to only this category's questions
        category_questions = [q for q in qa_data if q['skill'] == category]

        # Randomly pick num_per_skill questions (or fewer if not enough exist)
        sample_size = min(num_per_skill, len(category_questions))
        picked = random.sample(category_questions, sample_size)

        selected_questions.extend(picked)

    return selected_questions


final_questions = generate_questions(qa_data, matched_categories, num_per_skill=2)

print(f"\n---- GENERATED INTERVIEW QUESTIONS ({len(final_questions)} total) ----")
for i, q in enumerate(final_questions, 1):
    print(f"\n{i}. [{q['skill']}] {q['question']}")