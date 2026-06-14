import os
import re
import json
import random
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
from pypdf import PdfReader

# ============================================================
# SETUP
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, '..', 'data', 'qa_dataset.json')
resume_path = os.path.join(script_dir, '..', 'resumes', 'sample_resume.pdf')

with open(dataset_path, 'r') as f:
    qa_data = json.load(f)

print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

spell = SpellChecker()
TECHNICAL_TERMS = [
    "tuples", "tuple", "dict", "dicts", "args", "kwargs", "numpy",
    "pandas", "sklearn", "dataframe", "dataframes", "boolean",
    "iterable", "iterables", "lambda", "lambdas", "decorators",
    "decorator", "constructor", "constructors", "polymorphism",
    "encapsulation", "inheritance", "overfitting", "underfitting",
    "regex", "regexes", "json", "api", "apis", "github", "sql",
    "mysql", "postgresql", "nosql", "html", "css", "javascript",
    "frontend", "backend", "fullstack", "dbms", "oop", "knn",
    "cnn", "rnn", "tensorflow", "pytorch", "matplotlib", "seaborn",
    "datatype", "datatypes", "subclass", "subclasses", "superclass",
    "metadata", "preprocessing", "hyperparameter", "hyperparameters"
]
spell.word_frequency.load_words(TECHNICAL_TERMS)

FILLER_WORDS = [
    "um", "uh", "like", "basically", "actually", "literally",
    "you know", "i mean", "sort of", "kind of", "so yeah",
    "right", "well", "okay so"
]

SKILL_MAPPING = {
    "Python": "Python", "Java": "Java", "SQL": "SQL",
    "Machine Learning": "Machine Learning", "Deep Learning": "Machine Learning",
    "NumPy": "Python", "Pandas": "Python", "Scikit-learn": "Machine Learning",
    "AWS": "Cloud", "Azure": "Cloud", "Cloud": "Cloud",
    "OOP": "OOP", "Data Structures": "Data Structures & Algorithms",
    "Algorithms": "Data Structures & Algorithms", "DBMS": "DBMS",
}

print("Ready!\n")


# ============================================================
# MODULE 1: RESUME SKILL EXTRACTOR
# ============================================================

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def extract_skills(text, skill_list):
    text_lower = text.lower()
    found_skills = []
    for skill in skill_list:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return found_skills


def map_skills_to_categories(skills, mapping):
    categories = set()
    for skill in skills:
        if skill in mapping:
            categories.add(mapping[skill])
    return categories


# ============================================================
# MODULE 2: QUESTION GENERATOR
# ============================================================

def generate_questions(qa_data, categories, num_per_skill=2):
    selected = []
    for category in categories:
        pool = [q for q in qa_data if q['skill'] == category]
        sample_size = min(num_per_skill, len(pool))
        selected.extend(random.sample(pool, sample_size))
    return selected


# ============================================================
# MODULE 3: SCORING FUNCTIONS
# ============================================================

def calculate_relevance(candidate_answer, ideal_answer, model):
    embeddings = model.encode([candidate_answer, ideal_answer])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return max(0, round(similarity.item() * 100, 2))


def calculate_completeness(candidate_answer, key_concepts):
    answer_lower = candidate_answer.lower()
    matched = [c for c in key_concepts if c.lower() in answer_lower]
    missing = [c for c in key_concepts if c.lower() not in answer_lower]
    score = round((len(matched) / len(key_concepts)) * 100, 2)
    return score, matched, missing


def count_filler_words(text, filler_list):
    text_lower = text.lower()
    filler_counts = {}
    total = 0
    for filler in filler_list:
        if " " in filler:
            count = text_lower.count(filler)
        else:
            pattern = r'\b' + re.escape(filler) + r'\b'
            count = len(re.findall(pattern, text_lower))
        if count > 0:
            filler_counts[filler] = count
            total += count
    return total, filler_counts


def check_spelling(text, filler_list, spell_checker):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    filler_set = set(f.replace(" ", "") for f in filler_list)
    words_to_check = [w for w in words if w not in filler_set]
    misspelled = spell_checker.unknown(words_to_check)
    score = max(0, 100 - (len(misspelled) * 15))
    return score, list(misspelled)


def generate_feedback(relevance, completeness, fillers,
                       filler_breakdown, spelling, matched, missing):
    feedback = []
    if relevance >= 75:
        feedback.append("Your answer is highly relevant to the question.")
    elif relevance >= 50:
        feedback.append("Your answer is somewhat relevant but could be more focused.")
    else:
        feedback.append("Your answer doesn't closely address the question. Try again.")

    if completeness >= 80:
        feedback.append("You covered most of the important concepts. Well done!")
    elif completeness >= 40:
        feedback.append(f"You covered some key points but missed: {', '.join(missing)}.")
    else:
        feedback.append(f"Missing several important concepts: {', '.join(missing)}. Include these in your answer.")

    if fillers == 0:
        feedback.append("Excellent — no filler words used!")
    elif fillers <= 2:
        feedback.append(f"You used {fillers} filler word(s). Try to reduce these.")
    else:
        feedback.append(f"You used {fillers} filler words ({', '.join(filler_breakdown.keys())}). Practice speaking concisely.")

    if spelling == 100:
        feedback.append("No spelling issues detected.")
    else:
        feedback.append("Some spelling issues detected. Review your answer.")

    return feedback


def evaluate_answer(question_data, candidate_answer):
    relevance = calculate_relevance(candidate_answer, question_data['ideal_answer'], model)
    completeness, matched, missing = calculate_completeness(candidate_answer, question_data['key_concepts'])
    fillers, filler_breakdown = count_filler_words(candidate_answer, FILLER_WORDS)
    spelling, misspelled = check_spelling(candidate_answer, FILLER_WORDS, spell)

    filler_penalty = min(fillers * 5, 20)
    overall = round(
        relevance * 0.40 +
        completeness * 0.35 +
        spelling * 0.15 +
        max(0, 100 - filler_penalty) * 0.10,
        2
    )

    feedback = generate_feedback(relevance, completeness, fillers,
                                  filler_breakdown, spelling, matched, missing)
    return {
        "question": question_data['question'],
        "skill": question_data['skill'],
        "relevance_score": relevance,
        "completeness_score": completeness,
        "spelling_score": spelling,
        "filler_word_count": fillers,
        "overall_score": overall,
        "feedback": feedback
    }


# ============================================================
# MAIN INTERVIEW SESSION
# ============================================================

def run_interview():
    print("=" * 60)
    print("   AI INTERVIEW COACH — INTERACTIVE SESSION")
    print("=" * 60)

    # Step 1: Extract skills from resume
    print("\nReading your resume...")
    resume_text = extract_text_from_pdf(resume_path)
    SKILL_KEYWORDS = list(SKILL_MAPPING.keys())
    found_skills = extract_skills(resume_text, SKILL_KEYWORDS)
    categories = map_skills_to_categories(found_skills, SKILL_MAPPING)
    print(f"Skills detected: {found_skills}")
    print(f"Question categories: {categories}")

    # Step 2: Generate questions
    questions = generate_questions(qa_data, categories, num_per_skill=2)
    print(f"\n{len(questions)} questions generated for this session.\n")

    # Step 3: Run the interview question by question
    all_results = []
    for i, question in enumerate(questions, 1):
        print("=" * 60)
        print(f"QUESTION {i} of {len(questions)} [{question['skill']}]")
        print(f"\n{question['question']}\n")
        print("Your answer (press Enter twice when done):")

        # Collect multi-line input until user presses Enter twice
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        candidate_answer = " ".join(lines)

        if not candidate_answer.strip():
            print("No answer provided. Skipping.\n")
            continue

        # Evaluate answer
        result = evaluate_answer(question, candidate_answer)
        all_results.append(result)

        # Show immediate feedback
        print(f"\nSCORES:")
        print(f"  Relevance:    {result['relevance_score']}/100")
        print(f"  Completeness: {result['completeness_score']}/100")
        print(f"  Spelling:     {result['spelling_score']}/100")
        print(f"  Filler words: {result['filler_word_count']}")
        print(f"  OVERALL:      {result['overall_score']}/100")
        print(f"\nFEEDBACK:")
        for line in result['feedback']:
            print(f"  - {line}")
        input("\nPress Enter to continue to next question...")

    # Step 4: Final Summary Report
    if all_results:
        print("\n" + "=" * 60)
        print("   FINAL INTERVIEW REPORT")
        print("=" * 60)

        avg_overall = round(sum(r['overall_score'] for r in all_results) / len(all_results), 2)
        avg_relevance = round(sum(r['relevance_score'] for r in all_results) / len(all_results), 2)
        avg_completeness = round(sum(r['completeness_score'] for r in all_results) / len(all_results), 2)
        total_fillers = sum(r['filler_word_count'] for r in all_results)

        print(f"\nQuestions attempted: {len(all_results)}")
        print(f"Average Overall Score:   {avg_overall}/100")
        print(f"Average Relevance:       {avg_relevance}/100")
        print(f"Average Completeness:    {avg_completeness}/100")
        print(f"Total Filler Words Used: {total_fillers}")

        # Performance label
        if avg_overall >= 80:
            label = "Excellent! You're well prepared for interviews."
        elif avg_overall >= 60:
            label = "Good effort! Focus on using technical terms and reducing fillers."
        elif avg_overall >= 40:
            label = "Needs improvement. Practice answering with more complete technical explanations."
        else:
            label = "Keep practicing! Focus on understanding core concepts before your interview."

        print(f"\nOVERALL PERFORMANCE: {label}")

        # Save results to outputs folder
        output_path = os.path.join(script_dir, '..', 'outputs', 'session_results.json')
        with open(output_path, 'w') as f:
            json.dump({
                "summary": {
                    "questions_attempted": len(all_results),
                    "avg_overall_score": avg_overall,
                    "avg_relevance": avg_relevance,
                    "avg_completeness": avg_completeness,
                    "total_fillers": total_fillers,
                    "performance_label": label
                },
                "detailed_results": all_results
            }, f, indent=2)

        print(f"\nDetailed results saved to: outputs/session_results.json")
        print("=" * 60)


# Run the session
run_interview()