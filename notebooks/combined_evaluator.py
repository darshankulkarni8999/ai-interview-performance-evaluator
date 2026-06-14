import os
import re
import json
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

# ============================================================
# SETUP — Load everything we need once
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, '..', 'data', 'qa_dataset.json')

with open(dataset_path, 'r') as f:
    qa_data = json.load(f)

print("Loading models... (first time may take a minute)")
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

print("Models loaded!\n")


# ============================================================
# SCORING FUNCTIONS (from previous modules)
# ============================================================

def calculate_relevance(candidate_answer, ideal_answer, model):
    """Semantic similarity between candidate and ideal answer (0-100)."""
    embeddings = model.encode([candidate_answer, ideal_answer])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    score = similarity.item()
    return max(0, round(score * 100, 2))


def calculate_completeness(candidate_answer, key_concepts):
    """Percentage of key concepts mentioned in the answer."""
    answer_lower = candidate_answer.lower()
    found = 0
    matched_concepts = []
    missing_concepts = []

    for concept in key_concepts:
        if concept.lower() in answer_lower:
            found += 1
            matched_concepts.append(concept)
        else:
            missing_concepts.append(concept)

    score = round((found / len(key_concepts)) * 100, 2)
    return score, matched_concepts, missing_concepts


def count_filler_words(text, filler_list):
    """Counts filler words like 'um', 'like', 'basically'."""
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
    """Checks for misspelled words, ignoring fillers and technical terms."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    filler_set = set(f.replace(" ", "") for f in filler_list)
    words_to_check = [w for w in words if w not in filler_set]

    misspelled = spell_checker.unknown(words_to_check)
    num_issues = len(misspelled)
    score = max(0, 100 - (num_issues * 15))

    return score, list(misspelled)


# ============================================================
# FEEDBACK GENERATOR
# ============================================================

def generate_feedback(relevance, completeness, fillers, filler_breakdown,
                       spelling, matched_concepts, missing_concepts):
    """
    Generates human-readable feedback based on all scores.
    Uses simple if-else rules - this is RULE-BASED, not AI-generated,
    but it's still a valid and explainable feedback system.
    """
    feedback_lines = []

    # --- Relevance feedback ---
    if relevance >= 75:
        feedback_lines.append("Your answer is highly relevant to the question.")
    elif relevance >= 50:
        feedback_lines.append("Your answer is somewhat relevant, but could be more focused on the core topic.")
    else:
        feedback_lines.append("Your answer doesn't seem closely related to the question. Re-read the question and try to address it more directly.")

    # --- Completeness feedback ---
    if completeness >= 80:
        feedback_lines.append("You covered most of the important concepts.")
    elif completeness >= 40:
        feedback_lines.append(f"You covered some key points, but missed: {', '.join(missing_concepts)}.")
    else:
        feedback_lines.append(f"Your answer is missing several important concepts: {', '.join(missing_concepts)}. Try including these terms with explanations.")

    # --- Filler words feedback ---
    if fillers == 0:
        feedback_lines.append("Great job avoiding filler words!")
    elif fillers <= 2:
        feedback_lines.append(f"You used a few filler words ({fillers} total). Try to reduce these for a more confident delivery.")
    else:
        feedback_lines.append(f"You used {fillers} filler words ({', '.join(filler_breakdown.keys())}). Practice speaking more concisely to sound more confident.")

    # --- Spelling feedback ---
    if spelling == 100:
        feedback_lines.append("No spelling issues detected.")
    else:
        feedback_lines.append("Some words may be misspelled - double check your answer.")

    return feedback_lines


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_answer(question_data, candidate_answer):
    """
    Takes a question (with ideal_answer and key_concepts) and a
    candidate's answer, and returns a full evaluation report.
    """
    relevance = calculate_relevance(candidate_answer, question_data['ideal_answer'], model)
    completeness, matched, missing = calculate_completeness(candidate_answer, question_data['key_concepts'])
    fillers, filler_breakdown = count_filler_words(candidate_answer, FILLER_WORDS)
    spelling, misspelled = check_spelling(candidate_answer, FILLER_WORDS, spell)

    # --- Overall Score Calculation ---
    # Weighted average: relevance and completeness matter most,
    # spelling matters least. Filler words reduce score via penalty.
    filler_penalty = min(fillers * 5, 20)  # max 20 point penalty

    overall_score = (
        relevance * 0.40 +
        completeness * 0.35 +
        spelling * 0.15 +
        max(0, 100 - filler_penalty) * 0.10
    )
    overall_score = round(overall_score, 2)

    feedback = generate_feedback(
        relevance, completeness, fillers, filler_breakdown,
        spelling, matched, missing
    )

    return {
        "question": question_data['question'],
        "relevance_score": relevance,
        "completeness_score": completeness,
        "spelling_score": spelling,
        "filler_word_count": fillers,
        "filler_breakdown": filler_breakdown,
        "misspelled_words": misspelled,
        "overall_score": overall_score,
        "feedback": feedback
    }


# ============================================================
# TEST THE FULL PIPELINE
# ============================================================

if __name__ == "__main__":
    # Pick a sample question
    sample_question = qa_data[0]  # "What is the difference between a list and a tuple?"

    # Sample candidate answer
    candidate_answer = "Um, so basically lists can be changed after creation but, like, tuples cannot be changed. Lists use [] and tuples use ()."

    result = evaluate_answer(sample_question, candidate_answer)

    print("=" * 60)
    print(f"QUESTION: {result['question']}")
    print("=" * 60)
    print(f"\nOVERALL SCORE: {result['overall_score']}/100\n")
    print(f"Relevance:    {result['relevance_score']}/100")
    print(f"Completeness: {result['completeness_score']}/100")
    print(f"Spelling:     {result['spelling_score']}/100")
    print(f"Filler words: {result['filler_word_count']} -> {result['filler_breakdown']}")
    print(f"\nFEEDBACK:")
    for line in result['feedback']:
        print(f" - {line}")