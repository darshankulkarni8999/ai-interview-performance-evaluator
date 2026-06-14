import re
from spellchecker import SpellChecker

# ---- STEP 1: Sample candidate answer (same as before) ----
candidate_answer = "Um, so basically lists can be changed after creation but, like, tuples cannot be changed. Lists use [] and tuples use ()."


# ---- STEP 2: Filler Word Detection ----
FILLER_WORDS = [
    "um", "uh", "like", "basically", "actually", "literally",
    "you know", "i mean", "sort of", "kind of", "so yeah",
    "right", "well", "okay so"
]


def count_filler_words(text, filler_list):
    """
    Counts how many times filler words appear in the text.
    Uses word boundaries (\\b) so 'like' doesn't match inside
    words like 'likely'.
    """
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


total_fillers, filler_breakdown = count_filler_words(candidate_answer, FILLER_WORDS)
print("---- FILLER WORD ANALYSIS ----")
print(f"Total filler words: {total_fillers}")
print(f"Breakdown: {filler_breakdown}")


# ---- STEP 3: Spelling Check with Technical Dictionary ----
spell = SpellChecker()

# Common technical/CS terms that general dictionaries often flag as
# "misspelled" but are completely valid in our domain (interviews,
# programming, ML). We add these to the spell checker's known words
# so they aren't flagged.
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

# Add these words to the spell checker's dictionary
spell.word_frequency.load_words(TECHNICAL_TERMS)


def check_spelling(text, filler_list, spell_checker):
    """
    Checks each word in the text against a dictionary of correctly
    spelled English words (now including technical terms).
    Returns misspelled words and a score.
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())

    filler_set = set(f.replace(" ", "") for f in filler_list)
    words_to_check = [w for w in words if w not in filler_set]

    misspelled = spell_checker.unknown(words_to_check)

    num_issues = len(misspelled)
    spelling_score = max(0, 100 - (num_issues * 15))

    return spelling_score, list(misspelled)


spelling_score, misspelled_words = check_spelling(candidate_answer, FILLER_WORDS, spell)

print(f"\n---- SPELLING SCORE: {spelling_score}/100 ----")
print(f"Misspelled words: {misspelled_words}")