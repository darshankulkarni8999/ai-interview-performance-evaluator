import os
import json
from sentence_transformers import SentenceTransformer, util

# ---- STEP 1: Load the Q&A dataset ----
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, '..', 'data', 'qa_dataset.json')

with open(dataset_path, 'r') as f:
    qa_data = json.load(f)


# ---- STEP 2: Load the Sentence Transformer model ----
# 'all-MiniLM-L6-v2' is a small, fast, popular model good for
# semantic similarity tasks. It converts sentences into numerical
# vectors (embeddings) that capture MEANING, not just exact words.
print("Loading model... (first time may take a minute)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")


# ---- STEP 3: Pick a sample question to test ----
# Let's test with the first Python question in the dataset
sample = qa_data[0]
print(f"\nQUESTION: {sample['question']}")
print(f"\nIDEAL ANSWER: {sample['ideal_answer']}")


# ---- STEP 4: Simulate a candidate's answer ----
# For now, we'll hardcode a sample answer to test the system.
# Later, this will come from actual user input.
candidate_answer = "Lists can be changed after creation but tuples cannot be changed. Lists use [] and tuples use ()."


# ---- STEP 5: Calculate Relevance Score using Cosine Similarity ----
def calculate_relevance(candidate_answer, ideal_answer, model):
    """
    Converts both answers into embeddings (numerical vectors that
    represent meaning) and calculates cosine similarity between them.

    Cosine similarity ranges from -1 to 1:
    - 1 means identical meaning
    - 0 means unrelated
    - Negative means opposite meaning (rare for text)

    We multiply by 100 to get a percentage-style score.
    """
    # Convert both texts into embeddings (vectors)
    embeddings = model.encode([candidate_answer, ideal_answer])

    # Calculate cosine similarity between the two vectors
    similarity = util.cos_sim(embeddings[0], embeddings[1])

    # similarity is a tensor like [[0.85]], extract the number
    score = similarity.item()

    # Convert to 0-100 scale, clamped at 0 minimum
    relevance_score = max(0, round(score * 100, 2))

    return relevance_score


relevance = calculate_relevance(candidate_answer, sample['ideal_answer'], model)
print(f"\n---- RELEVANCE SCORE: {relevance}/100 ----")


# ---- STEP 6: Calculate Completeness Score using key_concepts ----
def calculate_completeness(candidate_answer, key_concepts):
    """
    Checks how many of the 'key_concepts' (important keywords)
    appear in the candidate's answer. This is a simple keyword-based
    check - it doesn't use AI, just text matching.

    Returns the percentage of key concepts found.
    """
    answer_lower = candidate_answer.lower()
    found = 0
    matched_concepts = []

    for concept in key_concepts:
        if concept.lower() in answer_lower:
            found += 1
            matched_concepts.append(concept)

    completeness_score = round((found / len(key_concepts)) * 100, 2)
    return completeness_score, matched_concepts


completeness, matched = calculate_completeness(candidate_answer, sample['key_concepts'])
print(f"\n---- COMPLETENESS SCORE: {completeness}/100 ----")
print(f"Key concepts covered: {matched}")
print(f"Key concepts expected: {sample['key_concepts']}")