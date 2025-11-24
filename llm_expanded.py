# llm_analogy_advanced.py
# A slightly more realistic analogy for how LLMs "learn" probabilities
# and use context to pick the next token.

import random
from collections import defaultdict

# -----------------------------------------------------------
# PART 1 — "TRAINING": Build fake word statistics
# -----------------------------------------------------------

# Pretend this is our training data (tiny on purpose)
training_sentences = [
    "i want coffee",
    "i want coffee",
    "i want tea",
    "can i have coffee",
    "can i have juice",
    "i want juice",
    "tea please",
    "coffee please"
]

# Count how often each word follows another word
transition_counts = defaultdict(lambda: defaultdict(int))

for sentence in training_sentences:
    words = sentence.split()
    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        transition_counts[current_word][next_word] += 1

# -----------------------------------------------------------
# PART 2 — Helper: Convert counts into probability weights
# -----------------------------------------------------------

def get_probabilities(word):
    """Return two lists: next-word options and their probability weights."""
    next_words = list(transition_counts[word].keys())
    counts = list(transition_counts[word].values())
    total = sum(counts)
    weights = [c / total for c in counts]  # convert to probabilities
    return next_words, weights

# -----------------------------------------------------------
# PART 3 — Two types of predictors: naive + probability-based
# -----------------------------------------------------------

def next_word_naive():
    """Pick *any* word in the vocabulary with equal randomness."""
    vocab = {w for sentence in training_sentences for w in sentence.split()}
    return random.choice(list(vocab))

def next_word_weighted(context_word):
    """Pick the next token based on learned training patterns."""
    next_words, weights = get_probabilities(context_word)
    return random.choices(next_words, weights=weights, k=1)[0]

# -----------------------------------------------------------
# PART 4 — Generate a short fake sentence like an LLM
# -----------------------------------------------------------

def generate_sentence(start="i", steps=5):
    """Generate a small sentence by doing next-token prediction repeatedly."""
    current = start
    sentence = [current]

    for _ in range(steps):
        if current not in transition_counts:
            break  # no learned transitions
        current = next_word_weighted(current)
        sentence.append(current)

    return " ".join(sentence)

# -----------------------------------------------------------
# DEMO OUTPUT
# -----------------------------------------------------------

print(">>> SIMPLE PREDICTION EXAMPLES")
print("Naive (no learning):", next_word_naive())
print("Weighted (after 'i'):", next_word_weighted("i"))
print("Weighted (after 'want'):", next_word_weighted("want"))
print()

print(">>> GENERATED SENTENCES")
print(generate_sentence(start="i", steps=5))
print(generate_sentence(start="can", steps=5))
print(generate_sentence(start="coffee", steps=5))