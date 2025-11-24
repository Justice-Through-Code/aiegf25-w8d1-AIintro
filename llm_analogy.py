# llm_analogy.py
# Analogy: naive random choice vs. probability-weighted choice.
import random

choices = [("coffee", 0.6), ("tea", 0.3), ("juice", 0.1)] # we create a list of pairs (tuples)
words   = [w for (w, p) in choices] # list comprehension - pulls out all the word values
weights = [p for (w, p) in choices] # list comprehension - pulls out all the number values

def next_word_naive():
    # Picks any option with equal chance.
    return random.choice(words)

def next_word_weighted():
    # Picks based on learned "likelihoods" (weights).
    return random.choices(words, weights=weights, k=1)[0] 
    # words = list of options, weights = probabilities, and k=1 tells it to pick 1 item)

print("Naive pick:", next_word_naive())
print("Weighted pick:", next_word_weighted())

