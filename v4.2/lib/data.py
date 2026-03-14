"""
Corpus and bigram data for v22 experiments.
Lazy-loaded: bigram distribution computed on first access, not at import.
"""

import numpy as np
from collections import Counter

TEXT = (
    "a stitch in time saves nine. "
    "the early bird catches the worm. "
    "all that glitters is not gold. "
    "actions speak louder than words. "
    "fortune favors the bold. "
    "knowledge is power. "
    "practice makes perfect. "
    "the pen is mightier than the sword. "
    "where there is a will there is a way. "
    "birds of a feather flock together. "
    "every cloud has a silver lining. "
    "honesty is the best policy. "
    "look before you leap. "
    "better late than never. "
    "two wrongs do not make a right. "
    "when in rome do as the romans do. "
    "the squeaky wheel gets the grease. "
    "you can lead a horse to water but you cannot make it drink. "
    "curiosity killed the cat. "
    "do not put all your eggs in one basket. "
    "if you want something done right do it yourself. "
    "the apple does not fall far from the tree. "
    "people who live in glass houses should not throw stones. "
)

chars = sorted(set(TEXT))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB = len(chars)

# Lazy bigram cache
_bigram_cache = {}


def compute_bigram_dist():
    """Compute bigram probability distribution from TEXT. Cached after first call."""
    if 'dist' in _bigram_cache:
        return _bigram_cache['dist'], _bigram_cache['active']

    bigrams = {}
    for i in range(len(TEXT) - 1):
        c1, c2 = TEXT[i], TEXT[i + 1]
        if c1 in char_to_idx and c2 in char_to_idx:
            idx1 = char_to_idx[c1]
            bigrams.setdefault(idx1, []).append(char_to_idx[c2])

    dist = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i, nexts in bigrams.items():
        counts = Counter(nexts)
        total = sum(counts.values())
        for j, count in counts.items():
            dist[i, j] = count / total

    active = [i for i in range(VOCAB) if dist[i].sum() > 0.01]

    _bigram_cache['dist'] = dist
    _bigram_cache['active'] = active
    return dist, active


def bigram_targets():
    """Most common next char for each char. Returns (VOCAB,) int array."""
    dist, _ = compute_bigram_dist()
    targets = np.zeros(VOCAB, dtype=int)
    for i in range(VOCAB):
        if dist[i].sum() > 0:
            targets[i] = np.argmax(dist[i])
        else:
            targets[i] = (i + 1) % VOCAB
    return targets
