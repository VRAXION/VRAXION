"""
Generate permutation sorting data
==================================
Input: random permutation of digits 0-9
Output: always "0123456789"
Format: "7350218964=0123456789\n" = 21 bytes per example
10! = 3,628,800 possible permutations
"""
import os, random, itertools
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SORTED = "0123456789"

# Generate all 10! permutations is too many (3.6M), sample instead
random.seed(42)
examples = []
digits = list("0123456789")
for _ in range(50_000):
    random.shuffle(digits)
    inp = ''.join(digits)
    examples.append(f"{inp}={SORTED}\n")

print(f"Generated {len(examples)} permutation sorting examples")
print(f"Example: {examples[0].strip()}")
print(f"Example: {examples[1].strip()}")

# Shuffle and build corpus ~100KB+
random.shuffle(examples)
corpus_text = ''.join(examples)
# Repeat if needed
while len(corpus_text) < 200_000:
    random.shuffle(examples)
    corpus_text += ''.join(examples)

corpus_bytes = corpus_text.encode('ascii')
corpus = np.frombuffer(corpus_bytes, dtype=np.uint8).copy()

traindat_path = os.path.join(DATA_DIR, "perm_sorting.traindat")
corpus.tofile(traindat_path)
print(f"Saved {len(corpus)} bytes ({len(corpus)//21} examples) to {traindat_path}")

# --- Compute bigram table ---
bigram = np.zeros((256, 256), dtype=np.float64)
for i in range(len(corpus) - 1):
    bigram[corpus[i], corpus[i+1]] += 1

row_sums = bigram.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
bigram = (bigram / row_sums).astype(np.float32)

bigram_path = os.path.join(DATA_DIR, "perm_sorting_bigram.npy")
np.save(bigram_path, bigram)
print(f"Saved bigram to {bigram_path}")

# --- Sanity checks ---
eq = ord('=')
print(f"\nBigram sanity (output side):")
print(f"  '=' -> ", end="")
for d in "0123456789":
    print(f"'{d}'={bigram[eq, ord(d)]:.3f} ", end="")
print()

for d in "012345678":
    nd = str(int(d)+1)
    print(f"  '{d}' -> '{nd}' = {bigram[ord(d), ord(nd)]:.3f}  (want ~0.5+)")

print(f"\n  '9' -> '\\n' = {bigram[ord('9'), ord(chr(10))]:.3f}")
print(f"  '\\n' -> any digit (input start):")
for d in "0123456789":
    print(f"    '{d}'={bigram[10, ord(d)]:.3f}", end="")
print()
print(f"\nCorpus: {len(corpus)/1024:.1f} KB")
