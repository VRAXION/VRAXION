"""
Generate sorting training data + bigram table
==============================================
4 digits (0-9, repeats OK) -> sorted output.
Format: "5381=1358\n" = 10 bytes per example.
10^4 = 10,000 unique combinations, shuffled and repeated to ~100KB.
"""
import os, random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Generate all 10,000 combinations ---
examples = []
for a in range(10):
    for b in range(10):
        for c in range(10):
            for d in range(10):
                digits = [a, b, c, d]
                inp = ''.join(str(x) for x in digits)
                out = ''.join(str(x) for x in sorted(digits))
                examples.append(f"{inp}={out}\n")

print(f"Generated {len(examples)} unique sorting examples")
print(f"Example: {examples[5381].strip()}")

# Shuffle and repeat to ~100KB
random.seed(42)
random.shuffle(examples)
corpus_text = ''
while len(corpus_text) < 100_000:
    random.shuffle(examples)
    corpus_text += ''.join(examples)

corpus_bytes = corpus_text.encode('ascii')
corpus = np.frombuffer(corpus_bytes, dtype=np.uint8).copy()

# Save training data
traindat_path = os.path.join(DATA_DIR, "sorting.traindat")
corpus.tofile(traindat_path)
print(f"Saved {len(corpus)} bytes to {traindat_path}")

# --- Compute bigram table ---
bigram = np.zeros((256, 256), dtype=np.float64)
for i in range(len(corpus) - 1):
    bigram[corpus[i], corpus[i+1]] += 1

# Row-normalize
row_sums = bigram.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # avoid div by zero
bigram = (bigram / row_sums).astype(np.float32)

bigram_path = os.path.join(DATA_DIR, "sorting_bigram.npy")
np.save(bigram_path, bigram)
print(f"Saved bigram table to {bigram_path}")

# --- Sanity checks ---
eq = ord('=')
nl = ord('\n')
d0 = ord('0')

print(f"\nBigram sanity:")
print(f"  After '=': top predictions = ", end="")
top_after_eq = np.argsort(bigram[eq])[::-1][:5]
for b in top_after_eq:
    print(f"'{chr(b)}'={bigram[eq, b]:.3f} ", end="")
print()

print(f"  After '5': top predictions = ", end="")
top_after_5 = np.argsort(bigram[ord('5')])[::-1][:5]
for b in top_after_5:
    print(f"'{chr(b)}'={bigram[ord('5'), b]:.3f} ", end="")
print()

print(f"  After '\\n': top predictions = ", end="")
top_after_nl = np.argsort(bigram[nl])[::-1][:5]
for b in top_after_nl:
    print(f"'{chr(b)}'={bigram[nl, b]:.3f} ", end="")
print()

# Count active bytes
active = (bigram.sum(axis=1) > 0).sum()
print(f"\n  Active byte rows: {active}/256")
print(f"  Corpus size: {len(corpus)/1024:.1f} KB")
print(f"  Examples per 50-byte window: ~{50//10}")
