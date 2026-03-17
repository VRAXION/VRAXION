"""Real text test: can the self-wiring graph learn character bigrams from English text?
Tests on actual language data, not synthetic permutations.
Uses best config: both (retain + divnorm) at 8 ticks."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

# --- English text corpus (Wikipedia-style, ~3000 chars) ---
TEXT = """
The human brain is the central organ of the nervous system and with the spinal
cord makes up the central nervous system. The brain consists of the cerebrum the
brainstem and the cerebellum. It controls most of the activities of the body
processing integrating and coordinating the information it receives from the
sense organs and making decisions as to the instructions sent to the rest of the
body. The brain is contained in and protected by the skull bones of the head.
The cerebrum is the largest part of the human brain. It is divided into two
cerebral hemispheres. The cerebral cortex is an outer layer of grey matter
covering the core of white matter. The cortex is split into the neocortex and
the much smaller allocortex. The neocortex is made up of six neuronal layers
while the allocortex has three or four. Each hemisphere is conventionally divided
into four lobes the frontal temporal parietal and occipital lobes. The frontal
lobe is associated with executive functions including self control planning
reasoning and abstract thought while the occipital lobe is dedicated to vision.
Within each lobe cortical areas are associated with specific functions such as
the sensory motor and association regions. Although the left and right hemispheres
are broadly similar in shape and function some functions are associated with one
side or the other. The hippocampus is involved in memory formation and the
amygdala plays a role in fear and emotional processing. Neural networks in the
brain form complex interconnected circuits that enable thought perception and
behavior. Neurons communicate through electrical impulses and chemical signals
at specialized junctions called synapses. The strength of synaptic connections
can change over time through processes known as synaptic plasticity which is
believed to be the cellular basis of learning and memory. The brain consumes
about twenty percent of the total energy used by the human body despite making
up only about two percent of body mass. This high metabolic rate is necessary
to support the constant electrical signaling between neurons. Sleep is essential
for brain function allowing consolidation of memories and clearing of metabolic
waste products. The study of the brain is known as neuroscience and has made
significant advances in understanding brain structure and function through
techniques such as magnetic resonance imaging and electroencephalography.
Evolution has shaped the brain over millions of years with increasing complexity
seen across different species. The human brain with its remarkable capacity for
language abstract reasoning and consciousness represents one of the most complex
structures known in the universe. Understanding how the brain gives rise to mind
and consciousness remains one of the greatest challenges in science today.
"""

def build_char_bigrams(text):
    """Build character bigram targets from text."""
    # Normalize
    text = text.lower()
    # Keep only a-z and space
    clean = ''.join(c if c.isalpha() or c == ' ' else ' ' for c in text)
    clean = ' '.join(clean.split())  # normalize whitespace

    # Count bigram transitions
    transitions = Counter()
    for i in range(len(clean) - 1):
        transitions[(clean[i], clean[i+1])] += 1

    # Build vocabulary
    chars = sorted(set(clean))
    char2idx = {c: i for i, c in enumerate(chars)}
    V = len(chars)

    # For each char, find most likely next char
    targets = np.zeros(V, dtype=np.int32)
    confidence = np.zeros(V, dtype=np.float32)
    for i, c in enumerate(chars):
        nexts = {c2: count for (c1, c2), count in transitions.items() if c1 == c}
        if nexts:
            best_next = max(nexts, key=nexts.get)
            targets[i] = char2idx[best_next]
            total = sum(nexts.values())
            confidence[i] = nexts[best_next] / total
        else:
            targets[i] = i  # self-loop fallback

    return chars, char2idx, targets, confidence, clean


def build_word_bigrams(text, top_n=64):
    """Build word bigram targets from text."""
    text = text.lower()
    words = [w for w in text.split() if w.isalpha()]

    # Top N most common words
    word_counts = Counter(words)
    vocab = [w for w, _ in word_counts.most_common(top_n)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # Count word bigram transitions (only within vocabulary)
    transitions = Counter()
    for i in range(len(words) - 1):
        if words[i] in word2idx and words[i+1] in word2idx:
            transitions[(words[i], words[i+1])] += 1

    # For each word, find most likely next word
    targets = np.zeros(V, dtype=np.int32)
    confidence = np.zeros(V, dtype=np.float32)
    for i, w in enumerate(vocab):
        nexts = {w2: count for (w1, w2), count in transitions.items() if w1 == w}
        if nexts:
            best_next = max(nexts, key=nexts.get)
            targets[i] = word2idx[best_next]
            total = sum(nexts.values())
            confidence[i] = nexts[best_next] / total
        else:
            targets[i] = i

    return vocab, word2idx, targets, confidence


def forward_both(net, ticks=8, alpha=2.0):
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate(net, targets):
    logits = forward_both(net)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train(net, targets, budget, stale_limit=6000, log_every=4000):
    score = evaluate(net, targets)
    best = score
    stale = 0
    trajectory = [(0, best)]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate(net, targets)
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1
        if (att + 1) % log_every == 0:
            trajectory.append((att + 1, best))
        if best >= 0.99 or stale >= stale_limit:
            break

    if (att + 1) % log_every != 0:
        trajectory.append((att + 1, best))
    return best, att + 1, trajectory


def show_predictions(net, targets, labels, name):
    """Show what the network predicts for each input."""
    logits = forward_both(net)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V

    preds = np.argmax(probs, axis=1)
    correct = preds == targets

    print(f"\n  {name} predictions (first 30):")
    for i in range(min(30, V)):
        pred_label = labels[preds[i]]
        target_label = labels[targets[i]]
        mark = "OK" if correct[i] else "XX"
        conf = probs[i, preds[i]]
        print(f"    '{labels[i]}' → '{pred_label}' (target='{target_label}') "
              f"conf={conf:.2f} [{mark}]")


# ============================================================
# TEST 1: Character bigrams
# ============================================================
print("=" * 60)
print("TEST 1: CHARACTER BIGRAM PREDICTION")
print("=" * 60)

chars, char2idx, char_targets, char_conf, clean_text = build_char_bigrams(TEXT)
V_char = len(chars)

print(f"Vocabulary: {V_char} characters: {''.join(chars)}")
print(f"Text length: {len(clean_text)} chars")
print(f"Avg bigram confidence (how predictable): {char_conf.mean():.1%}")
print(f"Unique targets: {len(set(char_targets.tolist()))}/{V_char}")

# How many chars map to same target?
target_counts = Counter(char_targets.tolist())
print(f"Most common target: '{chars[target_counts.most_common(1)[0][0]]}' "
      f"({target_counts.most_common(1)[0][1]} chars map to it)")

# Train
BUDGET_CHAR = 16000
print(f"\nTraining V={V_char}, budget={BUDGET_CHAR}, both_8t...")
np.random.seed(42); random.seed(42)
net = SelfWiringGraph(V_char)
random.seed(42001)
t0 = time.time()
best, steps, traj = train(net, char_targets, BUDGET_CHAR)
elapsed = time.time() - t0

traj_str = " → ".join(f"{b*100:.1f}" for _, b in traj)
print(f"Score: {best*100:.1f}% in {steps} steps ({elapsed:.0f}s)")
print(f"Trajectory: {traj_str}")

# Show predictions
show_predictions(net, char_targets, chars, "Character bigram")

# Generate text!
print(f"\n  Generated text (starting from 't', 200 chars):")
logits = forward_both(net)
e = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = e / e.sum(axis=1, keepdims=True)
preds = np.argmax(probs, axis=1)

generated = []
idx = char2idx['t']
for _ in range(200):
    generated.append(chars[idx])
    idx = preds[idx]
print(f"  {''.join(generated)}")

# ============================================================
# TEST 2: Word bigrams (top 64 words)
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: WORD BIGRAM PREDICTION (top 64 words)")
print("=" * 60)

vocab, word2idx, word_targets, word_conf = build_word_bigrams(TEXT, top_n=64)
V_word = len(vocab)

print(f"Vocabulary: {V_word} words")
print(f"Top 10: {', '.join(vocab[:10])}")
print(f"Avg bigram confidence: {word_conf.mean():.1%}")
print(f"Unique targets: {len(set(word_targets.tolist()))}/{V_word}")

# Train
BUDGET_WORD = 48000
print(f"\nTraining V={V_word}, budget={BUDGET_WORD}, both_8t...")
np.random.seed(42); random.seed(42)
net_w = SelfWiringGraph(V_word)
random.seed(42001)
t0 = time.time()
best_w, steps_w, traj_w = train(net_w, word_targets, BUDGET_WORD, log_every=8000)
elapsed_w = time.time() - t0

traj_str = " → ".join(f"{b*100:.1f}" for _, b in traj_w)
print(f"Score: {best_w*100:.1f}% in {steps_w} steps ({elapsed_w:.0f}s)")
print(f"Trajectory: {traj_str}")

# Show word predictions
show_predictions(net_w, word_targets, vocab, "Word bigram")

# Generate word sequence
print(f"\n  Generated text (starting from 'the', 30 words):")
logits_w = forward_both(net_w)
e = np.exp(logits_w - logits_w.max(axis=1, keepdims=True))
probs_w = e / e.sum(axis=1, keepdims=True)
preds_w = np.argmax(probs_w, axis=1)

generated_w = []
idx = word2idx.get('the', 0)
for _ in range(30):
    generated_w.append(vocab[idx])
    idx = preds_w[idx]
print(f"  {' '.join(generated_w)}")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"Character bigram (V={V_char}): {best*100:.1f}%")
print(f"Word bigram (V={V_word}):      {best_w*100:.1f}%")
print(f"Random baseline char:        {100/V_char:.1f}%")
print(f"Random baseline word:        {100/V_word:.1f}%")
