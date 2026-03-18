"""Overnight text experiment runner.
Tests self-wiring graph on real English text (Gutenberg books).
Each invocation picks the next untested experiment and runs it.
Results saved to tests/results/overnight_text/"""
import sys, os, time, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'overnight_text')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Text loading and processing
# ============================================================

def load_all_text():
    """Load and concatenate all text files."""
    texts = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.endswith('.txt'):
            with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
    return '\n'.join(texts)


def clean_text(text):
    """Normalize to lowercase a-z + space."""
    text = text.lower()
    clean = ''.join(c if c.isalpha() or c == ' ' else ' ' for c in text)
    return ' '.join(clean.split())


def build_char_bigrams(text):
    """Build char bigram targets. Returns (chars, targets, confidence)."""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    V = len(chars)

    transitions = Counter()
    for i in range(len(text) - 1):
        transitions[(text[i], text[i+1])] += 1

    targets = np.zeros(V, dtype=np.int32)
    confidence = np.zeros(V, dtype=np.float32)
    for i, c in enumerate(chars):
        nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
        if nexts:
            best = max(nexts, key=nexts.get)
            targets[i] = char2idx[best]
            confidence[i] = nexts[best] / sum(nexts.values())
    return chars, targets, confidence


def build_word_bigrams(text, top_n=64):
    """Build word bigram targets for top_n words."""
    words = text.split()
    word_counts = Counter(words)
    vocab = [w for w, _ in word_counts.most_common(top_n)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    transitions = Counter()
    for i in range(len(words) - 1):
        if words[i] in word2idx and words[i+1] in word2idx:
            transitions[(words[i], words[i+1])] += 1

    targets = np.zeros(V, dtype=np.int32)
    confidence = np.zeros(V, dtype=np.float32)
    for i, w in enumerate(vocab):
        nexts = {w2: cnt for (w1, w2), cnt in transitions.items() if w1 == w}
        if nexts:
            best = max(nexts, key=nexts.get)
            targets[i] = word2idx[best]
            confidence[i] = nexts[best] / sum(nexts.values())
    return vocab, targets, confidence


# ============================================================
# Forward pass variants
# ============================================================

def forward_retain(net, ticks=8):
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
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def forward_divnorm(net, ticks=8, alpha=2.0):
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


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


FORWARD_FNS = {
    'retain': forward_retain,
    'divnorm': forward_divnorm,
    'both': forward_both,
}


# ============================================================
# Training
# ============================================================

def train_experiment(net, targets, budget, forward_fn, ticks, stale_limit=6000, log_every=None):
    if log_every is None:
        log_every = max(budget // 10, 1000)

    def evaluate():
        logits = forward_fn(net, ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        V = net.V
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    score = evaluate()
    best = score
    stale = 0
    trajectory = [(0, float(best))]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate()
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
            trajectory.append((att + 1, float(best)))
        if best >= 0.99 or stale >= stale_limit:
            break

    if (att + 1) % log_every != 0:
        trajectory.append((att + 1, float(best)))

    return float(best), att + 1, trajectory


def generate_text_char(net, forward_fn, ticks, chars, start_char='t', length=100):
    """Generate text using trained char bigram model."""
    logits = forward_fn(net, ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    char2idx = {c: i for i, c in enumerate(chars)}
    result = []
    idx = char2idx.get(start_char, 0)
    seen = set()
    for _ in range(length):
        result.append(chars[idx])
        next_idx = preds[idx]
        if (idx, next_idx) in seen and len(result) > 10:
            result.append('[LOOP]')
            break
        seen.add((idx, next_idx))
        idx = next_idx
    return ''.join(result)


def generate_text_word(net, forward_fn, ticks, vocab, start_word='the', length=30):
    """Generate text using trained word bigram model."""
    logits = forward_fn(net, ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    word2idx = {w: i for i, w in enumerate(vocab)}
    result = []
    idx = word2idx.get(start_word, 0)
    seen = set()
    for _ in range(length):
        result.append(vocab[idx])
        next_idx = preds[idx]
        if (idx, next_idx) in seen and len(result) > 5:
            result.append('[LOOP]')
            break
        seen.add((idx, next_idx))
        idx = next_idx
    return ' '.join(result)


# ============================================================
# Experiment definitions
# ============================================================

def get_experiments():
    """Define all experiments to run."""
    experiments = []

    # Char bigrams: different forward modes and tick counts
    for mode in ['retain', 'divnorm', 'both']:
        for ticks in [4, 8, 16]:
            for seed in [0, 42]:
                experiments.append({
                    'name': f'char_bigram_{mode}_t{ticks}_s{seed}',
                    'type': 'char',
                    'mode': mode,
                    'ticks': ticks,
                    'seed': seed,
                    'budget': 16000,
                })

    # Word bigrams V=32: different modes and ticks
    for mode in ['retain', 'divnorm', 'both']:
        for ticks in [4, 8, 16]:
            for seed in [0, 42]:
                experiments.append({
                    'name': f'word32_bigram_{mode}_t{ticks}_s{seed}',
                    'type': 'word',
                    'top_n': 32,
                    'mode': mode,
                    'ticks': ticks,
                    'seed': seed,
                    'budget': 16000,
                })

    # Word bigrams V=64: best modes, longer budget
    for mode in ['both', 'divnorm']:
        for ticks in [8, 16]:
            for seed in [0, 42]:
                experiments.append({
                    'name': f'word64_bigram_{mode}_t{ticks}_s{seed}',
                    'type': 'word',
                    'top_n': 64,
                    'mode': mode,
                    'ticks': ticks,
                    'seed': seed,
                    'budget': 48000,
                })

    # Word bigrams V=128: ambitious, long runs
    for mode in ['both', 'divnorm']:
        for ticks in [8, 16]:
            for seed in [0, 42]:
                experiments.append({
                    'name': f'word128_bigram_{mode}_t{ticks}_s{seed}',
                    'type': 'word',
                    'top_n': 128,
                    'mode': mode,
                    'ticks': ticks,
                    'seed': seed,
                    'budget': 96000,
                    'stale_limit': 15000,
                })

    return experiments


# ============================================================
# Main runner
# ============================================================

def main():
    # Load text
    raw = load_all_text()
    text = clean_text(raw)
    print(f"Loaded {len(text)} chars of clean text")

    # Precompute targets
    chars, char_targets, char_conf = build_char_bigrams(text)
    print(f"Char vocabulary: {len(chars)} chars, avg confidence: {char_conf.mean():.1%}")

    word_targets_cache = {}

    # Get all experiments
    all_exps = get_experiments()

    # Find which are already done
    done = set()
    for fn in os.listdir(RESULTS_DIR):
        if fn.endswith('.json'):
            done.add(fn.replace('.json', ''))

    pending = [e for e in all_exps if e['name'] not in done]

    if not pending:
        print("ALL EXPERIMENTS DONE! Nothing left to run.")
        # Print summary
        print_summary()
        return

    # Pick next experiment
    exp = pending[0]
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']} ({len(pending)} remaining)")
    print(f"{'='*60}")

    # Prepare targets
    if exp['type'] == 'char':
        V = len(chars)
        targets = char_targets
        labels = chars
    else:
        top_n = exp['top_n']
        if top_n not in word_targets_cache:
            word_targets_cache[top_n] = build_word_bigrams(text, top_n)
        vocab, targets, conf = word_targets_cache[top_n]
        V = len(vocab)
        labels = vocab
        print(f"Word vocabulary: {V} words, avg confidence: {conf.mean():.1%}")

    # Create network
    seed = exp['seed']
    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(V)

    # Shuffle targets (important: use same seed for reproducibility)
    if exp['type'] == 'char':
        # Char targets are derived from text, not shuffled
        pass
    # Word targets are already derived from text

    random.seed(seed * 1000 + 1)

    # Train
    forward_fn = FORWARD_FNS[exp['mode']]
    ticks = exp['ticks']
    budget = exp['budget']
    stale_limit = exp.get('stale_limit', 6000)

    t0 = time.time()
    best, steps, trajectory = train_experiment(
        net, targets, budget, forward_fn, ticks, stale_limit)
    elapsed = time.time() - t0

    # Generate sample text
    if exp['type'] == 'char':
        generated = generate_text_char(net, forward_fn, ticks, chars)
    else:
        generated = generate_text_word(net, forward_fn, ticks, vocab)

    # Count correct predictions
    logits = forward_fn(net, ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    correct = (preds[:V] == targets[:V]).sum()

    # Save results
    result = {
        'name': exp['name'],
        'type': exp['type'],
        'mode': exp['mode'],
        'ticks': ticks,
        'seed': seed,
        'V': V,
        'budget': budget,
        'score': best,
        'steps': steps,
        'elapsed': round(elapsed, 1),
        'trajectory': trajectory,
        'correct': int(correct),
        'total': int(V),
        'accuracy': round(correct / V * 100, 1),
        'generated': generated,
        'connections': net.count_connections(),
        'random_baseline': round(100 / V, 1),
    }

    result_path = os.path.join(RESULTS_DIR, f"{exp['name']}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Print results
    traj_str = " → ".join(f"{b*100:.1f}" for _, b in trajectory)
    print(f"\nResult: {best*100:.1f}% ({correct}/{V} correct) in {steps} steps ({elapsed:.0f}s)")
    print(f"Trajectory: {traj_str}")
    print(f"Generated: {generated[:100]}")
    print(f"Conns: {net.count_connections()}")
    print(f"Random baseline: {100/V:.1f}%")
    print(f"Saved to: {result_path}")
    print(f"\nRemaining: {len(pending)-1} experiments")

    # Quick summary of what we have so far
    print_summary()


def print_summary():
    """Print summary of all completed experiments."""
    results = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                results.append(json.load(f))

    if not results:
        return

    print(f"\n{'='*80}")
    print(f"OVERNIGHT RESULTS SUMMARY ({len(results)} experiments done)")
    print(f"{'='*80}")

    # Group by type and mode
    groups = {}
    for r in results:
        key = f"{r['type']}_{r.get('V', '?')}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for group_name in sorted(groups.keys()):
        group = groups[group_name]
        print(f"\n--- {group_name} ---")
        print(f"{'name':<40s} {'score':>7s} {'acc':>6s} {'steps':>7s} {'time':>5s}")
        for r in sorted(group, key=lambda x: -x['score']):
            print(f"{r['name']:<40s} {r['score']*100:6.1f}% {r['accuracy']:5.1f}% "
                  f"{r['steps']:7d} {r['elapsed']:4.0f}s")

    # Best per group
    print(f"\n--- BEST PER GROUP ---")
    for group_name in sorted(groups.keys()):
        group = groups[group_name]
        best = max(group, key=lambda x: x['score'])
        print(f"  {group_name}: {best['name']} = {best['score']*100:.1f}% "
              f"(random: {best['random_baseline']}%)")


if __name__ == '__main__':
    main()
