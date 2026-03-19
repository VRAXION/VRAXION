"""Overnight window LM sweep: one-hot encoding, different window sizes.
Input: window of N chars, each one-hot (27 dims) = N×27 input neurons.
Output: next char prediction (27 classes).
Uses batch mutation+selection (proven to work at 100% for permutation)."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'overnight_window')
os.makedirs(RESULTS_DIR, exist_ok=True)
ALPHA = 2.0
CHARS = list(' abcdefghijklmnopqrstuvwxyz')
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
N_CHARS = len(CHARS)  # 27


def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    clean = ''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text)
    return ' '.join(clean.split())


def make_windows_onehot(text, window_size, max_samples=50000):
    """Create windows with one-hot encoding. Input: (K, window*27), Target: (K,) class index."""
    n_input = window_size * N_CHARS
    n_samples = min(len(text) - window_size, max_samples)
    inputs = np.zeros((n_samples, n_input), dtype=np.float32)
    targets = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        for j in range(window_size):
            c = text[i + j]
            if c in CHAR2IDX:
                inputs[i, j * N_CHARS + CHAR2IDX[c]] = 1.0
        next_c = text[i + window_size]
        targets[i] = CHAR2IDX.get(next_c, 0)
    return inputs, targets


def forward_divnorm(net, patterns, ticks=8, alpha=ALPHA):
    K = patterns.shape[0]
    V = net.V; N = net.N
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = patterns
        raw = acts @ net.mask
        charges += raw
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    # Output: first N_CHARS neurons of output zone
    return charges[:, net.out_start:net.out_start + N_CHARS]


def evaluate(net, inputs, targets, sample_size=512):
    idx = np.random.choice(len(inputs), min(sample_size, len(inputs)), replace=False)
    logits = forward_divnorm(net, inputs[idx])
    preds = np.argmax(logits, axis=1)
    acc = (preds == targets[idx]).mean()
    # Also compute score (acc + target prob) for compatibility
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    tp = probs[np.arange(len(idx)), targets[idx]].mean()
    return float(acc), float(0.5 * acc + 0.5 * tp)


def train_window(net, inputs, targets, budget, sample_size=512,
                 stale_limit=8000, log_every=4000, ticks=8):
    acc, score = evaluate(net, inputs, targets, sample_size)
    best_acc = acc
    best_score = score
    stale = 0
    trajectory = [(0, float(best_acc))]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()

        new_acc, new_score = evaluate(net, inputs, targets, sample_size)
        if new_score > score:
            score = new_score
            acc = new_acc
            best_acc = max(best_acc, acc)
            best_score = max(best_score, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if (att + 1) % log_every == 0:
            trajectory.append((att + 1, float(best_acc)))

        if best_acc >= 0.99 or stale >= stale_limit:
            break

    if (att + 1) % log_every != 0:
        trajectory.append((att + 1, float(best_acc)))
    return best_acc, best_score, att + 1, trajectory


def get_experiments():
    experiments = []
    # Window sizes × budgets × seeds
    for window in [2, 3, 4]:
        for seed in [0, 42]:
            V = window * N_CHARS  # 54, 81, 108
            budget = 24000 if V <= 81 else 16000
            experiments.append({
                'name': f'win{window}_onehot_s{seed}',
                'window': window, 'V': V, 'seed': seed,
                'budget': budget, 'ticks': 8,
            })
    # Larger windows need more neurons and budget
    for window in [5, 6, 8]:
        for seed in [42]:
            V = window * N_CHARS
            experiments.append({
                'name': f'win{window}_onehot_s{seed}',
                'window': window, 'V': V, 'seed': seed,
                'budget': 16000, 'ticks': 8,
            })
    # Different NV ratios for window=3
    for nv in [4, 5]:
        experiments.append({
            'name': f'win3_nv{nv}_s42',
            'window': 3, 'V': 81, 'seed': 42,
            'budget': 24000, 'ticks': 8, 'nv_ratio': nv,
        })
    # Tick variants for window=3
    for ticks in [4, 16]:
        experiments.append({
            'name': f'win3_t{ticks}_s42',
            'window': 3, 'V': 81, 'seed': 42,
            'budget': 24000, 'ticks': ticks,
        })
    return experiments


def main():
    train_text = load_and_clean('pride_prejudice.txt')[:50000]
    test_text = load_and_clean('alice.txt')[:10000]

    # Baselines
    transitions = Counter()
    for i in range(len(train_text) - 1):
        transitions[(train_text[i], train_text[i+1])] += 1
    best_next = {}
    for c in set(train_text):
        nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
        if nexts: best_next[c] = max(nexts, key=nexts.get)
    bigram_test = sum(1 for i in range(len(test_text)-1)
                      if best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)

    # Trigram baseline
    tri_trans = Counter()
    for i in range(len(train_text) - 2):
        tri_trans[(train_text[i], train_text[i+1], train_text[i+2])] += 1
    tri_best = {}
    for i in range(len(train_text) - 1):
        key = (train_text[i], train_text[i+1])
        nexts = {c3: cnt for (c1, c2, c3), cnt in tri_trans.items() if (c1,c2) == key}
        if nexts: tri_best[key] = max(nexts, key=nexts.get)
    trigram_test = sum(1 for i in range(len(test_text)-2)
                       if tri_best.get((test_text[i], test_text[i+1])) == test_text[i+2]) / (len(test_text)-2)

    all_exps = get_experiments()
    done = {fn.replace('.json', '') for fn in os.listdir(RESULTS_DIR) if fn.endswith('.json')}
    pending = [e for e in all_exps if e['name'] not in done]

    if not pending:
        print("ALL DONE!")
        print_summary(bigram_test, trigram_test)
        return

    exp = pending[0]
    print(f"{'='*60}")
    print(f"Running: {exp['name']} ({len(pending)} remaining)")
    print(f"Window={exp['window']} V={exp['V']} N={exp['V']*exp.get('nv_ratio',3)} ticks={exp['ticks']}")
    print(f"Bigram: {bigram_test*100:.1f}% | Trigram: {trigram_test*100:.1f}%")
    print(f"{'='*60}")

    window = exp['window']
    V = exp['V']
    nv = exp.get('nv_ratio', 3)
    ticks = exp['ticks']

    train_inputs, train_targets = make_windows_onehot(train_text, window)
    test_inputs, test_targets = make_windows_onehot(test_text, window)
    print(f"Train: {len(train_inputs)} windows | Test: {len(test_inputs)} windows")

    old_nv = SelfWiringGraph.NV_RATIO
    SelfWiringGraph.NV_RATIO = nv
    np.random.seed(exp['seed']); random.seed(exp['seed'])
    net = SelfWiringGraph(V)
    SelfWiringGraph.NV_RATIO = old_nv

    random.seed(exp['seed'] * 1000 + 1)
    t0 = time.time()
    best_acc, best_score, steps, traj = train_window(
        net, train_inputs, train_targets, exp['budget'], ticks=ticks)
    elapsed = time.time() - t0

    # Test
    test_acc, test_score = evaluate(net, test_inputs, test_targets, min(2000, len(test_inputs)))

    traj_str = " -> ".join(f"{a*100:.1f}" for _, a in traj)
    print(f"\nTrain: {best_acc*100:.1f}% Test: {test_acc*100:.1f}% "
          f"Steps: {steps} Time: {elapsed:.0f}s Conns: {net.count_connections()}")
    print(f"Trajectory: {traj_str}")

    result = {
        'name': exp['name'], 'window': window, 'V': V, 'N': net.N,
        'nv_ratio': nv, 'ticks': ticks, 'seed': exp['seed'],
        'budget': exp['budget'], 'train_acc': best_acc, 'test_acc': test_acc,
        'train_score': best_score, 'test_score': test_score,
        'steps': steps, 'elapsed': round(elapsed, 1),
        'trajectory': traj, 'conns': net.count_connections(),
        'bigram_test': bigram_test, 'trigram_test': trigram_test,
    }
    with open(os.path.join(RESULTS_DIR, f"{exp['name']}.json"), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved. Remaining: {len(pending)-1}")
    print_summary(bigram_test, trigram_test)


def print_summary(bigram_test, trigram_test):
    results = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                results.append(json.load(f))
    if not results:
        return
    print(f"\n{'='*70}")
    print(f"OVERNIGHT WINDOW RESULTS ({len(results)} done)")
    print(f"Bigram: {bigram_test*100:.1f}% | Trigram: {trigram_test*100:.1f}%")
    print(f"{'name':<25s} {'win':>4s} {'N':>5s} {'train':>7s} {'test':>7s} "
          f"{'vs_bi':>7s} {'vs_tri':>7s} {'time':>5s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: -x['test_acc']):
        d_bi = r['test_acc'] - bigram_test
        d_tri = r['test_acc'] - trigram_test
        marker = " <<<" if r['test_acc'] > trigram_test else (" <<" if r['test_acc'] > bigram_test else "")
        print(f"{r['name']:<25s} {r['window']:4d} {r['N']:5d} {r['train_acc']*100:6.1f}% "
              f"{r['test_acc']*100:6.1f}% {d_bi*100:+6.1f}pp {d_tri*100:+6.1f}pp "
              f"{r['elapsed']:4.0f}s{marker}")


if __name__ == '__main__':
    main()
