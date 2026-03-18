"""Daytime automated experiments: seq char LM + divnorm scaling.
Each invocation picks the next untested experiment and runs it.
Results saved to tests/results/daytime/"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'daytime')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)
ALPHA = 2.0


# ============================================================
# Data
# ============================================================

def load_and_clean(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    clean = ''.join(c if (c >= 'a' and c <= 'z') or c == ' ' else ' ' for c in text)
    return ' '.join(clean.split())


# ============================================================
# Forward pass variants
# ============================================================

def forward_divnorm_batch(net, ticks=8, alpha=ALPHA):
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


def forward_divnorm_seq(net, world, ticks=8, alpha=ALPHA):
    act = net.state.copy()
    for t in range(ticks):
        if t == 0:
            act[:net.V] = world
        raw = act @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        net.charge += raw
        total = np.abs(act).sum() + 1e-6
        act = np.maximum(net.charge - net.THRESHOLD, 0.0)
        act /= (1.0 + alpha * total)
        net.charge = np.clip(net.charge, -1.0, 1.0)
    net.state = act.copy()
    return net.charge[net.out_start:net.out_start + net.V]


# ============================================================
# Sequential evaluation
# ============================================================

def evaluate_seq(net, text, char2idx, ticks=8, max_chars=200):
    net.reset()
    correct = 0
    V = net.V
    length = min(len(text) - 1, max_chars)
    for i in range(length):
        world = np.zeros(V, dtype=np.float32)
        world[char2idx[text[i]]] = 1.0
        output = forward_divnorm_seq(net, world, ticks)
        pred = np.argmax(output)
        if pred == char2idx[text[i + 1]]:
            correct += 1
    return correct / length


# ============================================================
# Permutation evaluation
# ============================================================

def evaluate_perm(net, targets, ticks=8):
    logits = forward_divnorm_batch(net, ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


# ============================================================
# Training loops
# ============================================================

def train_seq_experiment(net, train_text, char2idx, budget, ticks, seq_len,
                         stale_limit=8000, log_every=4000):
    eval_text = train_text[:seq_len + 1]

    def evaluate():
        return evaluate_seq(net, eval_text, char2idx, ticks, seq_len)

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


def train_perm_experiment(net, targets, budget, ticks,
                          stale_limit=15000, log_every=8000):
    score = evaluate_perm(net, targets, ticks)
    best = score
    stale = 0
    trajectory = [(0, float(best))]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate_perm(net, targets, ticks)

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


# ============================================================
# Experiment definitions
# ============================================================

def get_experiments():
    experiments = []

    # Part A: Sequential char LM — shorter seq_len for faster training
    for seq_len in [50, 100, 200]:
        for seed in [0, 42]:
            experiments.append({
                'name': f'seq_{seq_len}_t8_s{seed}',
                'type': 'seq', 'seq_len': seq_len,
                'ticks': 8, 'seed': seed, 'budget': 48000,
            })
    # Tick variants at seq_len=100
    for ticks in [4, 16]:
        experiments.append({
            'name': f'seq_100_t{ticks}_s42',
            'type': 'seq', 'seq_len': 100,
            'ticks': ticks, 'seed': 42, 'budget': 48000,
        })

    # Part B: Divnorm permutation scaling
    for V in [64, 128]:
        for ticks in [8, 16]:
            experiments.append({
                'name': f'perm_v{V}_divnorm_t{ticks}_96k',
                'type': 'perm', 'V': V,
                'ticks': ticks, 'seed': 42, 'budget': 96000,
                'stale_limit': 15000,
            })
    # V=64 extra long run
    experiments.append({
        'name': 'perm_v64_divnorm_t8_200k',
        'type': 'perm', 'V': 64,
        'ticks': 8, 'seed': 42, 'budget': 200000,
        'stale_limit': 30000,
    })

    # Part C: NV_RATIO exploration
    for nv in [4, 5]:
        experiments.append({
            'name': f'perm_v64_nv{nv}_divnorm_t8',
            'type': 'perm', 'V': 64, 'nv_ratio': nv,
            'ticks': 8, 'seed': 42, 'budget': 96000,
            'stale_limit': 15000,
        })

    return experiments


# ============================================================
# Main
# ============================================================

def main():
    all_exps = get_experiments()
    done = {fn.replace('.json', '') for fn in os.listdir(RESULTS_DIR) if fn.endswith('.json')}
    pending = [e for e in all_exps if e['name'] not in done]

    if not pending:
        print("ALL EXPERIMENTS DONE!")
        print_summary()
        return

    exp = pending[0]
    print(f"{'='*60}")
    print(f"Running: {exp['name']} ({len(pending)} remaining)")
    print(f"{'='*60}")

    seed = exp['seed']
    t0 = time.time()

    if exp['type'] == 'seq':
        # Load text data
        train_text = load_and_clean('pride_prejudice.txt') + ' ' + load_and_clean('frankenstein.txt')
        test_text = load_and_clean('alice.txt')
        chars = sorted(set(train_text + test_text))
        char2idx = {c: i for i, c in enumerate(chars)}
        V = len(chars)

        # Bigram baseline
        transitions = Counter()
        for i in range(len(train_text) - 1):
            transitions[(train_text[i], train_text[i+1])] += 1
        best_next = {}
        for c in chars:
            nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
            if nexts: best_next[c] = max(nexts, key=nexts.get)
            else: best_next[c] = ' '
        bigram_test = sum(1 for i in range(len(test_text)-1)
                         if best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)

        np.random.seed(seed); random.seed(seed)
        net = SelfWiringGraph(V)
        random.seed(seed * 1000 + 1)

        best, steps, traj = train_seq_experiment(
            net, train_text, char2idx, exp['budget'],
            exp['ticks'], exp['seq_len'])
        elapsed = time.time() - t0

        # Test on Alice
        test_accs = []
        sl = exp['seq_len']
        for start in range(0, min(len(test_text)-sl-1, 5000), sl):
            acc = evaluate_seq(net, test_text[start:start+sl+1], char2idx, exp['ticks'], sl)
            test_accs.append(acc)
        test_avg = np.mean(test_accs) if test_accs else 0

        result = {
            'name': exp['name'], 'type': 'seq',
            'V': V, 'seq_len': exp['seq_len'], 'ticks': exp['ticks'],
            'seed': seed, 'budget': exp['budget'],
            'train_best': best, 'test_acc': float(test_avg),
            'bigram_test': float(bigram_test),
            'steps': steps, 'elapsed': round(elapsed, 1),
            'trajectory': traj, 'conns': net.count_connections(),
        }
        print(f"Train: {best*100:.1f}% Test: {test_avg*100:.1f}% (bigram: {bigram_test*100:.1f}%)")

    elif exp['type'] == 'perm':
        V = exp['V']
        nv_ratio = exp.get('nv_ratio', 3)
        old_nv = SelfWiringGraph.NV_RATIO
        SelfWiringGraph.NV_RATIO = nv_ratio

        np.random.seed(seed); random.seed(seed)
        net = SelfWiringGraph(V)
        SelfWiringGraph.NV_RATIO = old_nv
        targets = np.arange(V)
        np.random.shuffle(targets)

        random.seed(seed * 1000 + 1)
        stale_limit = exp.get('stale_limit', 15000)
        best, steps, traj = train_perm_experiment(
            net, targets, exp['budget'], exp['ticks'], stale_limit)
        elapsed = time.time() - t0

        # Accuracy
        logits = forward_divnorm_batch(net, exp['ticks'])
        preds = np.argmax(logits, axis=1)
        acc = (preds[:V] == targets[:V]).mean()

        result = {
            'name': exp['name'], 'type': 'perm',
            'V': V, 'N': net.N, 'nv_ratio': nv_ratio,
            'ticks': exp['ticks'], 'seed': seed, 'budget': exp['budget'],
            'score': best, 'accuracy': float(acc),
            'steps': steps, 'elapsed': round(elapsed, 1),
            'trajectory': traj, 'conns': net.count_connections(),
        }
        print(f"Score: {best*100:.1f}% Acc: {acc*100:.1f}%")

    traj_str = " -> ".join(f"{b*100:.1f}" for _, b in traj)
    print(f"Steps: {steps} Time: {elapsed:.0f}s Conns: {net.count_connections()}")
    print(f"Trajectory: {traj_str}")

    path = os.path.join(RESULTS_DIR, f"{exp['name']}.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {path}")
    print(f"Remaining: {len(pending)-1}")
    print_summary()


def print_summary():
    results = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                results.append(json.load(f))
    if not results:
        return

    print(f"\n{'='*70}")
    print(f"DAYTIME RESULTS ({len(results)} done)")
    print(f"{'='*70}")

    seq_results = [r for r in results if r['type'] == 'seq']
    perm_results = [r for r in results if r['type'] == 'perm']

    if seq_results:
        print(f"\n--- Sequential Char LM ---")
        print(f"{'name':<25s} {'train':>7s} {'test':>7s} {'vs_bi':>7s} {'steps':>7s} {'time':>5s}")
        for r in sorted(seq_results, key=lambda x: -x['test_acc']):
            diff = r['test_acc'] - r['bigram_test']
            print(f"{r['name']:<25s} {r['train_best']*100:6.1f}% {r['test_acc']*100:6.1f}% "
                  f"{diff*100:+6.1f}pp {r['steps']:7d} {r['elapsed']:4.0f}s")

    if perm_results:
        print(f"\n--- Permutation Scaling ---")
        print(f"{'name':<35s} {'score':>7s} {'acc':>7s} {'steps':>7s} {'time':>5s} {'conns':>6s}")
        for r in sorted(perm_results, key=lambda x: -x['score']):
            print(f"{r['name']:<35s} {r['score']*100:6.1f}% {r['accuracy']*100:5.1f}% "
                  f"{r['steps']:7d} {r['elapsed']:4.0f}s {r['conns']:6d}")


if __name__ == '__main__':
    main()
