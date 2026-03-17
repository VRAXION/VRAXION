"""Lateral inhibition sweep: can we prevent universal activation?

Three mechanisms:
  1. Top-K: only K strongest neurons fire per tick (per input)
  2. Divisive normalization: acts /= (1 + total_activity)
  3. Subtract-mean: acts = max(acts - mean(acts), 0)

Each tested:
  - Post-hoc on nets trained @6 ticks
  - Train from scratch @8, 10, 12 ticks
  - Signal trace @16 ticks
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from lib.utils import score_batch, train_cyclic

V = 16
SEEDS = [42, 77, 123]
TRAIN_BUDGET = 25000


# ── Custom forward with lateral inhibition ──

def forward_inhibited(net, ticks, inhibit=None, inhib_param=None, inject_window=1):
    """Forward with lateral inhibition.

    inhibit: 'topk', 'divisive', 'submean', None
    inhib_param: K for topk, alpha for divisive, None for submean
    inject_window: hold input for this many ticks
    """
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    inp = np.eye(V, dtype=np.float32)

    for t in range(ticks):
        if t < inject_window:
            acts[:, :V] = inp

        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)

        # ── Lateral inhibition ──
        if inhibit == 'topk' and inhib_param is not None:
            K = inhib_param
            # Per-row (per-input): keep only top K activations
            if acts.shape[1] > K:
                # Partition: find K-th largest per row
                thresholds = np.partition(acts, -K, axis=1)[:, -K]
                mask = acts >= thresholds[:, None]
                # Break ties: if more than K pass, zero the excess
                acts = acts * mask

        elif inhibit == 'divisive':
            alpha = inhib_param if inhib_param is not None else 1.0
            # Total activity per input normalizes all neurons
            total = acts.sum(axis=1, keepdims=True)
            acts = acts / (1.0 + alpha * total)

        elif inhibit == 'submean':
            # Subtract row mean, re-threshold
            row_mean = acts.mean(axis=1, keepdims=True)
            acts = np.maximum(acts - row_mean, 0.0)

        elif inhibit == 'softmax_gate':
            # Soft competition: scale acts by softmax of charges
            # Each neuron's activation is weighted by its relative strength
            ch_pos = np.maximum(charges, 0.0)
            total = ch_pos.sum(axis=1, keepdims=True) + 1e-8
            gate = ch_pos / total  # normalized "importance"
            acts = acts * gate * N  # scale up so total energy ~ preserved

        elif inhibit == 'topk_soft':
            # Soft top-K: zero out below-K, but don't clip the survivors
            K = inhib_param if inhib_param is not None else 8
            if acts.shape[1] > K:
                thresholds = np.partition(acts, -K, axis=1)[:, -K]
                mask = (acts >= thresholds[:, None]).astype(np.float32)
                acts = acts * mask

        charges = np.clip(charges, -1.0, 1.0)

    return charges[:, net.out_start:net.out_start + V]


def score_inhibited(net, targets, V, ticks, **kwargs):
    logits = forward_inhibited(net, ticks, **kwargs)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = (preds == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp, acc


def make_inhib_score_fn(**kwargs):
    def fn(net, targets, V, ticks):
        return score_inhibited(net, targets, V, ticks, **kwargs)
    return fn


# ═══════════════════════════════════════════════════════════════
#  Inhibition configs
# ═══════════════════════════════════════════════════════════════

configs = {
    'baseline':       dict(),
    'topk_4':         dict(inhibit='topk', inhib_param=4),
    'topk_6':         dict(inhibit='topk', inhib_param=6),
    'topk_8':         dict(inhibit='topk', inhib_param=8),
    'topk_12':        dict(inhibit='topk', inhib_param=12),
    'topk_16':        dict(inhibit='topk', inhib_param=16),
    'topk_24':        dict(inhibit='topk', inhib_param=24),
    'divisive_0.5':   dict(inhibit='divisive', inhib_param=0.5),
    'divisive_1.0':   dict(inhibit='divisive', inhib_param=1.0),
    'divisive_2.0':   dict(inhibit='divisive', inhib_param=2.0),
    'divisive_5.0':   dict(inhibit='divisive', inhib_param=5.0),
    'submean':        dict(inhibit='submean'),
    'softmax_gate':   dict(inhibit='softmax_gate'),
    'topk8_soft':     dict(inhibit='topk_soft', inhib_param=8),
    'topk12_soft':    dict(inhibit='topk_soft', inhib_param=12),
}

EVAL_TICKS = [4, 6, 8, 10, 12, 16]


# ═══════════════════════════════════════════════════════════════
#  PART 1: Post-hoc on nets trained @6 ticks
# ═══════════════════════════════════════════════════════════════

print("=" * 90)
print("  PART 1: LATERAL INHIBITION — Post-hoc (train@6, eval@various)")
print("=" * 90)

trained_nets = {}
for seed in SEEDS:
    np.random.seed(seed)
    targets = np.random.randint(0, V, size=V)
    net = SelfWiringGraph(V)
    train_cyclic(net, targets, V, score_fn=score_batch, ticks=6,
                 max_att=TRAIN_BUDGET, stale_limit=2000,
                 crystal_budget=3000, crystal_window=200,
                 crystal_min_rate=0.003, verbose=False)
    sc6, _ = score_batch(net, targets, V, ticks=6)
    trained_nets[seed] = (net, targets)
    print(f"  Seed {seed}: score@6={sc6*100:.1f}% conns={net.count_connections()}")

print(f"\n  {'config':>16s}", end='')
for t in EVAL_TICKS:
    print(f" | t={t:2d}  ", end='')
print(f" | mean  | stable?")
print(f"  {'-'*16}", end='')
for _ in EVAL_TICKS:
    print(f"-+-------", end='')
print(f"-+-------+--------")

for cfg_name, cfg_kwargs in configs.items():
    scores_by_tick = {t: [] for t in EVAL_TICKS}
    for seed in SEEDS:
        net, targets = trained_nets[seed]
        for t in EVAL_TICKS:
            sc, acc = score_inhibited(net, targets, V, t, **cfg_kwargs)
            scores_by_tick[t].append(sc)

    means = [np.mean(scores_by_tick[t]) for t in EVAL_TICKS]
    overall = np.mean(means)
    # Stability: ratio of worst to best tick score
    stability = min(means) / max(means) if max(means) > 0 else 0
    peak = max(means)

    print(f"  {cfg_name:>16s}", end='')
    for m in means:
        print(f" | {m*100:4.1f}%", end='')
    stable_str = "YES" if stability > 0.85 else ("~" if stability > 0.7 else "NO")
    print(f" | {overall*100:4.1f}% | {stable_str} ({stability:.2f})")


# ═══════════════════════════════════════════════════════════════
#  PART 2: Train from scratch with inhibition @8, 10, 12 ticks
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"  PART 2: Train from scratch with lateral inhibition")
print(f"{'='*90}")

# Pick the most promising configs from Part 1
train_configs = {
    'baseline':       dict(),
    'topk_6':         dict(inhibit='topk', inhib_param=6),
    'topk_8':         dict(inhibit='topk', inhib_param=8),
    'topk_12':        dict(inhibit='topk', inhib_param=12),
    'topk_16':        dict(inhibit='topk', inhib_param=16),
    'divisive_1.0':   dict(inhibit='divisive', inhib_param=1.0),
    'divisive_2.0':   dict(inhibit='divisive', inhib_param=2.0),
    'submean':        dict(inhibit='submean'),
    'softmax_gate':   dict(inhibit='softmax_gate'),
}

train_ticks = [6, 8, 10, 12]

print(f"\n  {'config':>16s}", end='')
for t in train_ticks:
    print(f" |  t={t:2d}   ", end='')
print()
print(f"  {'-'*16}", end='')
for _ in train_ticks:
    print(f"-+---------", end='')
print()

for cfg_name, cfg_kwargs in train_configs.items():
    row = []
    for t in train_ticks:
        seed_scores = []
        for seed in SEEDS:
            np.random.seed(seed)
            targets = np.random.randint(0, V, size=V)
            net = SelfWiringGraph(V)
            sfn = make_inhib_score_fn(**cfg_kwargs)
            sc, acc, kept, cyc = train_cyclic(
                net, targets, V, score_fn=sfn, ticks=t,
                max_att=TRAIN_BUDGET, stale_limit=2000,
                crystal_budget=3000, crystal_window=200,
                crystal_min_rate=0.003, verbose=False)
            seed_scores.append(sc)
        row.append(np.mean(seed_scores))

    print(f"  {cfg_name:>16s}", end='')
    for r in row:
        print(f" | {r*100:5.1f}%  ", end='')
    # Show degradation: how much does score drop from t=6 to t=12
    drop = row[0] - row[-1]
    print(f"  drop={drop*100:+.1f}%")


# ═══════════════════════════════════════════════════════════════
#  PART 3: Signal trace for best configs @16 ticks
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"  PART 3: Signal traces @16 ticks (trained from scratch @10 ticks)")
print(f"{'='*90}")

# Train fresh nets @10 ticks with each config
trace_configs = {
    'baseline':     dict(),
    'topk_8':       dict(inhibit='topk', inhib_param=8),
    'topk_12':      dict(inhibit='topk', inhib_param=12),
    'divisive_1.0': dict(inhibit='divisive', inhib_param=1.0),
    'submean':      dict(inhibit='submean'),
}

seed = 42
np.random.seed(seed)
targets = np.random.randint(0, V, size=V)

for cfg_name, cfg_kwargs in trace_configs.items():
    np.random.seed(seed)
    targets_t = np.random.randint(0, V, size=V)
    net = SelfWiringGraph(V)
    sfn = make_inhib_score_fn(**cfg_kwargs)
    sc, acc, kept, cyc = train_cyclic(
        net, targets_t, V, score_fn=sfn, ticks=10,
        max_att=TRAIN_BUDGET, stale_limit=2000,
        crystal_budget=3000, crystal_window=200,
        crystal_min_rate=0.003, verbose=False)

    print(f"\n  --- {cfg_name} (trained@10, score={sc*100:.1f}%, conns={net.count_connections()}) ---")
    print(f"  {'tick':>4s} {'active':>7s} {'mean_act':>9s} {'max_act':>9s} {'correct':>8s}")

    # Trace at 16 ticks
    N = net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)

    for t in range(16):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)

        # Apply inhibition
        if cfg_kwargs.get('inhibit') == 'topk':
            K = cfg_kwargs['inhib_param']
            if acts.shape[1] > K:
                th = np.partition(acts, -K, axis=1)[:, -K]
                acts = acts * (acts >= th[:, None])
        elif cfg_kwargs.get('inhibit') == 'divisive':
            alpha = cfg_kwargs.get('inhib_param', 1.0)
            total = acts.sum(axis=1, keepdims=True)
            acts = acts / (1.0 + alpha * total)
        elif cfg_kwargs.get('inhibit') == 'submean':
            row_mean = acts.mean(axis=1, keepdims=True)
            acts = np.maximum(acts - row_mean, 0.0)

        charges = np.clip(charges, -1.0, 1.0)

        active = (acts > 0).sum()
        mean_a = acts[acts > 0].mean() if active > 0 else 0
        max_a = acts.max()
        out_logits = charges[:, net.out_start:net.out_start + V]
        preds = np.argmax(out_logits, axis=1)
        correct = (preds == targets_t).sum()

        print(f"  {t:4d} {active:7d} {mean_a:9.4f} {max_a:9.4f} {correct:4d}/{V}")

    # Also show score at various tick cutoffs
    print(f"    Scores: ", end='')
    for tc in [6, 8, 10, 12, 16]:
        sc_t, acc_t = score_inhibited(net, targets_t, V, tc, **cfg_kwargs)
        print(f"t={tc}:{sc_t*100:.1f}% ", end='')
    print()
