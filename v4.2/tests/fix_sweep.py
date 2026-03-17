"""Exhaustive sweep: what fixes signal amplification at high tick counts?

Tests 3 strategies + combos:
  A) Forced decay (override loss_pct)
  B) Input re-injection window (hold input for W ticks, not just t=0)
  C) Acts clipping (cap activation amplitude)

Each strategy is tested across tick counts and seeds.
Networks are trained with ticks=6 (known good), then EVALUATED at various ticks
with each fix applied — this isolates the fix effect from training dynamics.

ALSO: trains from scratch with each fix to see if the fix helps training too.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from lib.utils import score_batch, train_cyclic

V = 16
SEEDS = [42, 77, 123]
TRAIN_TICKS = 6
EVAL_TICKS = [4, 6, 8, 10, 12, 16]
TRAIN_BUDGET = 25000


# ── Custom forward functions ──

def forward_batch_custom(net, ticks, force_retain=None, inject_window=1, act_clip=None):
    """Modified forward_batch with fix options.

    force_retain: override net.retention (e.g. 0.7 = 30% decay per tick)
    inject_window: number of ticks to hold input (1 = original, >1 = re-inject)
    act_clip: clip activations to this max value (None = no clip)
    """
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = force_retain if force_retain is not None else float(net.retention)
    inp = np.eye(V, dtype=np.float32)

    for t in range(ticks):
        if t < inject_window:
            acts[:, :V] = inp
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        if act_clip is not None:
            np.clip(acts, 0, act_clip, out=acts)
        charges = np.clip(charges, -1.0, 1.0)

    return charges[:, net.out_start:net.out_start + V]


def score_custom(net, targets, V, ticks, **kwargs):
    """Score using custom forward."""
    logits = forward_batch_custom(net, ticks, **kwargs)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = (preds == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp, acc


# ── Monkey-patch for training with custom forward ──

def make_score_fn(inject_window=1, act_clip=None):
    """Create a score_fn compatible with train_cyclic that uses custom forward."""
    def fn(net, targets, V, ticks):
        return score_custom(net, targets, V, ticks,
                            inject_window=inject_window, act_clip=act_clip)
    return fn


# ═══════════════════════════════════════════════════════════════
#  PART 1: Train with ticks=6, then EVALUATE with fixes at higher ticks
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("  PART 1: Post-hoc fixes (train@6 ticks, eval@various)")
print("=" * 80)

# Define fix configs
fixes = {
    'baseline':       dict(),
    'decay_70':       dict(force_retain=0.70),
    'decay_80':       dict(force_retain=0.80),
    'decay_85':       dict(force_retain=0.85),
    'decay_90':       dict(force_retain=0.90),
    'inject_2':       dict(inject_window=2),
    'inject_3':       dict(inject_window=3),
    'inject_4':       dict(inject_window=4),
    'inject_half':    dict(),  # inject_window = ticks//2, set per-eval
    'clip_0.5':       dict(act_clip=0.5),
    'clip_1.0':       dict(act_clip=1.0),
    'clip_2.0':       dict(act_clip=2.0),
    'decay85+inj2':   dict(force_retain=0.85, inject_window=2),
    'decay85+clip1':  dict(force_retain=0.85, act_clip=1.0),
    'decay90+inj2':   dict(force_retain=0.90, inject_window=2),
    'decay80+inj3':   dict(force_retain=0.80, inject_window=3),
    'inj2+clip1':     dict(inject_window=2, act_clip=1.0),
    'inj3+clip1':     dict(inject_window=3, act_clip=1.0),
    'd85+inj2+cl1':   dict(force_retain=0.85, inject_window=2, act_clip=1.0),
}

# Train baseline networks
trained_nets = {}
for seed in SEEDS:
    np.random.seed(seed)
    targets = np.random.randint(0, V, size=V)
    net = SelfWiringGraph(V)
    train_cyclic(net, targets, V, score_fn=score_batch, ticks=TRAIN_TICKS,
                 max_att=TRAIN_BUDGET, stale_limit=2000,
                 crystal_budget=3000, crystal_window=200,
                 crystal_min_rate=0.003, verbose=False)
    sc6, acc6 = score_batch(net, targets, V, ticks=TRAIN_TICKS)
    trained_nets[seed] = (net, targets)
    print(f"  Seed {seed}: trained score@6={sc6*100:.1f}% acc={acc6*100:.0f}% "
          f"conns={net.count_connections()} retain={net.retention:.3f}")

# Evaluate each fix
print(f"\n  {'fix':>18s}", end='')
for t in EVAL_TICKS:
    print(f" | t={t:2d}  ", end='')
print(f" | mean")
print(f"  {'-'*18}", end='')
for _ in EVAL_TICKS:
    print(f"-+-------", end='')
print(f"-+------")

for fix_name, fix_kwargs in fixes.items():
    scores_by_tick = {t: [] for t in EVAL_TICKS}

    for seed in SEEDS:
        net, targets = trained_nets[seed]

        for t in EVAL_TICKS:
            kw = fix_kwargs.copy()
            if fix_name == 'inject_half':
                kw['inject_window'] = max(1, t // 2)
            sc, acc = score_custom(net, targets, V, t, **kw)
            scores_by_tick[t].append(sc)

    print(f"  {fix_name:>18s}", end='')
    all_means = []
    for t in EVAL_TICKS:
        m = np.mean(scores_by_tick[t])
        all_means.append(m)
        print(f" | {m*100:4.1f}%", end='')
    overall = np.mean(all_means)
    print(f" | {overall*100:4.1f}%")


# ═══════════════════════════════════════════════════════════════
#  PART 2: Train FROM SCRATCH with each fix — does it help training?
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  PART 2: Train from scratch with fix active (ticks=8, 10, 12)")
print(f"{'='*80}")

train_fixes = {
    'baseline':     dict(inject_window=1, act_clip=None),
    'inject_2':     dict(inject_window=2, act_clip=None),
    'inject_3':     dict(inject_window=3, act_clip=None),
    'clip_1.0':     dict(inject_window=1, act_clip=1.0),
    'inj2+clip1':   dict(inject_window=2, act_clip=1.0),
    'inj3+clip1':   dict(inject_window=3, act_clip=1.0),
}

train_tick_vals = [8, 10, 12]

print(f"\n  {'fix':>14s}", end='')
for t in train_tick_vals:
    print(f" |  ticks={t:2d}  ", end='')
print()
print(f"  {'-'*14}", end='')
for _ in train_tick_vals:
    print(f"-+-----------", end='')
print()

for fix_name, fix_kwargs in train_fixes.items():
    results = []
    for t in train_tick_vals:
        seed_scores = []
        for seed in SEEDS:
            np.random.seed(seed)
            targets = np.random.randint(0, V, size=V)
            net = SelfWiringGraph(V)
            sfn = make_score_fn(**fix_kwargs)
            sc, acc, kept, cyc = train_cyclic(
                net, targets, V, score_fn=sfn, ticks=t,
                max_att=TRAIN_BUDGET, stale_limit=2000,
                crystal_budget=3000, crystal_window=200,
                crystal_min_rate=0.003, verbose=False)
            seed_scores.append(sc)
        mean_sc = np.mean(seed_scores)
        results.append(mean_sc)
        print(f"  {fix_name:>14s} |  {mean_sc*100:5.1f}%    ", end='') if len(results) == 1 else None

    print(f"  {fix_name:>14s}", end='')
    for r in results:
        print(f" |  {r*100:5.1f}%    ", end='')
    print()


# ═══════════════════════════════════════════════════════════════
#  PART 3: Signal trace with best fix
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  PART 3: Signal trace comparison (baseline vs best fixes) @ 16 ticks")
print(f"{'='*80}")

# Use seed 42 trained net
net, targets = trained_nets[42]

configs = {
    'baseline':     dict(),
    'inject_3':     dict(inject_window=3),
    'clip_1.0':     dict(act_clip=1.0),
    'inj3+clip1':   dict(inject_window=3, act_clip=1.0),
    'decay85+inj3': dict(force_retain=0.85, inject_window=3),
    'd85+inj3+cl1': dict(force_retain=0.85, inject_window=3, act_clip=1.0),
}

for cfg_name, cfg_kwargs in configs.items():
    print(f"\n  --- {cfg_name} ---")
    print(f"  {'tick':>4s} {'active':>7s} {'mean_act':>9s} {'max_act':>9s} {'correct':>8s}")

    N = net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = cfg_kwargs.get('force_retain', float(net.retention))
    inj_w = cfg_kwargs.get('inject_window', 1)
    a_clip = cfg_kwargs.get('act_clip', None)
    inp = np.eye(V, dtype=np.float32)

    for t in range(16):
        if t < inj_w:
            acts[:, :V] = inp
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        if a_clip is not None:
            np.clip(acts, 0, a_clip, out=acts)
        charges = np.clip(charges, -1.0, 1.0)

        active = (acts > 0).sum()
        mean_a = acts[acts > 0].mean() if active > 0 else 0
        max_a = acts.max()
        out_logits = charges[:, net.out_start:net.out_start + V]
        preds = np.argmax(out_logits, axis=1)
        correct = (preds == targets).sum()
        print(f"  {t:4d} {active:7d} {mean_a:9.2f} {max_a:9.2f} {correct:4d}/{V}")
