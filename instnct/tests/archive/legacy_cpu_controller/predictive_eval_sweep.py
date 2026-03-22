"""
Predictive Eval Sweep — compare eval strategies for SelfWiringGraph.

Modes:
  BATCH       — stateless forward_batch (baseline)
  SEQ_BATCH   — 2-pass forward_batch (charge carries across passes)
  PRED_ERR    — 2-pass, fitness includes pass1→pass2 improvement
  NLL_BLEND   — accuracy + normalized negative log-likelihood
  CUMUL_STATE — charge/state persists BETWEEN mutation attempts
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from model.graph import SelfWiringGraph

V = 64
SEEDS = [42, 77, 123]
BUDGET = 4000
STALE = 3000
TICKS = 8
MODES = ['BATCH', 'SEQ_BATCH', 'PRED_ERR', 'NLL_BLEND', 'CUMUL_STATE']


def safe_softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _2pass_forward(net, V, ticks):
    """Run 2-pass batch forward, return (pass1_logits, pass2_logits)."""
    N = net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    eye = np.eye(V, dtype=np.float32)
    out = net.out_start
    results = []
    for p in range(2):
        for t in range(ticks):
            if t == 0:
                acts[:, :V] = eye
            raw = acts @ net.mask
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw
            charges *= retain
            acts = np.maximum(charges - net.THRESHOLD, 0.0)
            charges = np.clip(charges, -1.0, 1.0)
        results.append(charges[:, out:out + V].copy())
    return results


# ── Score functions ──

def score_batch(net, targets):
    logits = net.forward_batch(TICKS)
    probs = safe_softmax(logits)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp, acc


def score_seq_batch(net, targets):
    _, logits2 = _2pass_forward(net, V, TICKS)
    probs = safe_softmax(logits2)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp, acc


def score_pred_err(net, targets):
    logits1, logits2 = _2pass_forward(net, V, TICKS)
    probs1 = safe_softmax(logits1)
    probs2 = safe_softmax(logits2)
    tp1 = probs1[np.arange(V), targets[:V]].mean()
    tp2 = probs2[np.arange(V), targets[:V]].mean()
    acc = (np.argmax(probs2, axis=1)[:V] == targets[:V]).mean()
    improvement = max(0.0, tp2 - tp1)
    score = 0.3 * improvement + 0.4 * (0.5 * acc + 0.5 * tp2) + 0.3 * acc
    return score, acc


def score_nll_blend(net, targets):
    logits = net.forward_batch(TICKS)
    probs = safe_softmax(logits)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = np.clip(probs[np.arange(V), targets[:V]], 1e-10, 1.0)
    mean_nll = (-np.log(tp)).mean()
    norm_nll = 1.0 - min(mean_nll / np.log(V), 1.0)
    return 0.5 * acc + 0.5 * norm_nll, acc


def score_cumul(net, targets):
    N = net.N
    charges = np.tile(net.charge, (V, 1)).copy()
    acts = np.tile(net.state, (V, 1)).copy()
    retain = float(net.retention)
    eye = np.eye(V, dtype=np.float32)
    out = net.out_start
    for t in range(TICKS):
        if t == 0:
            acts[:, :V] += eye
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    net.charge = charges.mean(axis=0)
    net.state = acts.mean(axis=0)
    logits = charges[:, out:out + V]
    probs = safe_softmax(logits)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp, acc


SCORE_FN = {
    'BATCH': score_batch,
    'SEQ_BATCH': score_seq_batch,
    'PRED_ERR': score_pred_err,
    'NLL_BLEND': score_nll_blend,
    'CUMUL_STATE': score_cumul,
}


def train_one(mode, seed):
    np.random.seed(seed)
    random.seed(seed)
    targets = np.random.permutation(V).astype(np.int32)
    net = SelfWiringGraph(V)
    fn = SCORE_FN[mode]

    sc, acc = fn(net, targets)
    best_sc, best_acc = sc, acc
    stale = kept = 0

    t0 = time.time()
    for att in range(BUDGET):
        old_loss = int(net.loss_pct)
        if mode == 'CUMUL_STATE':
            old_ch, old_st = net.charge.copy(), net.state.copy()
        undo = net.mutate()
        new_sc, new_acc = fn(net, targets)
        if new_sc > sc:
            sc = new_sc
            if new_sc > best_sc:
                best_sc, best_acc = new_sc, new_acc
            kept += 1; stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            if mode == 'CUMUL_STATE':
                net.charge, net.state = old_ch, old_st
            stale += 1
            if random.randint(1, 20) <= 7:
                net.signal = np.int8(1 - int(net.signal))
            if random.randint(1, 20) <= 7:
                net.grow = np.int8(1 - int(net.grow))
        if best_sc >= 0.99 or stale >= STALE:
            break

    elapsed = time.time() - t0
    return best_sc, best_acc, kept, att + 1, net.count_connections(), elapsed


def main():
    print(f"Predictive Eval Sweep  V={V}  budget={BUDGET}  stale={STALE}")
    print(f"Seeds: {SEEDS}")
    print(f"{'─' * 75}")
    sys.stdout.flush()

    results = {}
    for mode in MODES:
        results[mode] = []
        for seed in SEEDS:
            sc, acc, kept, atts, conns, secs = train_one(mode, seed)
            results[mode].append((sc, acc, kept, atts, conns, secs))
            print(f"  {mode:12s}  seed={seed:3d}  acc={acc*100:5.1f}%  "
                  f"score={sc*100:5.1f}%  kept={kept:4d}  "
                  f"conns={conns:4d}  {secs:.1f}s")
            sys.stdout.flush()

    # Summary
    print(f"\n{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")
    print(f"  {'Mode':<14s}  {'Acc%':>8s}  {'Score%':>7s}  {'Kept':>5s}  "
          f"{'Conns':>5s}  {'Secs':>5s}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*5}")

    rankings = []
    for mode in MODES:
        runs = results[mode]
        accs = [r[1] for r in runs]
        scores = [r[0] for r in runs]
        kepts = [r[2] for r in runs]
        conns = [r[4] for r in runs]
        secs = [r[5] for r in runs]
        ma, ms = np.mean(accs)*100, np.mean(scores)*100
        sa = np.std(accs)*100
        print(f"  {mode:<14s}  {ma:5.1f}±{sa:4.1f}  {ms:6.1f}  "
              f"{np.mean(kepts):5.0f}  {np.mean(conns):5.0f}  "
              f"{np.mean(secs):5.1f}")
        rankings.append((mode, ms, ma, sa, np.mean(secs)))

    rankings.sort(key=lambda x: -x[1])
    print(f"\nRanking by score:")
    for i, (mode, sc, ac, std, secs) in enumerate(rankings):
        tag = " << BEST" if i == 0 else ""
        print(f"  {i+1}. {mode:<14s}  score={sc:5.1f}%  acc={ac:5.1f}±{std:.1f}%  "
              f"({secs:.0f}s){tag}")

    print(f"\n{'=' * 75}")
    print("Done.")


if __name__ == '__main__':
    main()
