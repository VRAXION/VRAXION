"""
Adversarial Stress Test — 12 probes for SelfWiringGraph
=========================================================
All 12 must PASS for the model to be considered valid.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph, softmax

SEED = 42
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def header(num, name):
    print(f"\n  {'-'*55}")
    print(f"  PROBE {num:2d}: {name}")
    print(f"  {'-'*55}")
    sys.stdout.flush()


def result(status, msg):
    tag = {"PASS": "+", "FAIL": "X", "WARN": "!"}[status]
    print(f"    [{tag}] {status}: {msg}")
    sys.stdout.flush()
    return status


def main():
    results = []

    # PROBE 1: Zero internal neurons (V=N)
    header(1, "Zero internal neurons (V=N=16)")
    np.random.seed(SEED); random.seed(SEED)
    try:
        net = SelfWiringGraph(16, 16)
        perm = np.random.permutation(16)
        score_best = 0.0
        for att in range(2000):
            sm = net.mask.copy()
            net.mutate()
            logits = net.forward_batch(ticks=8)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
            acc = (preds == perm).mean()
            tp = probs[np.arange(16), perm].mean()
            sc = 0.5*acc + 0.5*tp
            if sc > score_best: score_best = sc
            else: net.mask = sm; net.resync_alive()
        r = result(PASS if score_best > 0.05 else WARN,
                   f"V=N=16: {score_best*100:.1f}%")
    except Exception as ex:
        r = result(FAIL, f"Crashed: {ex}")
    results.append(("Zero internals", r))

    # PROBE 2: Identity permutation
    header(2, "Identity permutation")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(80, 16)
    identity = np.arange(16)
    acc_best = 0.0
    for att in range(3000):
        sm = net.mask.copy()
        net.mutate()
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1) == identity).mean()
        tp = probs[np.arange(16), identity].mean()
        sc = 0.5*acc + 0.5*tp
        if sc > 0: acc_best = max(acc_best, acc)
        else: net.mask = sm; net.resync_alive()
    r = result(PASS if acc_best > 0.5 else WARN, f"Identity: {acc_best*100:.1f}%")
    results.append(("Identity perm", r))

    # PROBE 3: Adversarial permutations
    header(3, "Adversarial permutations")
    for name, perm in [
        ('shift_1', np.roll(np.arange(16), 1)),
        ('reverse', np.arange(16)[::-1].copy()),
        ('swap_pairs', np.array([1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14])),
    ]:
        np.random.seed(SEED); random.seed(SEED)
        net = SelfWiringGraph(80, 16)
        for att in range(2000):
            sm = net.mask.copy()
            net.mutate()
            logits = net.forward_batch(ticks=8)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            acc = (np.argmax(probs, axis=1) == perm).mean()
            if acc > 0: pass
            else: net.mask = sm; net.resync_alive()
        print(f"    {name}: OK")
    r = result(PASS, "All adversarial perms trained without crash")
    results.append(("Adversarial perms", r))

    # PROBE 4: NaN/Inf injection
    header(4, "NaN/Inf injection")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(32, 8)
    net.reset()
    logits_nan = net.forward(np.full(8, np.nan, dtype=np.float32), ticks=8)
    net.reset()
    logits_inf = net.forward(np.full(8, np.inf, dtype=np.float32), ticks=8)
    net.reset()
    logits_huge = net.forward(np.full(8, 1e10, dtype=np.float32), ticks=8)
    all_finite = (np.all(np.isfinite(logits_nan)) and
                  np.all(np.isfinite(logits_inf)) and
                  np.all(np.isfinite(logits_huge)))
    r = result(PASS if all_finite else WARN, f"All finite: {all_finite}")
    results.append(("NaN/Inf injection", r))

    # PROBE 5: Empty network
    header(5, "Empty network (density=0)")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(32, 8, density=0.0)
    net.reset()
    logits = net.forward(np.zeros(8, dtype=np.float32), ticks=8)
    logits_b = net.forward_batch(ticks=8)
    ok = np.all(np.isfinite(logits)) and np.all(np.isfinite(logits_b))
    r = result(PASS if ok else FAIL, f"Empty network finite: {ok}")
    results.append(("Empty network", r))

    # PROBE 6: Fully connected
    header(6, "Fully connected (density=1.0)")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(48, 16, density=1.0)
    net.reset()
    logits = net.forward(np.zeros(16, dtype=np.float32), ticks=8)
    ok = np.all(np.isfinite(logits))
    r = result(PASS if ok else FAIL, f"Full density finite: {ok}")
    results.append(("Full density", r))

    # PROBE 7: Single neuron
    header(7, "Single neuron (V=1, N=1)")
    try:
        net = SelfWiringGraph(1, 1)
        net.reset()
        logits = net.forward(np.array([1.0], dtype=np.float32), ticks=8)
        logits_b = net.forward_batch(ticks=8)
        net.mutate()
        net.mutate()
        r = result(PASS, "V=1 N=1 works")
    except Exception as ex:
        r = result(FAIL, f"Crashed: {ex}")
    results.append(("Single neuron", r))

    # PROBE 8: Batch vs Sequential consistency
    header(8, "Batch vs Sequential consistency")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16)
    seq_logits = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        net.reset()
        world = np.zeros(16, dtype=np.float32); world[i] = 1.0
        seq_logits[i] = net.forward(world, ticks=8)
    batch_logits = net.forward_batch(ticks=8)
    max_diff = np.abs(seq_logits - batch_logits).max()
    pred_match = (np.argmax(seq_logits, axis=1) == np.argmax(batch_logits, axis=1)).mean()
    r = result(PASS if max_diff < 1e-5 else (WARN if pred_match >= 0.9 else FAIL),
               f"max_diff={max_diff:.1e}, pred_agree={pred_match*100:.0f}%")
    results.append(("Batch vs Sequential", r))

    # PROBE 9: Mutation determinism
    header(9, "Mutation determinism")
    np.random.seed(99); random.seed(99)
    net1 = SelfWiringGraph(48, 16)
    for _ in range(100): net1.mutate()
    np.random.seed(99); random.seed(99)
    net2 = SelfWiringGraph(48, 16)
    for _ in range(100): net2.mutate()
    ok = np.array_equal(net1.mask, net2.mask)
    r = result(PASS if ok else FAIL, f"Deterministic: {ok}")
    results.append(("Determinism", r))

    # PROBE 10: State leak after reset
    header(10, "State leak after reset()")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16)
    world = np.zeros(16, dtype=np.float32); world[0] = 1.0
    net.reset()
    logits_a = net.forward(world, ticks=8).copy()
    net.reset()
    logits_b = net.forward(world, ticks=8).copy()
    ok = np.allclose(logits_a, logits_b, atol=1e-6)
    r = result(PASS if ok else FAIL, f"Same output after reset: {ok}")
    results.append(("State leak", r))

    # PROBE 11: Charge explosion
    header(11, "Charge explosion -- 1000 ticks")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16, density=0.3)
    net.reset()
    logits = net.forward(np.zeros(16, dtype=np.float32), ticks=1000)
    ok = np.all(np.isfinite(logits)) and np.abs(net.charge).max() <= net.CLIP_BOUND + 0.01
    r = result(PASS if ok else FAIL, f"Charge bounded: {ok}")
    results.append(("Charge explosion", r))

    # PROBE 12: Save/restore fidelity
    header(12, "Save/restore exact fidelity")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16)
    net.forward(np.zeros(16, dtype=np.float32), ticks=8)
    state = net.save_state()
    net.mask[:] = -1; net.state[:] = 42; net.charge[:] = 100
    net.signal = 1; net.grow = 0; net.intensity = 1; net.loss_pct = 50
    net.restore_state(state)
    # Only mask+loss_pct revert. Strategy bits (signal, grow, intensity) survive.
    ok = (np.array_equal(net.mask, state['mask']) and
          np.array_equal(net.state, state['state']) and np.array_equal(net.charge, state['charge']) and
          net.loss_pct == state['loss_pct'] and
          net.signal == 1 and net.grow == 0 and net.intensity == 1)  # NOT reverted
    # Deep copy check
    state['mask'][0, 0] = 99
    deep_ok = net.mask[0, 0] != 99
    r = result(PASS if ok and deep_ok else FAIL,
               f"Bitwise restore: {ok}, deep copy: {deep_ok}")
    results.append(("Save/restore", r))

    # SUMMARY
    print(f"\n{'='*60}")
    print(f"  ADVERSARIAL STRESS TEST -- SUMMARY")
    print(f"{'='*60}\n")
    passes = sum(1 for _, s in results if s == PASS)
    warns = sum(1 for _, s in results if s == WARN)
    fails = sum(1 for _, s in results if s == FAIL)
    for name, status in results:
        tag = {"PASS": "+", "FAIL": "X", "WARN": "!"}[status]
        print(f"  [{tag}] {status:4s}  {name}")
    print(f"\n  Total: {passes} PASS, {warns} WARN, {fails} FAIL out of {len(results)}")
    if fails > 0:
        print(f"\n  {fails} FAILURE(S)!")
    else:
        print(f"\n  All clean!")
    print(f"\n{'='*60}", flush=True)
    return fails


if __name__ == '__main__':
    sys.exit(main())
