"""A/B test: Learnable per-neuron rho vs fixed rho

C19 Soft-Wave Canon modulates firing threshold:
  effective_theta = theta * (1 + rho * sin(tick * freq + phase))

Uses pre-trained checkpoint. Instead of argmax accuracy (too coarse),
we use softmax entropy (lower = more confident predictions) and
mean target probability as metrics.

Variants:
  A) rho=0 (no wave modulation — C19 disabled)
  B) rho=0.5 fixed (C19 with constant modulation)
  C) rho learnable from 0.5 (C19 with per-neuron optimization)
"""

from __future__ import annotations

import sys
import os
import time
import random
import statistics

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.graph import SelfWiringGraph


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def eval_net_detailed(net, text_bytes, ticks=8, seq_len=100):
    """Evaluate using multiple metrics: accuracy, entropy, target_prob."""
    V = net.V
    net.reset()
    correct = 0
    total = 0
    entropies = []
    target_probs = []
    logit_norms = []

    for i in range(min(seq_len, len(text_bytes) - 1)):
        inp = np.zeros(V, dtype=np.float32)
        inp[text_bytes[i] % V] = 1.0
        logits = net.forward(inp, ticks=ticks)

        logit_norms.append(float(np.linalg.norm(logits)))

        probs = softmax(logits)
        target = text_bytes[i + 1] % V

        # Accuracy
        if np.argmax(logits) == target:
            correct += 1
        total += 1

        # Target probability
        target_probs.append(float(probs[target]))

        # Entropy (lower = more confident)
        ent = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(float(ent))

    return {
        'accuracy': correct / total if total else 0.0,
        'mean_target_prob': float(np.mean(target_probs)),
        'mean_entropy': float(np.mean(entropies)),
        'mean_logit_norm': float(np.mean(logit_norms)),
    }


def run_trial(label, net, text_bytes, budget=400, ticks=8,
              freeze_rho=False, fixed_rho_val=None):
    """Run mutation search using target_prob as fitness."""

    if fixed_rho_val is not None:
        net.rho[:] = fixed_rho_val

    metrics = eval_net_detailed(net, text_bytes, ticks=ticks)
    best_score = metrics['mean_target_prob']
    best_metrics = metrics
    accepts = 0

    for step in range(1, budget + 1):
        state = net.save_state()
        undo = net.mutate()

        # If freezing rho, restore it after mutation
        if freeze_rho or fixed_rho_val is not None:
            net.rho[:] = state['rho']

        m = eval_net_detailed(net, text_bytes, ticks=ticks)
        score = m['mean_target_prob']

        if score > best_score:
            best_score = score
            best_metrics = m
            accepts += 1
        else:
            net.restore_state(state)

        if step % 100 == 0:
            print(f"    [{label}] step {step}: target_prob={best_score:.6f}, "
                  f"acc={best_metrics['accuracy']:.4f}, ent={best_metrics['mean_entropy']:.3f}, "
                  f"rho={net.rho.mean():.3f}±{net.rho.std():.3f}")
            sys.stdout.flush()

    return {
        **best_metrics,
        'accepts': accepts,
        'edges': net.count_connections(),
        'rho_mean': float(net.rho.mean()),
        'rho_std': float(net.rho.std()),
        'rho_min': float(net.rho.min()),
        'rho_max': float(net.rho.max()),
    }


if __name__ == "__main__":
    CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints",
                        "overnight_int4_brain_pruned.npz")
    BUDGET = 400
    TICKS = 8
    N_TRIALS = 3

    base_net = SelfWiringGraph.load(CKPT)
    H = base_net.H
    V = base_net.V

    print("=" * 70)
    print("  LEARNABLE RHO A/B TEST")
    print(f"  V={V}, H={H}, edges={base_net.count_connections()}")
    print(f"  budget={BUDGET}, ticks={TICKS}, trials={N_TRIALS}")
    print("=" * 70)

    # First: baseline eval with no training to check signal
    print("\n  Baseline check (no training):")
    for rho_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        net = SelfWiringGraph.load(CKPT)
        net.rho[:] = rho_val
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        with open(os.path.join(data_dir, "alice.txt"), "rb") as f:
            all_data = list(f.read())
        text_bytes = [b % V for b in all_data[:500]]
        m = eval_net_detailed(net, text_bytes, ticks=TICKS)
        print(f"    rho={rho_val:.2f}: target_prob={m['mean_target_prob']:.6f}, "
              f"acc={m['accuracy']:.4f}, entropy={m['mean_entropy']:.3f}, "
              f"logit_norm={m['mean_logit_norm']:.3f}")
    sys.stdout.flush()

    # Load text data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "alice.txt"), "rb") as f:
        all_data = list(f.read())
    text_bytes = [b % V for b in all_data[:500]]
    print(f"\nData: {len(text_bytes)} bytes (mapped to vocab {V})")

    CONFIGS = {
        "A_rho0_no_wave": {"fixed_rho_val": 0.0, "freeze_rho": True},
        "B_rho05_fixed": {"fixed_rho_val": 0.5, "freeze_rho": True},
        "C_rho_learnable": {"fixed_rho_val": None, "freeze_rho": False},
    }

    all_results = {name: [] for name in CONFIGS}

    for trial in range(N_TRIALS):
        seed = 42 + trial * 100
        print(f"\n{'─'*60}")
        print(f"  Trial {trial + 1}/{N_TRIALS} (seed={seed})")
        print(f"{'─'*60}")

        for name, cfg in CONFIGS.items():
            random.seed(seed)
            np.random.seed(seed)
            net = SelfWiringGraph.load(CKPT)

            t0 = time.time()
            res = run_trial(
                name, net, text_bytes, budget=BUDGET, ticks=TICKS, **cfg
            )
            res['time'] = time.time() - t0
            all_results[name].append(res)

            print(f"  {name}: tp={res['mean_target_prob']:.6f}, "
                  f"acc={res['accuracy']:.4f}, ent={res['mean_entropy']:.3f}, "
                  f"rho={res['rho_mean']:.3f}±{res['rho_std']:.3f}, "
                  f"accepts={res['accepts']}, time={res['time']:.0f}s")
            sys.stdout.flush()

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY: LEARNABLE RHO A/B TEST")
    print(f"{'='*70}")
    print(f"{'Config':<22s} {'TargetProb':>10s} {'Accuracy':>10s} {'Entropy':>8s} "
          f"{'Accepts':>8s} {'Rho':>8s}")
    print("-" * 70)
    for name in CONFIGS:
        tps = [r['mean_target_prob'] for r in all_results[name]]
        accs = [r['accuracy'] for r in all_results[name]]
        ents = [r['mean_entropy'] for r in all_results[name]]
        accepts = [r['accepts'] for r in all_results[name]]
        rhos = [r['rho_mean'] for r in all_results[name]]
        print(f"{name:<22s} {statistics.mean(tps):10.6f} {statistics.mean(accs)*100:9.2f}% "
              f"{statistics.mean(ents):>7.3f} {statistics.mean(accepts):>7.0f} "
              f"{statistics.mean(rhos):>7.3f}")

    baseline_tps = [r['mean_target_prob'] for r in all_results["A_rho0_no_wave"]]
    baseline_mean = statistics.mean(baseline_tps)
    print(f"\n  Deltas vs A_rho0 baseline (tp={baseline_mean:.6f}):")
    for name in list(CONFIGS.keys())[1:]:
        tps = [r['mean_target_prob'] for r in all_results[name]]
        delta = statistics.mean(tps) - baseline_mean
        pct = (delta / baseline_mean * 100) if baseline_mean > 0 else 0
        print(f"    {name:<22s}: tp {delta:+.6f} ({pct:+.2f}%)")

    print(f"\n  Rho evolution (learnable C only):")
    for i, r in enumerate(all_results["C_rho_learnable"]):
        print(f"    Trial {i+1}: mean={r['rho_mean']:.3f}, std={r['rho_std']:.3f}, "
              f"range=[{r['rho_min']:.3f}, {r['rho_max']:.3f}]")

    print(f"\n{'='*70}")
