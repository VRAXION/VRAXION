"""A/B test v2: Spike-based readout with real thresholds.

The v1 test was inconclusive because theta=0 makes spike and charge readout
nearly identical. This v2 test compares:

  A) theta=0 + charge readout    (current baseline)
  B) theta=2 + spike_count readout  (selective spikes, bio-inspired)
  C) theta=2 + charge readout       (control: does theta=2 help on its own?)
  D) theta=0 + spike_count readout  (control: does spike readout help at theta=0?)

Higher budget (500 steps) and theta mutation enabled for B variants.
"""

from __future__ import annotations

import sys
import os
import time
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.graph import SelfWiringGraph


def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


def rollout_tracked(act, charge, mask, theta, decay, ticks, injected, input_duration=2):
    """Rollout returning charge and spike_count."""
    H = mask.shape[0]
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    ret = 1.0 - decay

    spike_count = np.zeros(H, dtype=np.float32)

    for t in range(int(ticks)):
        if t < int(input_duration):
            act = act + injected

        raw = np.zeros(H, dtype=np.float32)
        if len(rs):
            np.add.at(raw, cs, act[rs] * sp_vals)
        charge += raw
        charge *= ret
        act = np.maximum(charge - theta, 0.0)
        charge = np.maximum(charge, 0.0)

        fired = charge >= theta
        spike_count += fired.astype(np.float32)

    return act, charge, spike_count


def eval_score(mask, H, theta, decay, text_bytes, bp, input_proj, output_proj,
               bigram, readout_mode="charge", ticks=8):
    """Bigram cosine eval."""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    total_cos = 0.0
    n = 0

    for i in range(len(text_bytes) - 1):
        act = state.copy()
        injected = bp[text_bytes[i]] @ input_proj

        act, charge, spike_count = rollout_tracked(
            act, charge, mask, theta, decay, ticks, injected
        )
        state = act.copy()

        rv = spike_count if readout_mode == "spike_count" else charge

        out = rv @ output_proj
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        e = np.exp(sims - sims.max())
        pred = e / e.sum()
        target_dist = bigram[text_bytes[i]]
        cos = np.dot(pred, target_dist) / (
            np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8
        )
        total_cos += cos
        n += 1
    return total_cos / n if n else 0.0


def eval_accuracy(mask, H, theta, decay, text_bytes, bp, input_proj, output_proj,
                  readout_mode="charge", ticks=8):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0
    total = 0

    for i in range(len(text_bytes) - 1):
        act = state.copy()
        injected = bp[text_bytes[i]] @ input_proj
        act, charge, spike_count = rollout_tracked(
            act, charge, mask, theta, decay, ticks, injected
        )
        state = act.copy()
        rv = spike_count if readout_mode == "spike_count" else charge
        out = rv @ output_proj
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i + 1]:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def run_search(config, H, bp, input_proj, output_proj, bigram, all_data,
               budget=500, seq_len=200, ticks=8, seed=42):
    """Hill-climbing with specific readout + theta config."""
    readout_mode = config['readout']
    theta_init = config['theta_init']
    theta_mutate = config.get('theta_mutate', False)

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, theta_init, dtype=np.float32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)

    if theta_mutate:
        schedule = ['add', 'add', 'flip', 'theta', 'decay', 'decay', 'decay', 'decay']
    else:
        schedule = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']

    def get_seqs():
        return [all_data[off:off + seq_len]
                for off in [np_rng.randint(0, len(all_data) - seq_len) for _ in range(2)]]

    seqs = get_seqs()
    best_score = np.mean([
        eval_score(mask, H, theta, decay, s, bp, input_proj, output_proj,
                   bigram, readout_mode, ticks)
        for s in seqs
    ])
    accepts = 0

    for step in range(1, budget + 1):
        ptype = schedule[(step - 1) % len(schedule)]
        rs, cs = np.where(mask != 0)
        alive = list(zip(rs.tolist(), cs.tolist()))
        if not alive and ptype != 'add':
            ptype = 'add'

        new_mask = mask.copy()
        new_theta = theta.copy()
        new_decay = decay.copy()
        valid = True

        if ptype == 'add':
            r = rng.randint(0, H - 1); c = rng.randint(0, H - 1)
            if r == c or mask[r, c] != 0:
                valid = False
            else:
                new_mask[r, c] = 1.0
        elif ptype == 'flip':
            if not alive:
                valid = False
            else:
                r, c = alive[rng.randint(0, len(alive) - 1)]
                nc = rng.randint(0, H - 1)
                if nc == r or nc == c or mask[r, nc] != 0:
                    valid = False
                else:
                    new_mask[r, c] = 0.0; new_mask[r, nc] = 1.0
        elif ptype == 'theta':
            idx = rng.randint(0, H - 1)
            new_theta[idx] = max(0.0, min(5.0, theta[idx] + rng.uniform(-0.3, 0.3)))
        elif ptype == 'decay':
            idx = rng.randint(0, H - 1)
            new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

        if not valid:
            continue

        seqs = get_seqs()
        new_score = np.mean([
            eval_score(new_mask, H, new_theta, new_decay, s, bp,
                       input_proj, output_proj, bigram, readout_mode, ticks)
            for s in seqs
        ])

        if new_score > best_score + 0.00005:
            mask = new_mask; theta = new_theta; decay = new_decay
            best_score = new_score; accepts += 1

        if step % 100 == 0:
            print(f"    step {step}: score={best_score:.6f}, edges={int(np.count_nonzero(mask))}, "
                  f"accepts={accepts}")
            sys.stdout.flush()

    return {
        'final_score': best_score,
        'accepts': accepts,
        'edges': int(np.count_nonzero(mask)),
        'mask': mask, 'theta': theta, 'decay': decay,
    }


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    BUDGET = 500; TICKS = 8; EVAL_LEN = 500

    print("=" * 70)
    print("  SPIKE READOUT A/B v2 — theta + readout interaction")
    print(f"  H={H}, budget={BUDGET}, ticks={TICKS}")
    print("=" * 70)
    sys.stdout.flush()

    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(BASE_DIR, "data")
    with open(os.path.join(data_dir, "pride_prejudice.txt"), "rb") as f:
        all_data = list(f.read())
    bigram = np.load(os.path.join(data_dir, "bigram_table.npy"))
    print(f"Data: {len(all_data)/1e6:.1f} MB")

    bp = make_bp(IO)
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO, hidden_ratio=NV, projection_scale=1.0)
    input_proj = ref.input_projection
    output_proj = ref.output_projection

    CONFIGS = {
        "A_charge_theta0": {"readout": "charge", "theta_init": 0.0},
        "B_spike_theta2_mut": {"readout": "spike_count", "theta_init": 2.0, "theta_mutate": True},
        "C_charge_theta2_mut": {"readout": "charge", "theta_init": 2.0, "theta_mutate": True},
        "D_spike_theta0": {"readout": "spike_count", "theta_init": 0.0},
    }

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n{'─'*60}")
        print(f"  {name}: readout={cfg['readout']}, theta_init={cfg['theta_init']}, "
              f"theta_mutate={cfg.get('theta_mutate', False)}")
        print(f"{'─'*60}")
        sys.stdout.flush()
        t0 = time.time()

        res = run_search(cfg, H, bp, input_proj, output_proj, bigram, all_data,
                         budget=BUDGET, ticks=TICKS, seed=42)
        elapsed = time.time() - t0

        # Eval accuracy
        eval_rng = np.random.RandomState(9999)
        off = eval_rng.randint(0, len(all_data) - EVAL_LEN)
        eval_bytes = all_data[off:off + EVAL_LEN]
        acc = eval_accuracy(res['mask'], H, res['theta'], res['decay'], eval_bytes,
                            bp, input_proj, output_proj, cfg['readout'], TICKS)

        res['accuracy'] = acc; res['time'] = elapsed
        results[name] = res

        print(f"  RESULT: score={res['final_score']:.6f}, acc={acc*100:.2f}%, "
              f"accepts={res['accepts']}, edges={res['edges']}, time={elapsed:.0f}s")
        sys.stdout.flush()

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'Score':>10s} {'Acc':>8s} {'Accepts':>8s} {'Edges':>6s} {'Time':>6s}")
    print("-" * 70)
    for name in CONFIGS:
        r = results[name]
        print(f"{name:<25s} {r['final_score']:10.6f} {r['accuracy']*100:7.2f}% "
              f"{r['accepts']:>7d} {r['edges']:>6d} {r['time']:>5.0f}s")

    baseline = results["A_charge_theta0"]["final_score"]
    baseline_acc = results["A_charge_theta0"]["accuracy"]
    print(f"\n  Deltas vs baseline A (score={baseline:.6f}, acc={baseline_acc*100:.2f}%):")
    for name in list(CONFIGS.keys())[1:]:
        r = results[name]
        ds = r['final_score'] - baseline
        da = (r['accuracy'] - baseline_acc) * 100
        print(f"    {name:<25s}: score {ds:+.6f}, acc {da:+.2f}pp")

    print(f"\n{'='*70}")
