"""A/B test: Spike-based readout vs charge-based readout.

Hypothesis: reading out from spike activity (how many times each neuron fired)
instead of the final charge vector might be more biologically plausible and
could capture different information. In the brain, only spikes are measured —
not membrane potential.

Variants:
  A) charge_readout  — current baseline: readout from final charge vector
  B) hard_spike_count — accumulate binary spike counts across ticks, readout from that
  C) soft_spike_accum — accumulate soft-thresholded activity across ticks
  D) spike_rate       — hard spike count normalized by number of ticks

We test on the English bigram task (same as main recipe).
"""

from __future__ import annotations

import sys
import os
import time
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.graph import SelfWiringGraph


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


def rollout_with_spike_tracking(
    act, charge, mask, theta, decay, ticks, injected, input_duration=2
):
    """Run rollout and return (final_act, final_charge, spike_count, soft_spike_accum).

    spike_count: how many times each neuron fired (binary threshold crossing)
    soft_spike_accum: sum of max(charge - theta, 0) across all ticks
    """
    H = mask.shape[0]
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    ret = 1.0 - decay

    spike_count = np.zeros(H, dtype=np.float32)
    soft_accum = np.zeros(H, dtype=np.float32)

    for t in range(int(ticks)):
        if t < int(input_duration):
            act = act + injected

        raw = np.zeros(H, dtype=np.float32)
        if len(rs):
            np.add.at(raw, cs, act[rs] * sp_vals)
        charge += raw
        charge *= ret

        # Soft threshold (current behavior)
        act = np.maximum(charge - theta, 0.0)
        charge = np.maximum(charge, 0.0)

        # Track spikes
        fired = (charge >= theta) if np.any(theta > 0) else (act > 0)
        spike_count += fired.astype(np.float32)
        soft_accum += act

    return act, charge, spike_count, soft_accum


def eval_bigram_variant(
    mask, H, theta, decay, text_bytes, bp, input_proj, output_proj,
    bigram, readout_mode="charge", ticks=8
):
    """Bigram cosine eval with selectable readout mode.

    readout_mode:
      'charge'          — readout from final charge (baseline)
      'hard_spike_count' — readout from spike count vector
      'soft_spike_accum' — readout from soft spike accumulator
      'spike_rate'       — readout from spike count / ticks
    """
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)

    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    total_cos = 0.0
    n = 0

    for i in range(len(text_bytes) - 1):
        act = state.copy()
        injected = bp[text_bytes[i]] @ input_proj

        act, charge, spike_count, soft_accum = rollout_with_spike_tracking(
            act, charge, mask, theta, decay, ticks, injected, input_duration=2
        )

        state = act.copy()

        # Select readout vector
        if readout_mode == "charge":
            readout_vec = charge
        elif readout_mode == "hard_spike_count":
            readout_vec = spike_count
        elif readout_mode == "soft_spike_accum":
            readout_vec = soft_accum
        elif readout_mode == "spike_rate":
            readout_vec = spike_count / ticks
        else:
            raise ValueError(f"Unknown readout_mode: {readout_mode}")

        out = readout_vec @ output_proj
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


def eval_accuracy_variant(
    mask, H, theta, decay, text_bytes, bp, input_proj, output_proj,
    readout_mode="charge", ticks=8
):
    """Classic next-byte accuracy with selectable readout mode."""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)

    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0
    total = 0

    for i in range(len(text_bytes) - 1):
        act = state.copy()
        injected = bp[text_bytes[i]] @ input_proj

        act, charge, spike_count, soft_accum = rollout_with_spike_tracking(
            act, charge, mask, theta, decay, ticks, injected, input_duration=2
        )

        state = act.copy()

        if readout_mode == "charge":
            readout_vec = charge
        elif readout_mode == "hard_spike_count":
            readout_vec = spike_count
        elif readout_mode == "soft_spike_accum":
            readout_vec = soft_accum
        elif readout_mode == "spike_rate":
            readout_vec = spike_count / ticks
        else:
            raise ValueError(f"Unknown readout_mode: {readout_mode}")

        out = readout_vec @ output_proj
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i + 1]:
            correct += 1
        total += 1

    return correct / total if total else 0.0


# ── Hill-climbing search per readout variant ────────────────────────────────

def search_with_readout(
    readout_mode, H, bp, input_proj, output_proj, bigram,
    all_data, budget=200, seq_len=200, ticks=8, seed=42
):
    """Run mutation search optimizing for a specific readout mode."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Init network (empty start, same as recipe)
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.zeros(H, dtype=np.float32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)

    schedule = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']

    def get_seqs():
        seqs = []
        for _ in range(2):
            off = np_rng.randint(0, len(all_data) - seq_len)
            seqs.append(all_data[off:off + seq_len])
        return seqs

    # Initial score
    seqs = get_seqs()
    best_score = np.mean([
        eval_bigram_variant(mask, H, theta, decay, s, bp, input_proj, output_proj, bigram,
                            readout_mode=readout_mode, ticks=ticks)
        for s in seqs
    ])
    accepts = 0
    scores = [best_score]

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
            r = rng.randint(0, H - 1)
            c = rng.randint(0, H - 1)
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
                    new_mask[r, c] = 0.0
                    new_mask[r, nc] = 1.0
        elif ptype == 'decay':
            idx = rng.randint(0, H - 1)
            new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

        if not valid:
            scores.append(best_score)
            continue

        seqs = get_seqs()
        new_score = np.mean([
            eval_bigram_variant(new_mask, H, new_theta, new_decay, s, bp,
                                input_proj, output_proj, bigram,
                                readout_mode=readout_mode, ticks=ticks)
            for s in seqs
        ])

        if new_score > best_score + 0.00005:
            mask = new_mask
            theta = new_theta
            decay = new_decay
            best_score = new_score
            accepts += 1

        scores.append(best_score)

    return {
        'final_score': best_score,
        'accepts': accepts,
        'edges': int(np.count_nonzero(mask)),
        'scores': scores,
        'mask': mask,
        'theta': theta,
        'decay': decay,
    }


# ── Main A/B test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    IO = 256
    NV = 4
    H = IO * NV  # 1024
    BUDGET = 200
    SEQ_LEN = 200
    TICKS = 8
    EVAL_LEN = 400

    print("=" * 70)
    print("  SPIKE READOUT A/B TEST")
    print("  Charge readout vs spike-based readout")
    print(f"  H={H}, budget={BUDGET}, ticks={TICKS}")
    print("=" * 70)
    sys.stdout.flush()

    # Load data (use local text files since fineweb may not be available)
    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(BASE_DIR, "data")
    text_path = os.path.join(data_dir, "pride_prejudice.txt")
    with open(text_path, "rb") as f:
        all_data = list(f.read())
    bigram_path = os.path.join(data_dir, "bigram_table.npy")
    bigram = np.load(bigram_path)
    print(f"Data: {len(all_data)/1e6:.1f} MB, bigram: {bigram.shape}")

    bp = make_bp(IO)

    # Build projections at scale=1.0
    random.seed(42)
    np.random.seed(42)
    ref = SelfWiringGraph(IO, hidden_ratio=NV, projection_scale=1.0)
    input_proj = ref.input_projection
    output_proj = ref.output_projection

    MODES = ["charge", "hard_spike_count", "soft_spike_accum", "spike_rate"]
    results = {}

    for mode in MODES:
        print(f"\n{'─'*60}")
        print(f"  Running: {mode}")
        print(f"{'─'*60}")
        sys.stdout.flush()
        t0 = time.time()

        res = search_with_readout(
            readout_mode=mode,
            H=H,
            bp=bp,
            input_proj=input_proj,
            output_proj=output_proj,
            bigram=bigram,
            all_data=all_data,
            budget=BUDGET,
            seq_len=SEQ_LEN,
            ticks=TICKS,
            seed=42,
        )
        elapsed = time.time() - t0

        # Final accuracy eval on longer sequence
        eval_rng = np.random.RandomState(9999)
        eval_bytes = all_data[eval_rng.randint(0, len(all_data) - EVAL_LEN):
                              eval_rng.randint(0, len(all_data) - EVAL_LEN) + EVAL_LEN]
        acc = eval_accuracy_variant(
            res['mask'], H, res['theta'], res['decay'], eval_bytes, bp,
            input_proj, output_proj, readout_mode=mode, ticks=TICKS
        )

        res['accuracy'] = acc
        res['time'] = elapsed
        results[mode] = res

        print(f"  Score: {res['final_score']:.6f}")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Accepts: {res['accepts']}/{BUDGET}")
        print(f"  Edges: {res['edges']}")
        print(f"  Time: {elapsed:.1f}s")
        sys.stdout.flush()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY: SPIKE READOUT A/B TEST")
    print(f"{'='*70}")
    print(f"{'Mode':<22s} {'Score':>10s} {'Accuracy':>10s} {'Accepts':>8s} {'Edges':>6s} {'Time':>6s}")
    print("-" * 70)
    for mode in MODES:
        r = results[mode]
        print(f"{mode:<22s} {r['final_score']:10.6f} {r['accuracy']*100:9.2f}% "
              f"{r['accepts']:>7d} {r['edges']:>6d} {r['time']:>5.1f}s")

    baseline = results["charge"]["final_score"]
    print(f"\n  Deltas vs charge baseline ({baseline:.6f}):")
    for mode in MODES[1:]:
        delta = results[mode]["final_score"] - baseline
        pct = (delta / abs(baseline) * 100) if baseline != 0 else 0
        acc_delta = (results[mode]["accuracy"] - results["charge"]["accuracy"]) * 100
        print(f"    {mode:<22s}: score {delta:+.6f} ({pct:+.2f}%), "
              f"acc {acc_delta:+.2f}pp")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")
