"""
INSTNCT — Ticks × Loop Length: burst test from plateau checkpoint
=================================================================
Load a plateaued L2 Alt checkpoint.
For each (ticks, loop_length) combo:
  1. Reload checkpoint (fresh copy)
  2. Burst-inject 100 loops of that length
  3. Eval at that tick count
  4. Report accuracy

No training — just injection + eval. Instant results.
"""
import sys, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256; INPUT_DURATION = 2
PRED_NEURONS = list(range(0, 10))
CKPT = "checkpoints/selfdir/self_directed_l2_alt.npz"
N_LOOPS = 100  # burst size


def make_alternating(rng, n=48):
    a, b = rng.randint(0, 10, size=2)
    while b == a:
        b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]


def eval_net(net, seqs, ticks):
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for seq in seqs:
        for i in range(len(seq) - 1):
            injected = net.input_projection[int(seq[i])]
            state, charge = SelfWiringGraph.rollout_token(
                injected, mask=net.mask, theta=net._theta_f32,
                decay=net.decay, ticks=ticks, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )
            pred_charges = charge[PRED_NEURONS] if max(PRED_NEURONS) < H else np.zeros(10)
            if int(np.argmax(pred_charges)) == int(seq[i + 1]):
                correct += 1
            total += 1
    return correct / total if total else 0.0


def burst_loops(net, n_loops, loop_len):
    """Insert n_loops loops of exact length. No selection, just inject."""
    H = net.H
    added_total = 0
    for _ in range(n_loops):
        if H < loop_len:
            continue
        nodes = random.sample(range(H), loop_len)
        for i in range(len(nodes)):
            r, c = nodes[i], nodes[(i + 1) % len(nodes)]
            if r != c and not net.mask[r, c]:
                net.mask[r, c] = True
                added_total += 1
    net.resync_alive()
    return added_total


def main():
    master_rng = np.random.RandomState(99)
    eval_sets = [make_alternating(master_rng) for _ in range(8)]

    # Baseline: load checkpoint, eval at ticks=8
    base_net = SelfWiringGraph.load(CKPT)
    base_acc = np.mean([eval_net(base_net, [s], 8) for s in eval_sets])
    print(f"=" * 80)
    print(f"  Ticks × Loop Length: burst {N_LOOPS} loops from plateau checkpoint")
    print(f"  Checkpoint: {CKPT} | H={base_net.H} | edges={len(base_net.alive)}")
    print(f"  Baseline (no loops, ticks=8): {base_acc:.3f}")
    print(f"=" * 80)

    tick_options = [8, 10, 12, 16, 24]
    loop_options = [2, 3, 4, 5, 8]

    # Header
    print(f"\n  {'':>10}", end="")
    for t in tick_options:
        print(f" | T={t:>2}", end="")
    print(f" |  cycles@T=8")
    print(f"  {'-'*10}", end="")
    for _ in tick_options:
        print(f"-+------", end="")
    print(f"-+-----------")

    for loop_len in loop_options:
        print(f"  Loop={loop_len:>4}", end="")
        for ticks in tick_options:
            random.seed(42); np.random.seed(42)
            net = SelfWiringGraph.load(CKPT)
            new_edges = burst_loops(net, N_LOOPS, loop_len)
            acc = np.mean([eval_net(net, [s], ticks) for s in eval_sets])
            marker = "*" if acc > base_acc + 0.05 else " "
            print(f" |{marker}{acc:.3f}", end="")
        cycles_at_8 = 8.0 / loop_len
        print(f" |  {cycles_at_8:.1f}")

    # Also test: no loops but different ticks (pure tick effect)
    print(f"\n  {'No loops':>10}", end="")
    for ticks in tick_options:
        net = SelfWiringGraph.load(CKPT)
        acc = np.mean([eval_net(net, [s], ticks) for s in eval_sets])
        print(f" | {acc:.3f}", end="")
    print(f" |  baseline")

    print(f"\n  * = >5% above baseline ({base_acc:.3f})")
    print(f"  {N_LOOPS} loops injected per cell (no training, just insertion + eval)")


if __name__ == "__main__":
    main()
