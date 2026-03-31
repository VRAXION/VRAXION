"""
INSTNCT — Plateau Breaker A/B: Loop vs Motif insertion
=======================================================
When single-mutation hill climbing plateaus, try atomic multi-edge
insertions to cross valleys.

A) Loop: circular chain (A→B→C→A) — feedback circuit
B) Motif: fan-out hub (A→B, A→C, A→D) + convergence (B→E, C→E, D→E)
   This creates a "broadcast then collect" pattern.
C) Combo: alternate loop and motif

All three: train normally until plateau, then inject motif, continue.
Task: L2 Alt (alternating pattern) — the one that plateaus at ~34%.
"""
import sys, time, random, argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256
TICKS = 8
INPUT_DURATION = 2
PLATEAU_WINDOW = 500   # if no accept for this many steps → inject motif
PRED_NEURONS = list(range(0, 10))
CONF_NEURON = 10


def make_alternating(rng, n=48):
    a, b = rng.randint(0, 10, size=2)
    while b == a:
        b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]


def eval_net(net, seqs):
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
                decay=net.decay, ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )
            pred_charges = charge[PRED_NEURONS] if max(PRED_NEURONS) < H else np.zeros(10)
            if int(np.argmax(pred_charges)) == int(seq[i + 1]):
                correct += 1
            total += 1

    return correct / total if total else 0.0


# --- Motif insertions ---

def insert_loop(net, length=3):
    """Insert a circular loop: A→B→C→A. Atomic."""
    H = net.H
    nodes = random.sample(range(H), min(length, H))
    added = []
    for i in range(len(nodes)):
        r, c = nodes[i], nodes[(i + 1) % len(nodes)]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True
            added.append((r, c))
    net.resync_alive()
    return len(added)


def insert_fan_motif(net, fan_out=3):
    """Insert fan-out + convergence motif:
    Hub → [A, B, C] → Sink
    Creates broadcast-then-collect pattern.
    """
    H = net.H
    nodes = random.sample(range(H), min(2 + fan_out, H))
    hub = nodes[0]
    sink = nodes[1]
    middles = nodes[2:]
    added = []
    for m in middles:
        # hub → middle
        if not net.mask[hub, m]:
            net.mask[hub, m] = True
            added.append((hub, m))
        # middle → sink
        if not net.mask[m, sink]:
            net.mask[m, sink] = True
            added.append((m, sink))
    net.resync_alive()
    return len(added)


def insert_chain_motif(net, length=4):
    """Insert a directed chain: A→B→C→D (no loop back).
    Creates a delay line / sequential activation path.
    """
    H = net.H
    nodes = random.sample(range(H), min(length, H))
    added = []
    for i in range(len(nodes) - 1):
        r, c = nodes[i], nodes[i + 1]
        if not net.mask[r, c]:
            net.mask[r, c] = True
            added.append((r, c))
    net.resync_alive()
    return len(added)


def insert_combo(net, step_count):
    """Alternate between motif types."""
    choice = step_count % 3
    if choice == 0:
        return insert_loop(net, length=random.randint(3, 5)), "loop"
    elif choice == 1:
        return insert_fan_motif(net, fan_out=random.randint(2, 4)), "fan"
    else:
        return insert_chain_motif(net, length=random.randint(3, 5)), "chain"


# --- Training with plateau-triggered insertion ---

def run_arm(label, insert_fn, seed, steps, eval_sets):
    """Train with full mutate(), inject motifs at plateau."""
    rng_local = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=64, density=4,
        theta_init=1, decay_init=0.10, seed=seed,
    )

    def avg_acc():
        return np.mean([eval_net(net, [s]) for s in eval_sets])

    best_acc = avg_acc()
    accepts = 0
    stale = 0
    injections = 0
    history = []
    log_every = max(1, steps // 25)

    for step in range(1, steps + 1):
        # Plateau detection → inject motif
        if stale >= PLATEAU_WINDOW and insert_fn is not None:
            if insert_fn == 'loop':
                n = insert_loop(net, length=random.randint(3, 5))
            elif insert_fn == 'fan':
                n = insert_fan_motif(net, fan_out=random.randint(2, 4))
            elif insert_fn == 'combo':
                n, _ = insert_combo(net, injections)
            else:
                n = 0
            injections += 1
            stale = 0
            # Re-evaluate after injection
            new_acc = avg_acc()
            if new_acc > best_acc:
                best_acc = new_acc
                accepts += 1

        # Normal mutation
        state_snap = net.save_state()
        undo = net.mutate()
        new_acc = avg_acc()

        if new_acc > best_acc:
            best_acc = new_acc
            accepts += 1
            stale = 0
        else:
            net.restore_state(state_snap)
            stale += 1

        if step % log_every == 0:
            history.append((step, best_acc, len(net.alive), accepts, injections))

    return best_acc, accepts, injections, len(net.alive), history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    master_rng = np.random.RandomState(99)
    eval_sets = [make_alternating(master_rng) for _ in range(5)]

    print("=" * 95)
    print("  Plateau Breaker A/B: Loop vs Fan-Motif vs Combo")
    print(f"  Task: L2 Alt | H=64 | Plateau window: {PLATEAU_WINDOW} steps | Budget: {args.steps}")
    print("=" * 95)

    arms = [
        ("Baseline (no insert)", None),
        ("Loop (3-5 cycle)",     'loop'),
        ("Fan (hub→spread→sink)",'fan'),
        ("Combo (loop+fan+chain)",'combo'),
    ]

    results = []
    for label, insert_fn in arms:
        print(f"\n>>> {label}")
        t0 = time.time()
        acc, accepts, injects, edges, hist = run_arm(
            label, insert_fn, args.seed, args.steps, eval_sets
        )
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        for step, a, e, ac, inj in hist:
            inj_str = f" inj={inj}" if inj > 0 else ""
            print(f"    step {step:5d} | acc={a:.3f} | edges={e:4d} | acc#={ac}{inj_str}")
        results.append((label, acc, accepts, injects, edges))

    # Summary
    print(f"\n{'='*95}")
    print(f"  RESULTS — L2 Alternating, {args.steps} steps, H=64")
    print(f"{'='*95}")
    print(f"  {'':>28} | {'Acc':>5} | {'Accepts':>7} | {'Injects':>7} | {'Edges':>5}")
    for label, acc, accepts, injects, edges in results:
        print(f"  {label:>28} | {acc:5.3f} | {accepts:7d} | {injects:7d} | {edges:5d}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  Winner: {best[0]} ({best[1]:.3f})")
    baseline_acc = results[0][1]
    for label, acc, _, _, _ in results[1:]:
        d = acc - baseline_acc
        print(f"    {label}: {d:+.3f} vs baseline")


if __name__ == "__main__":
    main()
