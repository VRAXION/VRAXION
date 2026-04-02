"""
INSTNCT — Brain Motifs A/B: Loop vs CPG vs Reservoir
=====================================================
A) Loop (3-5 cycle) — feedback circuit (previous winner: 64.2%)
B) CPG (reciprocal inhibition pair) — oscillator, inherent alternation
C) Dense cluster (mini reservoir) — 5-8 neurons all-to-all, rich dynamics
D) CPG + Loop combo

Task: L2 Alt | H=64 | Plateau-triggered insertion
"""
import sys, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256; TICKS = 8; INPUT_DURATION = 2
PLATEAU_WINDOW = 500
PRED_NEURONS = list(range(0, 10))


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

def insert_loop(net):
    """Circular feedback: A→B→C→A"""
    H = net.H
    length = random.randint(3, 5)
    nodes = random.sample(range(H), min(length, H))
    added = 0
    for i in range(len(nodes)):
        r, c = nodes[i], nodes[(i + 1) % len(nodes)]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True; added += 1
    net.resync_alive()
    return added, "loop"


def insert_cpg(net):
    """Central Pattern Generator: reciprocal inhibition pair.

    Pick one excitatory (E) and one inhibitory (I) neuron.
    Wire: E→I and I→E. This creates an oscillator:
    E fires → activates I → I inhibits E → E stops → I stops → E fires...

    Also wire both to 2 random downstream neurons for output.
    """
    H = net.H
    excit_idx = np.where(net.polarity)[0]   # True = excitatory
    inhib_idx = np.where(~net.polarity)[0]  # False = inhibitory

    if len(excit_idx) < 1 or len(inhib_idx) < 1:
        return 0, "cpg(skip)"

    e = int(random.choice(excit_idx))
    i = int(random.choice(inhib_idx))
    if e == i:
        return 0, "cpg(skip)"

    added = 0
    # Reciprocal connection
    if not net.mask[e, i]:
        net.mask[e, i] = True; added += 1
    if not net.mask[i, e]:
        net.mask[i, e] = True; added += 1

    # Downstream outputs (so the oscillation reaches the network)
    others = [n for n in range(H) if n != e and n != i]
    if len(others) >= 2:
        targets = random.sample(others, 2)
        for t in targets:
            if not net.mask[e, t]:
                net.mask[e, t] = True; added += 1
            if not net.mask[i, t]:
                net.mask[i, t] = True; added += 1

    net.resync_alive()
    return added, "cpg"


def insert_dense_cluster(net, size=6):
    """Mini reservoir: densely connected subgraph.

    Pick `size` neurons, connect all-to-all internally.
    Add 2 input edges (from random outsiders) and 2 output edges.
    Creates rich internal dynamics for the mutation to exploit.
    """
    H = net.H
    if H < size + 4:
        return 0, "cluster(skip)"

    cluster = random.sample(range(H), size)
    outsiders = [n for n in range(H) if n not in cluster]

    added = 0
    # Internal all-to-all (skip existing + self-loops)
    for a in cluster:
        for b in cluster:
            if a != b and not net.mask[a, b]:
                net.mask[a, b] = True; added += 1

    # 2 input edges from outside
    if len(outsiders) >= 2:
        inputs = random.sample(outsiders, 2)
        targets = random.sample(cluster, 2)
        for src, dst in zip(inputs, targets):
            if not net.mask[src, dst]:
                net.mask[src, dst] = True; added += 1

    # 2 output edges to outside
    if len(outsiders) >= 2:
        sources = random.sample(cluster, 2)
        outputs = random.sample(outsiders, 2)
        for src, dst in zip(sources, outputs):
            if not net.mask[src, dst]:
                net.mask[src, dst] = True; added += 1

    net.resync_alive()
    return added, "cluster"


def insert_cpg_plus_loop(net):
    """CPG for oscillation + loop for feedback. Best of both?"""
    a1, _ = insert_cpg(net)
    a2, _ = insert_loop(net)
    return a1 + a2, "cpg+loop"


# --- Training with plateau injection ---

def run_arm(label, insert_fn, seed, steps, eval_sets):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=64, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)

    def avg_acc():
        return np.mean([eval_net(net, [s]) for s in eval_sets])

    best_acc = avg_acc()
    accepts = 0; stale = 0; injections = 0
    log_every = max(1, steps // 20)

    for step in range(1, steps + 1):
        if stale >= PLATEAU_WINDOW and insert_fn is not None:
            n, mtype = insert_fn(net)
            injections += 1
            stale = 0
            new_acc = avg_acc()
            if new_acc > best_acc:
                best_acc = new_acc; accepts += 1

        state_snap = net.save_state()
        undo = net.mutate()
        new_acc = avg_acc()
        if new_acc > best_acc:
            best_acc = new_acc; accepts += 1; stale = 0
        else:
            net.restore_state(state_snap); stale += 1

        if step % log_every == 0:
            inj_str = f" inj={injections}" if injections > 0 else ""
            print(f"    {label:>10} step {step:5d} | acc={best_acc:.3f} "
                  f"| edges={len(net.alive):4d} | acc#={accepts}{inj_str}")

    return best_acc, accepts, injections, len(net.alive)


def main():
    master_rng = np.random.RandomState(99)
    eval_sets = [make_alternating(master_rng) for _ in range(5)]
    STEPS = 6000; SEED = 42

    print("=" * 85)
    print("  Brain Motifs A/B: Loop vs CPG vs Dense Cluster")
    print(f"  Task: L2 Alt | H=64 | Plateau: {PLATEAU_WINDOW} | Budget: {STEPS}")
    print("=" * 85)

    arms = [
        ("Loop",       insert_loop),
        ("CPG",        insert_cpg),
        ("Cluster(6)", lambda net: insert_dense_cluster(net, size=6)),
        ("CPG+Loop",   insert_cpg_plus_loop),
    ]

    results = []
    for label, fn in arms:
        print(f"\n>>> {label}")
        t0 = time.time()
        acc, accepts, injects, edges = run_arm(label, fn, SEED, STEPS, eval_sets)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        results.append((label, acc, accepts, injects, edges))

    print(f"\n{'='*85}")
    print(f"  RESULTS — L2 Alt, {STEPS} steps, H=64")
    print(f"{'='*85}")
    print(f"  {'':>12} | {'Acc':>5} | {'Acc#':>4} | {'Inj':>3} | {'Edges':>5}")
    for label, acc, accepts, injects, edges in results:
        print(f"  {label:>12} | {acc:5.3f} | {accepts:4d} | {injects:3d} | {edges:5d}")

    best = max(results, key=lambda x: x[1])
    loop_acc = results[0][1]
    print(f"\n  Winner: {best[0]} ({best[1]:.3f})")
    for label, acc, _, _, _ in results:
        d = acc - loop_acc
        print(f"    {label}: {d:+.3f} vs Loop")


if __name__ == "__main__":
    main()
