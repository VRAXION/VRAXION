"""
INSTNCT — Fan Inhibitory Hub Test
===================================
Does fan motif work when the hub is inhibitory?

A) Loop (baseline winner from previous test)
B) Fan with random hub (mostly excitatory — previous result: +1.3%)
C) Fan with inhibitory hub only (fewer but stronger, FlyWire-aligned)

Quick test on L2 Alt plateau.
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


def insert_loop(net):
    H = net.H
    length = random.randint(3, 5)
    nodes = random.sample(range(H), min(length, H))
    added = 0
    for i in range(len(nodes)):
        r, c = nodes[i], nodes[(i + 1) % len(nodes)]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True
            added += 1
    net.resync_alive()
    return added


def insert_fan_random(net, fan_out=3):
    """Fan with random hub (mostly excitatory at 90%)."""
    H = net.H
    nodes = random.sample(range(H), min(2 + fan_out, H))
    hub, sink = nodes[0], nodes[1]
    added = 0
    for m in nodes[2:]:
        if not net.mask[hub, m]:
            net.mask[hub, m] = True; added += 1
        if not net.mask[m, sink]:
            net.mask[m, sink] = True; added += 1
    net.resync_alive()
    return added


def insert_fan_inhibitory(net, fan_out=3):
    """Fan where hub MUST be inhibitory neuron.
    Inhibitory = polarity False. These have 2x fan-out (FlyWire).
    Creates a strong brake that sees many neurons.
    """
    H = net.H
    # Find inhibitory neurons
    inhib_idx = np.where(~net.polarity)[0]
    if len(inhib_idx) == 0:
        return 0  # no inhibitory neurons available

    hub = int(random.choice(inhib_idx))
    # Pick random targets (mix of excit and inhib)
    candidates = [n for n in range(H) if n != hub]
    if len(candidates) < fan_out + 1:
        return 0
    targets = random.sample(candidates, fan_out + 1)
    sink = targets[0]
    middles = targets[1:]

    added = 0
    for m in middles:
        if not net.mask[hub, m]:
            net.mask[hub, m] = True; added += 1
        if not net.mask[m, sink]:
            net.mask[m, sink] = True; added += 1
    net.resync_alive()
    return added


def insert_loop_plus_inhib_fan(net):
    """Best of both: loop for memory + inhibitory fan for regulation."""
    a = insert_loop(net)
    b = insert_fan_inhibitory(net, fan_out=random.randint(2, 4))
    return a + b


def run_arm(label, insert_fn, seed, steps, eval_sets):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=64, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)

    def avg_acc():
        return np.mean([eval_net(net, [s]) for s in eval_sets])

    best_acc = avg_acc()
    accepts = 0; stale = 0; injections = 0
    history = []
    log_every = max(1, steps // 20)

    for step in range(1, steps + 1):
        if stale >= PLATEAU_WINDOW and insert_fn is not None:
            insert_fn(net)
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
            history.append((step, best_acc, len(net.alive), accepts, injections))
            print(f"    {label:>12} step {step:5d} | acc={best_acc:.3f} | edges={len(net.alive):4d} "
                  f"| acc#={accepts}{inj_str}")

    return best_acc, accepts, injections, len(net.alive), history


def main():
    master_rng = np.random.RandomState(99)
    eval_sets = [make_alternating(master_rng) for _ in range(5)]
    STEPS = 6000
    SEED = 42

    print("=" * 85)
    print("  Fan Inhibitory Hub Test: does fan work with inhibitory hub?")
    print(f"  Task: L2 Alt | H=64 | Plateau: {PLATEAU_WINDOW} steps | Budget: {STEPS}")
    print("=" * 85)

    arms = [
        ("Loop",        insert_loop),
        ("Fan(random)", insert_fan_random),
        ("Fan(inhib)",  lambda net: insert_fan_inhibitory(net, fan_out=random.randint(2, 4))),
        ("Loop+InhFan", insert_loop_plus_inhib_fan),
    ]

    results = []
    for label, fn in arms:
        print(f"\n>>> {label}")
        t0 = time.time()
        acc, accepts, injects, edges, hist = run_arm(label, fn, SEED, STEPS, eval_sets)
        print(f"    Done in {time.time()-t0:.1f}s")
        results.append((label, acc, accepts, injects, edges))

    print(f"\n{'='*85}")
    print(f"  RESULTS")
    print(f"{'='*85}")
    print(f"  {'':>14} | {'Acc':>5} | {'Acc#':>4} | {'Inj':>3} | {'Edges':>5}")
    for label, acc, accepts, injects, edges in results:
        print(f"  {label:>14} | {acc:5.3f} | {accepts:4d} | {injects:3d} | {edges:5d}")

    best = max(results, key=lambda x: x[1])
    base = results[0][1]  # loop as reference
    print(f"\n  Best: {best[0]} ({best[1]:.3f})")
    for label, acc, _, _, _ in results:
        d = acc - base
        print(f"    {label}: {d:+.3f} vs Loop")


if __name__ == "__main__":
    main()
