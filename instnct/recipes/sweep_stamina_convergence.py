"""
INSTNCT — Stamina Convergence: fixed constant or learnable?
=============================================================
If the optimal drain/regen is ALWAYS the same regardless of
network seed, size, density → it's a physical constant.
If it varies → it needs to be learnable.

Sweep: 5 different network seeds × 3 drain values × 2 regen values
on the same task. Check if the winner is always the same.
"""
import sys, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256; TICKS = 8; INPUT_DURATION = 2
PRED_NEURONS = list(range(0, 10))


def make_alternating(rng, n=40):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]


def eval_crosstoken(net, seq, drain, regen_period):
    """Eval with continuous stamina, cross-token persistent."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n_edges, 255, dtype=np.uint8)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))
    correct = 0; total = 0

    for idx in range(len(seq) - 1):
        injected = net.input_projection[int(seq[idx])]
        act = state.copy(); cur_charge = charge.copy()
        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0:
                cur_charge = np.maximum(cur_charge - 1.0, 0.0)
            if tick < INPUT_DURATION:
                act = act + injected
            if len(rows):
                if tick % regen_period == 0:
                    stamina[:] = np.clip(stamina.astype(np.int16) + 1, 0, 255).astype(np.uint8)
                s_mult = stamina.astype(np.float32) / 255.0
                raw = np.zeros(H, dtype=np.float32)
                np.add.at(raw, cols, act[rows] * s_mult)
            else:
                raw = np.zeros(H, dtype=np.float32)
            cur_charge += raw
            np.clip(cur_charge, 0.0, 15.0, out=cur_charge)
            eff_theta = np.clip(net._theta_f32 * SelfWiringGraph.WAVE_LUT[net.channel, tick % 8], 1.0, 15.0)
            can_fire = (refractory == 0)
            fired = (cur_charge >= eff_theta) & can_fire
            refractory[refractory > 0] -= 1; refractory[fired] = 1
            act = fired.astype(np.float32) * net._polarity_f32
            cur_charge[fired] = 0.0
            if len(rows):
                fs = fired[rows]
                if np.any(fs):
                    stamina[fs] = np.clip(stamina[fs].astype(np.int16) - drain, 0, 255).astype(np.uint8)
        state = act; charge = cur_charge
        if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def train_net(H, density, seed, steps=2000):
    """Train a network quickly with standard path + loop injection."""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=density,
                          theta_init=1, decay_init=0.10, seed=seed)
    train_seqs = [make_alternating(np.random.RandomState(seed + i), 30) for i in range(2)]

    def qeval():
        net.reset()
        sc = SelfWiringGraph.build_sparse_cache(net.mask)
        accs = []
        for seq in train_seqs:
            state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
            c = 0; t = 0
            for i in range(len(seq)-1):
                state, charge = SelfWiringGraph.rollout_token(
                    net.input_projection[int(seq[i])], mask=net.mask,
                    theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                    input_duration=INPUT_DURATION, state=state, charge=charge,
                    sparse_cache=sc, polarity=net._polarity_f32,
                    refractory=net.refractory, channel=net.channel)
                if int(np.argmax(charge[PRED_NEURONS])) == int(seq[i+1]): c += 1
                t += 1
            accs.append(c/t if t else 0)
        return np.mean(accs)

    best = qeval(); stale = 0
    for step in range(1, steps + 1):
        if stale >= 300:
            nodes = random.sample(range(H), random.randint(3, 5))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0
            new = qeval()
            if new > best: best = new
        snap = net.save_state(); net.mutate()
        new = qeval()
        if new > best: best = new; stale = 0
        else: net.restore_state(snap); stale += 1
    return net, best


def main():
    # Network configs to test
    net_configs = [
        (64, 4, 42),    # standard
        (64, 4, 123),   # different seed
        (64, 4, 999),   # another seed
        (32, 4, 42),    # smaller H
        (64, 8, 42),    # higher density
    ]

    drain_options = [1, 2, 4]
    regen_options = [4, 6]

    eval_rng = np.random.RandomState(77)
    eval_seqs = [make_alternating(eval_rng, 40) for _ in range(4)]

    print("=" * 90)
    print("  Stamina Convergence: do all networks prefer the same drain/regen?")
    print(f"  Drains: {drain_options} | Regens: {regen_options}")
    print("=" * 90)

    # Train all networks first
    nets = []
    for H, density, seed in net_configs:
        label = f"H={H} d={density}% s={seed}"
        print(f"\n  Training {label}...", end="", flush=True)
        t0 = time.time()
        net, acc = train_net(H, density, seed, steps=2000)
        print(f" acc={acc:.3f} edges={len(net.alive)} ({time.time()-t0:.0f}s)")
        nets.append((label, net))

    # Sweep stamina configs on each network
    print(f"\n  Sweeping stamina configs...")

    # Header
    configs = [(d, r) for d in drain_options for r in regen_options]
    config_labels = [f"d{d}r{r}" for d, r in configs]
    print(f"\n  {'Network':>20} |", " | ".join(f"{l:>5}" for l in config_labels), "| Winner")
    print(f"  {'-'*20}-+" + "-+-".join("-"*5 for _ in configs) + "-+-------")

    winners = {cl: 0 for cl in config_labels}

    for label, net in nets:
        results = {}
        for drain, regen in configs:
            cl = f"d{drain}r{regen}"
            accs = [eval_crosstoken(net, s, drain, regen) for s in eval_seqs]
            results[cl] = np.mean(accs)

        best_cl = max(results, key=results.get)
        winners[best_cl] += 1

        vals = " | ".join(f"{results[cl]:5.3f}" for cl in config_labels)
        print(f"  {label:>20} | {vals} | {best_cl}")

    # Summary
    print(f"\n{'='*90}")
    print(f"  CONVERGENCE ANALYSIS")
    print(f"{'='*90}")
    print(f"  Winner distribution:")
    for cl, count in sorted(winners.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {cl:>6}: {count}/{len(nets)} {bar}")

    # Is there a single winner?
    top = max(winners.values())
    top_configs = [cl for cl, c in winners.items() if c == top]

    if len(top_configs) == 1 and top == len(nets):
        print(f"\n  VERDICT: CONVERGES → {top_configs[0]} wins ALL configs")
        print(f"  → This is a PHYSICAL CONSTANT, not learnable")
    elif len(top_configs) == 1 and top >= len(nets) * 0.6:
        print(f"\n  VERDICT: MOSTLY CONVERGES → {top_configs[0]} wins {top}/{len(nets)}")
        print(f"  → Strong candidate for fixed constant")
    else:
        print(f"\n  VERDICT: DOES NOT CONVERGE → winners vary by network")
        print(f"  → Needs to be LEARNABLE (per-network or per-edge)")


if __name__ == "__main__":
    main()
