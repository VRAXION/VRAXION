"""
INSTNCT — Learnable Stamina: drain & regen as mutatable params
===============================================================
Make drain and regen_period LEARNABLE — they start random and
the mutation+selection loop finds the best values.

If all networks converge to the same drain/regen → physical constant.
If they diverge → needs to be per-network (or per-edge).

Extra budget: 5000 steps (more than usual) to let convergence happen.
5 different network seeds. Track drain/regen trajectory over time.
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
    """Eval with continuous stamina, cross-token."""
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

    rp = max(1, int(round(regen_period)))
    dr = max(1, int(round(drain)))

    for idx in range(len(seq) - 1):
        injected = net.input_projection[int(seq[idx])]
        act = state.copy(); cur_charge = charge.copy()
        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0:
                cur_charge = np.maximum(cur_charge - 1.0, 0.0)
            if tick < INPUT_DURATION:
                act = act + injected
            if len(rows):
                if tick % rp == 0:
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
                    stamina[fs] = np.clip(stamina[fs].astype(np.int16) - dr, 0, 255).astype(np.uint8)
        state = act; charge = cur_charge
        if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def run_learnable(H, density, seed, steps, eval_seqs):
    """Train with drain/regen as learnable params that mutate."""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=density,
                          theta_init=1, decay_init=0.10, seed=seed)

    # Learnable stamina params — start at random
    rng = random.Random(seed)
    drain = rng.uniform(0.5, 6.0)       # continuous, rounded for eval
    regen_period = rng.uniform(2.0, 10.0)  # continuous, rounded for eval

    def avg_acc():
        return np.mean([eval_crosstoken(net, s, drain, regen_period) for s in eval_seqs])

    best_acc = avg_acc()
    best_drain = drain
    best_regen = regen_period
    accepts = 0; stale = 0
    trajectory = []
    log_every = max(1, steps // 20)

    for step in range(1, steps + 1):
        # Loop injection at plateau
        if stale >= 400:
            nodes = random.sample(range(H), random.randint(3, 5))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0
            new = avg_acc()
            if new > best_acc: best_acc = new; accepts += 1

        # Save everything
        snap = net.save_state()
        old_drain = drain
        old_regen = regen_period

        # Mutate network (standard)
        net.mutate()

        # Mutate stamina params (1/3 chance each, small drift)
        if random.random() < 0.33:
            drain = max(0.1, drain + random.gauss(0, 0.5))
        if random.random() < 0.33:
            regen_period = max(1.0, regen_period + random.gauss(0, 1.0))

        new_acc = avg_acc()
        if new_acc > best_acc:
            best_acc = new_acc
            best_drain = drain
            best_regen = regen_period
            accepts += 1; stale = 0
        else:
            net.restore_state(snap)
            drain = old_drain
            regen_period = old_regen
            stale += 1

        if step % log_every == 0:
            trajectory.append((step, best_acc, best_drain, best_regen))

    return best_acc, best_drain, best_regen, accepts, trajectory


def main():
    master_rng = np.random.RandomState(77)
    eval_seqs = [make_alternating(master_rng, 40) for _ in range(3)]
    STEPS = 5000

    net_configs = [
        (64, 4, 42),
        (64, 4, 123),
        (64, 4, 999),
        (32, 4, 42),
        (64, 8, 42),
    ]

    print("=" * 95)
    print("  Learnable Stamina: drain & regen evolve via mutation")
    print(f"  Steps: {STEPS} (extra budget) | Drain/regen drift: gaussian ±0.5/±1.0")
    print(f"  5 networks, each finds its own optimal drain/regen")
    print("=" * 95)

    results = []
    for H, density, seed in net_configs:
        label = f"H={H} d={density}% s={seed}"
        print(f"\n  >>> {label}")
        t0 = time.time()
        acc, drain, regen, accepts, traj = run_learnable(
            H, density, seed, STEPS, eval_seqs
        )
        elapsed = time.time() - t0
        print(f"      acc={acc:.3f} | drain={drain:.2f} | regen={regen:.1f} "
              f"| acc#={accepts} | {elapsed:.0f}s")

        # Print trajectory
        for step, a, d, r in traj:
            print(f"      step {step:5d} | acc={a:.3f} | drain={d:.2f} | regen={r:.1f}")

        results.append((label, acc, drain, regen, accepts))

    # Convergence analysis
    print(f"\n{'='*95}")
    print(f"  CONVERGENCE ANALYSIS")
    print(f"{'='*95}")
    drains = [r[2] for r in results]
    regens = [r[3] for r in results]

    print(f"\n  {'Network':>20} | {'Drain':>6} | {'Regen':>6} | {'Acc':>5}")
    for label, acc, drain, regen, _ in results:
        print(f"  {label:>20} | {drain:6.2f} | {regen:6.1f} | {acc:5.3f}")

    drain_mean = np.mean(drains)
    drain_std = np.std(drains)
    regen_mean = np.mean(regens)
    regen_std = np.std(regens)

    print(f"\n  Drain:  mean={drain_mean:.2f} ± {drain_std:.2f}")
    print(f"  Regen:  mean={regen_mean:.1f} ± {regen_std:.1f}")

    # CV (coefficient of variation) — low = convergent
    drain_cv = drain_std / (drain_mean + 1e-6)
    regen_cv = regen_std / (regen_mean + 1e-6)
    print(f"\n  Drain CV: {drain_cv:.2f} ({'CONVERGED' if drain_cv < 0.3 else 'DIVERGED'})")
    print(f"  Regen CV: {regen_cv:.2f} ({'CONVERGED' if regen_cv < 0.3 else 'DIVERGED'})")

    if drain_cv < 0.3 and regen_cv < 0.3:
        print(f"\n  VERDICT: BOTH CONVERGE → drain≈{drain_mean:.1f}, regen≈{regen_mean:.0f}")
        print(f"  → Fix as physical constants, no need for learnable")
    elif drain_cv < 0.3:
        print(f"\n  VERDICT: Drain converges (≈{drain_mean:.1f}), regen diverges → regen maybe learnable")
    elif regen_cv < 0.3:
        print(f"\n  VERDICT: Regen converges (≈{regen_mean:.0f}), drain diverges → drain maybe learnable")
    else:
        print(f"\n  VERDICT: BOTH DIVERGE → need learnable stamina params")


if __name__ == "__main__":
    main()
