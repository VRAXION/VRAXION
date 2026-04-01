"""
INSTNCT — Per-Edge Drain: global vs per-edge learnable
========================================================
A) Global drain (1 value) — previous winner
B) Per-edge drain — each edge has own fatigue rate, mutates

Per-edge: drain stored alongside stamina, mutates like edge weights.
More targeted than per-neuron: specific connections tire at their
own rate based on selection pressure.
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
REGEN_PERIOD = 2; H = 64


def make_alternating(rng, n=40):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]


def eval_crosstoken(net, seq, edge_drain, regen_period):
    """Eval with per-edge drain rates."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n_edges, 255, dtype=np.uint8)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))
    rp = max(1, int(round(regen_period)))

    # Per-edge drain (int, clipped)
    ed = np.clip(np.round(edge_drain[:n_edges]).astype(np.int16), 1, 15)

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
                    stamina[fs] = np.clip(stamina[fs].astype(np.int16) - ed[fs], 0, 255).astype(np.uint8)
        state = act; charge = cur_charge
        if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def run_arm(label, mode, seed, steps, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    n_edges = len(net.alive)

    if mode == 'global':
        edge_drain = np.full(n_edges, random.uniform(1.0, 4.0), dtype=np.float32)
    else:
        edge_drain = np.random.uniform(0.5, 5.0, size=n_edges).astype(np.float32)

    def avg_acc():
        # Resize edge_drain if edges changed
        nonlocal edge_drain
        n = len(net.alive)
        if n > len(edge_drain):
            edge_drain = np.concatenate([edge_drain,
                np.full(n - len(edge_drain), np.mean(edge_drain), dtype=np.float32)])
        elif n < len(edge_drain):
            edge_drain = edge_drain[:n]
        return np.mean([eval_crosstoken(net, s, edge_drain, REGEN_PERIOD) for s in eval_seqs])

    best_acc = avg_acc()
    accepts = 0; stale = 0
    log_every = max(1, steps // 15)

    for step in range(1, steps + 1):
        if stale >= 400:
            nodes = random.sample(range(H), random.randint(3, 5))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0
            new = avg_acc()
            if new > best_acc: best_acc = new; accepts += 1

        snap = net.save_state()
        old_drain = edge_drain.copy()
        net.mutate()

        # Mutate drain
        n = len(net.alive)
        if n > len(edge_drain):
            edge_drain = np.concatenate([edge_drain,
                np.full(n - len(edge_drain), np.mean(edge_drain), dtype=np.float32)])
        elif n < len(edge_drain):
            edge_drain = edge_drain[:n]

        if mode == 'global':
            if random.random() < 0.2:
                edge_drain[:] = max(0.1, edge_drain[0] + random.gauss(0, 0.3))
        else:
            # Per-edge: mutate 1-5 random edges' drain
            n_mut = random.randint(1, min(5, len(edge_drain)))
            for _ in range(n_mut):
                idx = random.randint(0, len(edge_drain) - 1)
                edge_drain[idx] = max(0.1, edge_drain[idx] + random.gauss(0, 0.5))

        new_acc = avg_acc()
        if new_acc > best_acc:
            best_acc = new_acc; accepts += 1; stale = 0
        else:
            net.restore_state(snap)
            edge_drain = old_drain
            # Resize back if needed
            n2 = len(net.alive)
            if n2 != len(edge_drain):
                if n2 > len(edge_drain):
                    edge_drain = np.concatenate([edge_drain,
                        np.full(n2 - len(edge_drain), np.mean(edge_drain), dtype=np.float32)])
                else:
                    edge_drain = edge_drain[:n2]
            stale += 1

        if step % log_every == 0:
            d_mean = float(np.mean(edge_drain))
            d_std = float(np.std(edge_drain))
            print(f"    {label:>12} step {step:5d} | acc={best_acc:.3f} "
                  f"| drain={d_mean:.2f}±{d_std:.2f} | edges={len(net.alive)} | acc#={accepts}")

    return best_acc, accepts, edge_drain.copy(), net


def main():
    master_rng = np.random.RandomState(77)
    eval_seqs = [make_alternating(master_rng, 40) for _ in range(3)]
    STEPS = 3000; SEED = 42

    print("=" * 85)
    print("  Per-Edge Drain: global vs per-edge learnable")
    print(f"  H={H} | Steps={STEPS} | Regen fixed={REGEN_PERIOD} ticks")
    print("=" * 85)

    print(f"\n>>> A) Global drain")
    t0 = time.time()
    acc_a, accepts_a, drain_a, net_a = run_arm("Global", 'global', SEED, STEPS, eval_seqs)
    print(f"    Done in {time.time()-t0:.0f}s")

    print(f"\n>>> B) Per-edge drain")
    t0 = time.time()
    acc_b, accepts_b, drain_b, net_b = run_arm("Per-edge", 'per_edge', SEED, STEPS, eval_seqs)
    print(f"    Done in {time.time()-t0:.0f}s")

    print(f"\n{'='*85}")
    print(f"  RESULTS")
    print(f"{'='*85}")
    print(f"  {'Global':>12}: acc={acc_a:.3f}, drain={np.mean(drain_a):.2f} (uniform), edges={len(net_a.alive)}")
    print(f"  {'Per-edge':>12}: acc={acc_b:.3f}, drain={np.mean(drain_b):.2f}±{np.std(drain_b):.2f}, edges={len(net_b.alive)}")

    delta = acc_b - acc_a
    print(f"\n  Delta: {delta:+.3f} {'(PER-EDGE WINS)' if delta > 0.005 else '(GLOBAL WINS)' if delta < -0.005 else '(TIE)'}")

    if np.std(drain_b) > 0.3:
        print(f"\n  Per-edge drain distribution:")
        print(f"    min={drain_b.min():.2f}, max={drain_b.max():.2f}, mean={np.mean(drain_b):.2f}, std={np.std(drain_b):.2f}")
        # Quartiles
        q = np.percentile(drain_b, [10, 25, 50, 75, 90])
        print(f"    Percentiles: p10={q[0]:.2f} p25={q[1]:.2f} p50={q[2]:.2f} p75={q[3]:.2f} p90={q[4]:.2f}")
    else:
        print(f"\n  Per-edge drain converged: std={np.std(drain_b):.2f} → global is sufficient")


if __name__ == "__main__":
    main()
