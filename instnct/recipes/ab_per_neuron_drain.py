"""
INSTNCT — Per-Neuron Drain: global vs per-neuron learnable
============================================================
A) Global drain (single value, mutates)
B) Per-neuron drain (like theta — each neuron has own drain rate)
Both: regen fixed at 2 ticks (converged value from previous sweep)

If per-neuron drain differentiates (hub neurons ≠ peripheral) → needed.
If all neurons converge to same drain → global is fine.
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
REGEN_PERIOD = 2  # fixed (converged from learnable sweep)
H = 64


def make_alternating(rng, n=40):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]


def eval_crosstoken(net, seq, drain_per_neuron, regen_period):
    """Eval with per-neuron drain. drain_per_neuron[source] = drain for that neuron's edges."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n_edges, 255, dtype=np.uint8)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))

    # Pre-compute per-edge drain from per-neuron drain
    edge_drain = np.round(drain_per_neuron[rows]).astype(np.int16)
    edge_drain = np.clip(edge_drain, 1, 15)

    correct = 0; total = 0
    rp = max(1, int(round(regen_period)))

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
            # Per-edge drain based on source neuron's drain rate
            if len(rows):
                fs = fired[rows]
                if np.any(fs):
                    stamina[fs] = np.clip(stamina[fs].astype(np.int16) - edge_drain[fs], 0, 255).astype(np.uint8)
        state = act; charge = cur_charge
        if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def run_arm(label, mode, seed, steps, eval_seqs):
    """mode='global' or 'per_neuron'"""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    rng = random.Random(seed)

    if mode == 'global':
        drain_pn = np.full(H, rng.uniform(1.0, 4.0), dtype=np.float32)
    else:
        drain_pn = np.random.uniform(0.5, 5.0, size=H).astype(np.float32)

    def avg_acc():
        return np.mean([eval_crosstoken(net, s, drain_pn, REGEN_PERIOD) for s in eval_seqs])

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
        old_drain = drain_pn.copy()
        net.mutate()

        # Mutate drain
        if mode == 'global':
            if random.random() < 0.2:
                drain_pn[:] = max(0.1, drain_pn[0] + random.gauss(0, 0.3))
        else:
            # Per-neuron: mutate 1-3 random neurons' drain
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, H - 1)
                drain_pn[idx] = max(0.1, drain_pn[idx] + random.gauss(0, 0.5))

        new_acc = avg_acc()
        if new_acc > best_acc:
            best_acc = new_acc; accepts += 1; stale = 0
        else:
            net.restore_state(snap)
            drain_pn[:] = old_drain
            stale += 1

        if step % log_every == 0:
            d_mean = float(np.mean(drain_pn))
            d_std = float(np.std(drain_pn))
            print(f"    {label:>12} step {step:5d} | acc={best_acc:.3f} "
                  f"| drain={d_mean:.2f}±{d_std:.2f} | acc#={accepts}")

    return best_acc, accepts, drain_pn.copy()


def main():
    master_rng = np.random.RandomState(77)
    eval_seqs = [make_alternating(master_rng, 40) for _ in range(3)]
    STEPS = 3000; SEED = 42

    print("=" * 85)
    print("  Per-Neuron Drain: global vs per-neuron learnable")
    print(f"  H={H} | Steps={STEPS} | Regen fixed={REGEN_PERIOD} ticks")
    print("=" * 85)

    # A) Global
    print(f"\n>>> A) Global drain")
    t0 = time.time()
    acc_a, accepts_a, drain_a = run_arm("Global", 'global', SEED, STEPS, eval_seqs)
    ta = time.time() - t0
    print(f"    Done in {ta:.0f}s: acc={acc_a:.3f}, drain={np.mean(drain_a):.2f}")

    # B) Per-neuron
    print(f"\n>>> B) Per-neuron drain")
    t0 = time.time()
    acc_b, accepts_b, drain_b = run_arm("Per-neuron", 'per_neuron', SEED, STEPS, eval_seqs)
    tb = time.time() - t0
    print(f"    Done in {tb:.0f}s: acc={acc_b:.3f}, drain mean={np.mean(drain_b):.2f}±{np.std(drain_b):.2f}")

    # Analysis
    print(f"\n{'='*85}")
    print(f"  RESULTS")
    print(f"{'='*85}")
    print(f"  {'Global':>12}: acc={acc_a:.3f}, drain={np.mean(drain_a):.2f} (uniform)")
    print(f"  {'Per-neuron':>12}: acc={acc_b:.3f}, drain={np.mean(drain_b):.2f}±{np.std(drain_b):.2f}")

    delta = acc_b - acc_a
    print(f"\n  Delta: {delta:+.3f} {'(PER-NEURON WINS)' if delta > 0.005 else '(GLOBAL WINS)' if delta < -0.005 else '(TIE)'}")

    # Did per-neuron drain differentiate?
    if np.std(drain_b) > 0.3:
        print(f"\n  Per-neuron drain DIFFERENTIATED (std={np.std(drain_b):.2f})")
        # Check: do inhibitory neurons have different drain?
        inhib = ~net_b_polarity if 'net_b_polarity' in dir() else np.zeros(H, dtype=bool)
        print(f"  Drain distribution: min={drain_b.min():.2f}, max={drain_b.max():.2f}")
        print(f"  Top-5 drain: {sorted(drain_b)[-5:]}")
        print(f"  Bottom-5 drain: {sorted(drain_b)[:5]}")
    else:
        print(f"\n  Per-neuron drain CONVERGED to ~same value (std={np.std(drain_b):.2f})")
        print(f"  → Global drain is sufficient")


if __name__ == "__main__":
    main()
