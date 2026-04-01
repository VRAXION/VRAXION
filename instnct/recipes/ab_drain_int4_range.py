"""
INSTNCT — Int4 Full Range: [1-6] vs [1-15] per-edge drain
===========================================================
Int4[1-6] won at 75.0%. But if we're using int4 anyway (like theta),
why not give the full [1-15] range? Does the network USE it?
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


def eval_crosstoken(net, seq, edge_drain_int):
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n, 255, dtype=np.uint8)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))
    ed = edge_drain_int[:n]
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
                if tick % REGEN_PERIOD == 0:
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


def run_arm(label, max_drain, seed, steps, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    n_edges = len(net.alive)
    drain = np.random.randint(1, max_drain + 1, size=n_edges).astype(np.int16)

    def resize():
        nonlocal drain
        n = len(net.alive)
        if n > len(drain):
            drain = np.concatenate([drain, np.random.randint(1, max_drain + 1, size=n - len(drain)).astype(np.int16)])
        elif n < len(drain):
            drain = drain[:n]

    def avg_acc():
        resize()
        return np.mean([eval_crosstoken(net, s, drain) for s in eval_seqs])

    best_acc = avg_acc()
    accepts = 0; stale = 0
    log_every = max(1, steps // 12)

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
        old_drain = drain.copy()
        net.mutate(); resize()

        for _ in range(random.randint(1, 5)):
            if len(drain) == 0: break
            idx = random.randint(0, len(drain) - 1)
            drain[idx] = np.clip(drain[idx] + random.choice([-1, 0, 1]), 1, max_drain)

        new_acc = avg_acc()
        if new_acc > best_acc:
            best_acc = new_acc; accepts += 1; stale = 0
        else:
            net.restore_state(snap); drain = old_drain; resize(); stale += 1

        if step % log_every == 0:
            print(f"    {label:>12} step {step:5d} | acc={best_acc:.3f} "
                  f"| drain={np.mean(drain):.1f}±{np.std(drain):.1f} [{drain.min()}-{drain.max()}] | acc#={accepts}")

    return best_acc, accepts, drain.copy()


def main():
    master_rng = np.random.RandomState(77)
    eval_seqs = [make_alternating(master_rng, 40) for _ in range(3)]
    STEPS = 3000; SEED = 42

    print("=" * 85)
    print("  Int4 Full Range: [1-6] vs [1-15] per-edge drain")
    print(f"  H={H} | Steps={STEPS} | Regen fixed={REGEN_PERIOD}")
    print("=" * 85)

    arms = [
        ("Int4[1-6]", 6),
        ("Int4[1-15]", 15),
    ]

    results = []
    for label, max_d in arms:
        print(f"\n>>> {label}")
        t0 = time.time()
        acc, accepts, drain = run_arm(label, max_d, SEED, STEPS, eval_seqs)
        print(f"    Done in {time.time()-t0:.0f}s")
        results.append((label, acc, accepts, drain, max_d))

    print(f"\n{'='*85}")
    print(f"  RESULTS")
    print(f"{'='*85}")
    for label, acc, accepts, drain, _ in results:
        print(f"  {label:>12}: acc={acc:.3f} | drain={np.mean(drain):.1f}±{np.std(drain):.1f} "
              f"| range=[{drain.min()},{drain.max()}] | acc#={accepts}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  Winner: {best[0]} ({best[1]:.3f})")

    # Distribution analysis
    for label, _, _, drain, max_d in results:
        vals, counts = np.unique(drain, return_counts=True)
        print(f"\n  {label} distribution:")
        for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
            bar = "█" * max(1, c // 3)
            print(f"    drain={v:2d}: {c:3d} edges {bar}")

        # Did it USE the full range?
        used = len(vals)
        print(f"    Used {used}/{max_d} values ({100*used/max_d:.0f}%)")
        p = np.percentile(drain, [10, 25, 50, 75, 90])
        print(f"    Percentiles: p10={p[0]:.0f} p25={p[1]:.0f} p50={p[2]:.0f} p75={p[3]:.0f} p90={p[4]:.0f}")


if __name__ == "__main__":
    main()
