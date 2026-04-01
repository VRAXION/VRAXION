"""
INSTNCT — Stamina Resolution: how many bits for stamina itself?
================================================================
Currently stamina is uint8 [0-255] → 256 levels.
Test: uint2 (4 levels), uint4 (16), uint6 (64), uint8 (256).
Same per-edge drain[1-6], regen fixed 2 ticks.
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


def eval_crosstoken(net, seq, edge_drain, max_stamina):
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n, max_stamina, dtype=np.int16)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))
    ed = edge_drain[:n]

    # Scale regen to match resolution: +1 at uint8 = proportional at lower res
    # At uint8: regen +1/255 = 0.39% per regen tick
    # Keep same RATE: regen = max(1, max_stamina // 255)
    # Actually simpler: regen always +1, drain scaled proportionally
    regen_amt = max(1, round(max_stamina / 255))

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
                    stamina[:] = np.clip(stamina + regen_amt, 0, max_stamina)
                s_mult = stamina.astype(np.float32) / float(max_stamina)
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
                    # Scale drain proportionally to resolution
                    scaled_drain = np.clip(ed[fs] * max(1, round(max_stamina / 255)), 1, max_stamina)
                    stamina[fs] = np.clip(stamina[fs] - scaled_drain, 0, max_stamina)
        state = act; charge = cur_charge
        if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def run_arm(label, max_stamina, seed, steps, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    n_edges = len(net.alive)
    drain = np.random.randint(1, 7, size=n_edges).astype(np.int16)

    def resize():
        nonlocal drain
        n = len(net.alive)
        if n > len(drain):
            drain = np.concatenate([drain, np.random.randint(1, 7, size=n - len(drain)).astype(np.int16)])
        elif n < len(drain):
            drain = drain[:n]

    def avg_acc():
        resize()
        return np.mean([eval_crosstoken(net, s, drain, max_stamina) for s in eval_seqs])

    best_acc = avg_acc()
    accepts = 0; stale = 0

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
            drain[idx] = np.clip(drain[idx] + random.choice([-1, 0, 1]), 1, 6)

        new_acc = avg_acc()
        if new_acc > best_acc:
            best_acc = new_acc; accepts += 1; stale = 0
        else:
            net.restore_state(snap); drain = old_drain; resize(); stale += 1

    return best_acc, accepts


def main():
    master_rng = np.random.RandomState(77)
    eval_seqs = [make_alternating(master_rng, 40) for _ in range(3)]
    STEPS = 3000; SEED = 42

    print("=" * 70)
    print("  Stamina Resolution: how many bits?")
    print(f"  H={H} | Steps={STEPS} | Drain=[1-6] per-edge")
    print("=" * 70)

    arms = [
        ("uint2 (4 lvl)",     3),     # 0-3
        ("uint4 (16 lvl)",   15),     # 0-15
        ("uint6 (64 lvl)",   63),     # 0-63
        ("uint8 (256 lvl)", 255),     # 0-255
    ]

    results = []
    for label, max_s in arms:
        print(f"\n  {label}...", end="", flush=True)
        t0 = time.time()
        acc, accepts = run_arm(label, max_s, SEED, STEPS, eval_seqs)
        elapsed = time.time() - t0
        print(f" acc={acc:.3f} | acc#={accepts} | {elapsed:.0f}s")
        results.append((label, acc, accepts, max_s))

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    for label, acc, accepts, max_s in sorted(results, key=lambda x: -x[1]):
        bits = len(bin(max_s)) - 2
        marker = " <<<" if acc == max(r[1] for r in results) else ""
        print(f"  {label:>18}: acc={acc:.3f} | bits={bits} | acc#={accepts}{marker}")


if __name__ == "__main__":
    main()
