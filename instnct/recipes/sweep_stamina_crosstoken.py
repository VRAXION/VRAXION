"""
INSTNCT — Cross-Token Stamina: Eval-Only Sweep
================================================
Load a pre-trained checkpoint, evaluate with different stamina configs.
No training — just measure how stamina affects accuracy on a fixed graph.

Then: train briefly with the best config to see if it helps mutation.
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


def make_alternating(rng, n=60):
    a, b = rng.randint(0, 10, size=2)
    while b == a:
        b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]


def eval_crosstoken(net, seq, mode, drain_amt, regen_period, stamina_init):
    """Eval one sequence with persistent stamina."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n_edges, stamina_init, dtype=np.uint8) if mode != 'none' else None

    correct = 0; total = 0
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))

    for token_idx in range(len(seq) - 1):
        injected = net.input_projection[int(seq[token_idx])]
        # 8-tick rollout with stamina
        act = state.copy()
        cur_charge = charge.copy()

        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0:
                cur_charge = np.maximum(cur_charge - 1.0, 0.0)
            if tick < INPUT_DURATION:
                act = act + injected

            if stamina is not None and len(rows):
                if tick % regen_period == 0:
                    stamina[:] = np.clip(stamina.astype(np.int16) + 1, 0, 255).astype(np.uint8)
                if mode == 'continuous':
                    s_mult = stamina.astype(np.float32) / 255.0
                else:
                    s_mult = np.ones(n_edges, dtype=np.float32)
                    s_mult[stamina < 171] = 0.5
                    s_mult[stamina < 85] = 0.0
                raw = np.zeros(H, dtype=np.float32)
                np.add.at(raw, cols, act[rows] * s_mult)
            else:
                raw = np.zeros(H, dtype=np.float32)
                if len(rows):
                    np.add.at(raw, cols, act[rows])

            cur_charge += raw
            np.clip(cur_charge, 0.0, 15.0, out=cur_charge)
            eff_theta = np.clip(net._theta_f32 * SelfWiringGraph.WAVE_LUT[net.channel, tick % 8], 1.0, 15.0)
            can_fire = (refractory == 0)
            fired = (cur_charge >= eff_theta) & can_fire
            refractory[refractory > 0] -= 1
            refractory[fired] = 1
            act = fired.astype(np.float32) * net._polarity_f32
            cur_charge[fired] = 0.0

            if stamina is not None and len(rows):
                fs = fired[rows]
                if np.any(fs):
                    stamina[fs] = np.clip(stamina[fs].astype(np.int16) - drain_amt, 0, 255).astype(np.uint8)

        state = act
        charge = cur_charge

        pred = int(np.argmax(charge[PRED_NEURONS])) if max(PRED_NEURONS) < H else 0
        if pred == int(seq[token_idx + 1]):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    stam_info = (int(stamina.min()), int(stamina.max()), float(stamina.std()), float(np.mean(stamina))) if stamina is not None else None
    return acc, stam_info


def main():
    master_rng = np.random.RandomState(99)
    eval_seqs = [make_alternating(master_rng, n=60) for _ in range(5)]

    # Phase 1: Build a decent network first (no stamina, fast standard path)
    print("=" * 90)
    print("  Cross-Token Stamina Sweep (eval-only on trained graph)")
    print("=" * 90)

    SEED = 42; H = 64
    random.seed(SEED); np.random.seed(SEED)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=SEED)

    # Quick training with standard path + loop injection
    print("\n  Phase 1: Training base network (no stamina, standard rollout)...")
    t0 = time.time()

    def quick_eval(net):
        net.reset()
        sc = SelfWiringGraph.build_sparse_cache(net.mask)
        accs = []
        for seq in eval_seqs[:2]:
            state = np.zeros(H, dtype=np.float32)
            charge = np.zeros(H, dtype=np.float32)
            correct = 0; total = 0
            for i in range(len(seq) - 1):
                state, charge = SelfWiringGraph.rollout_token(
                    net.input_projection[int(seq[i])], mask=net.mask,
                    theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                    input_duration=INPUT_DURATION, state=state, charge=charge,
                    sparse_cache=sc, polarity=net._polarity_f32,
                    refractory=net.refractory, channel=net.channel)
                if int(np.argmax(charge[PRED_NEURONS])) == int(seq[i+1]):
                    correct += 1
                total += 1
            accs.append(correct/total if total else 0)
        return np.mean(accs)

    best_acc = quick_eval(net)
    accepts = 0; stale = 0
    for step in range(1, 4001):
        if stale >= 400:
            nodes = random.sample(range(H), random.randint(3, 5))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]:
                    net.mask[r, c] = True
            net.resync_alive()
            stale = 0
            new = quick_eval(net)
            if new > best_acc: best_acc = new; accepts += 1

        snap = net.save_state()
        net.mutate()
        new = quick_eval(net)
        if new > best_acc:
            best_acc = new; accepts += 1; stale = 0
        else:
            net.restore_state(snap); stale += 1

    print(f"  Done in {time.time()-t0:.0f}s: acc={best_acc:.3f}, edges={len(net.alive)}, accepts={accepts}")

    # Phase 2: Eval this trained network with different stamina configs
    print(f"\n  Phase 2: Evaluating stamina configs on trained graph...")

    configs = [
        ("No stamina",         'none',       0, 6, 255),
        ("Step d=1 r=6",       'stepped',    1, 6, 255),
        ("Step d=4 r=6",       'stepped',    4, 6, 255),
        ("Cont d=1 r=6",      'continuous',  1, 6, 255),
        ("Cont d=2 r=6",      'continuous',  2, 6, 255),
        ("Cont d=4 r=6",      'continuous',  4, 6, 255),
        ("Cont d=4 r=4",      'continuous',  4, 4, 255),
        ("Cont d=8 r=6",      'continuous',  8, 6, 255),
        ("Cont d=2 r=6 i128", 'continuous',  2, 6, 128),
        ("Cont d=4 r=6 i128", 'continuous',  4, 6, 128),
    ]

    results = []
    for label, mode, drain, regen, init in configs:
        accs = []
        stam_info = None
        for seq in eval_seqs:
            a, si = eval_crosstoken(net, seq, mode, drain, regen, init)
            accs.append(a)
            stam_info = si
        avg = np.mean(accs)
        stam_str = f"[{stam_info[0]},{stam_info[1]}] avg={stam_info[3]:.0f} std={stam_info[2]:.1f}" if stam_info else "N/A"
        print(f"    {label:>22}: acc={avg:.3f} | stamina={stam_str}")
        results.append((label, avg, stam_info))

    # Summary
    results.sort(key=lambda x: -x[1])
    print(f"\n{'='*90}")
    print(f"  RESULTS — sorted by accuracy")
    print(f"{'='*90}")
    baseline = [r for r in results if r[0] == "No stamina"][0][1]
    for label, acc, stam in results:
        delta = acc - baseline
        marker = " <<<" if acc == results[0][1] else ""
        stam_str = f"avg={stam[3]:.0f} std={stam[2]:.1f}" if stam else ""
        print(f"  {label:>22} | {acc:.3f} | {delta:+.3f} | {stam_str}{marker}")

    winner = results[0]
    print(f"\n  Winner: {winner[0]} ({winner[1]:.3f}, {winner[1]-baseline:+.3f} vs baseline)")


if __name__ == "__main__":
    main()
