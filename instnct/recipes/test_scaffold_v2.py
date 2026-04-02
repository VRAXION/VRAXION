"""
INSTNCT — Scaffold v2: proper params, larger network, full mutate
==================================================================
Fix smoke test issues: theta=1, density=4% per layer, full mutate().
Frozen loops protected from add/remove but everything else mutates.

A) Scaffold → melt (the idea)
B) Flat baseline (same total H, same steps)
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


def make_cycle3(rng, n=30):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]

def make_alternating(rng, n=30):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]


def build_scaffold(L1, L2_loops, L3, seed):
    """Build scaffolded network with proper params."""
    L2 = L2_loops * 3
    H = L1 + L2 + L3

    # Use standard init: theta=1, decay=0.10, density will be manual
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    # net already has random 4% edges everywhere — keep them

    # ADD frozen loops in Layer 2 zone (on top of existing edges)
    frozen = set()
    base = L1
    for loop in range(L2_loops):
        a = base + loop * 3
        b = base + loop * 3 + 1
        c = base + loop * 3 + 2
        for r, col in [(a,b), (b,c), (c,a)]:
            if not net.mask[r, col]:
                net.mask[r, col] = True
            frozen.add((r, col))

    # Cross-loop chain
    for i in range(L2_loops - 1):
        src = base + i * 3
        dst = base + (i + 1) * 3
        if not net.mask[src, dst]:
            net.mask[src, dst] = True
        frozen.add((src, dst))

    # Global loop-back
    if L2_loops > 1:
        last = base + (L2_loops - 1) * 3
        if not net.mask[last, base]:
            net.mask[last, base] = True
        frozen.add((last, base))

    net.resync_alive()
    return net, frozen, (L1, L2, L3)


def eval_acc(net, seqs):
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    H = net.H
    st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
    c = 0; t = 0
    for seq in seqs:
        for i in range(len(seq) - 1):
            st, ch = SelfWiringGraph.rollout_token(
                net.input_projection[int(seq[i])], mask=net.mask,
                theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                input_duration=INPUT_DURATION, state=st, charge=ch,
                sparse_cache=sc, polarity=net._polarity_f32,
                refractory=net.refractory, channel=net.channel)
            if int(np.argmax(ch[PRED_NEURONS])) == int(seq[i+1]): c += 1
            t += 1
    return c / t if t else 0.0


def mutate_protect_frozen(net, frozen):
    """Full mutate() but restore any frozen edge that gets removed."""
    frozen_state = {(r,c): bool(net.mask[r,c]) for r,c in frozen}
    undo = net.mutate()
    restored = 0
    for r, c in frozen:
        if frozen_state[r,c] and not net.mask[r, c]:
            net.mask[r, c] = True
            restored += 1
    if restored:
        net.resync_alive()
    return undo


def get_layer_stats(net, seqs, layers):
    L1, L2, L3 = layers; H = net.H
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
    fires = np.zeros(H); n = 0
    for seq in seqs[:1]:
        for i in range(min(10, len(seq)-1)):
            st, ch = SelfWiringGraph.rollout_token(
                net.input_projection[int(seq[i])], mask=net.mask,
                theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                input_duration=INPUT_DURATION, state=st, charge=ch,
                sparse_cache=sc, polarity=net._polarity_f32,
                refractory=net.refractory, channel=net.channel)
            fires += (np.abs(st) > 0).astype(np.float32); n += 1
    fr = fires / max(n, 1)
    return (f"L1:{np.mean(fr[:L1]):.2f} L2:{np.mean(fr[L1:L1+L2]):.2f} "
            f"L3:{np.mean(fr[L1+L2:]):.2f} edges:{len(net.alive)}")


def run_scaffold(task_fn, seed, steps_scaffold, steps_melt, eval_seqs, layers_cfg):
    L1, n_loops, L3 = layers_cfg
    random.seed(seed); np.random.seed(seed)
    net, frozen, layers = build_scaffold(L1, n_loops, L3, seed)
    H = net.H
    LOG = max(1, (steps_scaffold + steps_melt) // 15)

    best = eval_acc(net, eval_seqs); accepts = 0
    curve = [(0, best, 'S')]

    # Phase 1: Scaffold
    for step in range(1, steps_scaffold + 1):
        snap = net.save_state()
        mutate_protect_frozen(net, frozen)
        new = eval_acc(net, eval_seqs)
        if new > best: best = new; accepts += 1
        else: net.restore_state(snap)
        if step % LOG == 0:
            stats = get_layer_stats(net, eval_seqs, layers)
            curve.append((step, best, 'S'))
            print(f"    scaffold {step:5d} | acc={best:.3f} | {stats} | acc#={accepts}")

    pre_melt = best

    # Phase 2: Melt
    frozen = set()  # unfreeze
    melt_acc = 0
    for step in range(1, steps_melt + 1):
        snap = net.save_state()
        net.mutate()  # full, no protection
        new = eval_acc(net, eval_seqs)
        if new > best: best = new; melt_acc += 1
        else: net.restore_state(snap)
        if step % LOG == 0:
            stats = get_layer_stats(net, eval_seqs, layers)
            curve.append((steps_scaffold + step, best, 'M'))
            print(f"    melted   {step:5d} | acc={best:.3f} | {stats} | acc#={melt_acc}")

    return best, accepts, melt_acc, pre_melt, curve


def run_flat(H, task_fn, seed, total_steps, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    LOG = max(1, total_steps // 15)
    best = eval_acc(net, eval_seqs); accepts = 0
    for step in range(1, total_steps + 1):
        snap = net.save_state(); net.mutate()
        new = eval_acc(net, eval_seqs)
        if new > best: best = new; accepts += 1
        else: net.restore_state(snap)
        if step % LOG == 0:
            print(f"    flat     {step:5d} | acc={best:.3f} | edges:{len(net.alive)} | acc#={accepts}")
    return best, accepts


def main():
    SEEDS = [42, 123]
    TASKS = [("Alt", make_alternating), ("Cyc3", make_cycle3)]
    STEPS_S = 1500; STEPS_M = 1500
    # Config: L1=24, 4 loops (12 neurons), L3=24 → H=60
    L1 = 24; N_LOOPS = 4; L3 = 24; H = L1 + N_LOOPS * 3 + L3

    print("=" * 90)
    print(f"  Scaffold v2: L1={L1} + {N_LOOPS} loops({N_LOOPS*3}n) + L3={L3} = H={H}")
    print(f"  theta=1, density=4%, full mutate(), frozen loops protected")
    print(f"  Scaffold: {STEPS_S} steps → Melt: {STEPS_M} steps | Total: {STEPS_S+STEPS_M}")
    print("=" * 90)

    results = []
    for tn, tf in TASKS:
        for seed in SEEDS:
            eseqs = [tf(np.random.RandomState(77+i), 30) for i in range(3)]

            print(f"\n  === {tn} seed={seed} SCAFFOLD ===")
            acc_s, acc_scaffold, acc_melt, pre_melt, curve = run_scaffold(
                tf, seed, STEPS_S, STEPS_M, eseqs, (L1, N_LOOPS, L3))

            print(f"\n  === {tn} seed={seed} FLAT H={H} ===")
            acc_f, acc_flat = run_flat(H, tf, seed, STEPS_S + STEPS_M, eseqs)

            results.append((tn, seed, acc_s, pre_melt, acc_f))

    print(f"\n{'='*90}")
    print(f"  RESULTS")
    print(f"{'='*90}")
    print(f"  {'Task':>5} | {'Seed':>4} | {'Scaffold':>8} | {'Pre-melt':>8} | {'Post-melt':>9} | {'Flat':>5} | {'Winner':>8}")
    for tn, seed, final, pre_m, flat in results:
        w = "SCAFF" if final > flat + 0.005 else ("FLAT" if flat > final + 0.005 else "TIE")
        print(f"  {tn:>5} | {seed:4d} | {pre_m:8.3f} | {pre_m:8.3f} | {final:9.3f} | {flat:5.3f} | {w:>8}")

    # Averages
    scaff_avg = np.mean([r[2] for r in results])
    flat_avg = np.mean([r[4] for r in results])
    print(f"\n  Average: scaffold={scaff_avg:.3f} flat={flat_avg:.3f} delta={scaff_avg-flat_avg:+.3f}")


if __name__ == "__main__":
    main()
