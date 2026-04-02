"""
INSTNCT — Smart Burst Sweep: random vs guided vs coherent pathway
==================================================================
A) single:        1 default mutate() per step
B) burst7_random: 7× random forced_op per step
C) smart_burst:   coherent pathway builder (loop + wire + tune + prune)

Faster sweep: 2 sizes × 2 tasks × 2 seeds × 3 modes = 24 configs.
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
MAX_STAMINA = 15; STEPS = 2000; PLATEAU_WINDOW = 300

RANDOM_OPS = ['add', 'add_loop', 'remove', 'rewire', 'theta', 'decay', 'polarity']


def make_alternating(rng, n=30):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]

def make_cycle3(rng, n=30):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]

TASKS = [("Alt", make_alternating), ("Cyc3", make_cycle3)]


def smart_burst(net):
    """Coherent pathway builder: loop + wire-in + wire-out + tune + prune."""
    H = net.H
    alive_before = set(net.alive_set)

    # 1. ADD_LOOP: create feedback circuit (3-4 nodes)
    loop_len = random.randint(3, min(4, H))
    loop_nodes = random.sample(range(H), loop_len)
    loop_added = []
    for i in range(loop_len):
        r, c = loop_nodes[i], loop_nodes[(i + 1) % loop_len]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True
            loop_added.append((r, c))

    if not loop_added:
        # Loop collision — fallback to random burst
        net.resync_alive()
        for _ in range(5):
            net.mutate(forced_op=random.choice(RANDOM_OPS))
        return

    # 2. THETA DOWN on loop neurons → make them easier to fire
    for n in loop_nodes:
        if net.theta[n] > 1:
            net.theta[n] = np.uint8(max(1, int(net.theta[n]) - 1))
            net._theta_f32[n] = float(net.theta[n])

    # 3. WIRE IN: connect a random non-loop neuron → loop entry point
    non_loop = [n for n in range(H) if n not in loop_nodes]
    if non_loop:
        src = random.choice(non_loop)
        dst = loop_nodes[0]  # loop entry
        if src != dst and not net.mask[src, dst]:
            net.mask[src, dst] = True

    # 4. WIRE OUT: connect loop exit → a prediction neuron zone
    exit_node = loop_nodes[-1]
    target_zone = [n for n in PRED_NEURONS if n < H and n != exit_node]
    if target_zone:
        tgt = random.choice(target_zone)
        if not net.mask[exit_node, tgt]:
            net.mask[exit_node, tgt] = True

    # 5. PRUNE: remove a random edge that's NOT in the new loop
    new_alive = set()
    rows, cols = np.where(net.mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        new_alive.add((r, c))
    non_loop_edges = [e for e in new_alive if e not in set(loop_added)]
    if len(non_loop_edges) > 10:  # keep minimum edges
        victim = random.choice(non_loop_edges)
        net.mask[victim[0], victim[1]] = False

    net.resync_alive()


def eval_acc(net, seqs):
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    st = np.zeros(net.H, dtype=np.float32); ch = np.zeros(net.H, dtype=np.float32)
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


def run_config(H, task_fn, seed, mode, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)

    best = eval_acc(net, eval_seqs); accepts = 0; stale = 0

    for step in range(1, STEPS + 1):
        # Plateau loop injection (all arms get this)
        if stale >= PLATEAU_WINDOW:
            nodes = random.sample(range(H), min(random.randint(3, 5), H))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0

        snap = net.save_state()

        if mode == 'single':
            net.mutate()
        elif mode == 'burst7_random':
            for _ in range(7):
                net.mutate(forced_op=random.choice(RANDOM_OPS))
        elif mode == 'smart_burst':
            smart_burst(net)

        new = eval_acc(net, eval_seqs)
        if new > best:
            best = new; accepts += 1; stale = 0
        else:
            net.restore_state(snap); stale += 1

    return best, accepts


def main():
    H_OPTIONS = [32, 64]
    SEEDS = [42, 123]
    MODES = ['single', 'burst7_random', 'smart_burst']

    print("=" * 90)
    print("  Smart Burst Sweep: random vs coherent pathway builder")
    print(f"  H: {H_OPTIONS} | Seeds: {SEEDS} | Tasks: {len(TASKS)} | Steps: {STEPS}")
    print("=" * 90)

    task_eval = {}
    for tn, tf in TASKS:
        task_eval[tn] = [tf(np.random.RandomState(77 + i), 30) for i in range(3)]

    all_results = {m: {tn: [] for tn, _ in TASKS} for m in MODES}
    total = len(H_OPTIONS) * len(SEEDS) * len(TASKS) * len(MODES)
    done = 0

    for H in H_OPTIONS:
        for seed in SEEDS:
            for tn, tf in TASKS:
                for mode in MODES:
                    done += 1
                    print(f"  [{done:2d}/{total}] H={H} s={seed} {tn:>5} {mode:>14}...", end="", flush=True)
                    t0 = time.time()
                    acc, accepts = run_config(H, tf, seed, mode, task_eval[tn])
                    print(f" acc={acc:.3f} acc#={accepts} ({time.time()-t0:.0f}s)")
                    all_results[mode][tn].append((H, seed, acc))

    # Summary
    print(f"\n{'='*90}")
    print(f"  RESULTS — per task average (all H, all seeds)")
    print(f"{'='*90}")
    print(f"  {'':>14} |", " | ".join(f"{tn:>7}" for tn, _ in TASKS), "| overall")

    mode_overalls = {}
    for mode in MODES:
        row = f"  {mode:>14} |"
        tavgs = []
        for tn, _ in TASKS:
            accs = [r[2] for r in all_results[mode][tn]]
            avg = np.mean(accs)
            tavgs.append(avg)
            row += f" {avg:7.3f} |"
        overall = np.mean(tavgs)
        mode_overalls[mode] = overall
        row += f" {overall:.3f}"
        print(row)

    # Per H breakdown
    print(f"\n  Per network size:")
    print(f"  {'':>14} |", " | ".join(f"H={h:>3}" for h in H_OPTIONS), "| avg")
    for mode in MODES:
        row = f"  {mode:>14} |"
        havgs = []
        for H in H_OPTIONS:
            accs = [r[2] for all_t in all_results[mode].values() for r in all_t if r[0] == H]
            avg = np.mean(accs) if accs else 0
            havgs.append(avg)
            row += f" {avg:5.3f} |"
        row += f" {np.mean(havgs):.3f}"
        print(row)

    # Per seed breakdown (consistency)
    print(f"\n  Per seed (consistency):")
    print(f"  {'':>14} |", " | ".join(f"s={s}" for s in SEEDS), "| std")
    for mode in MODES:
        row = f"  {mode:>14} |"
        savgs = []
        for seed in SEEDS:
            accs = [r[2] for all_t in all_results[mode].values() for r in all_t if r[1] == seed]
            avg = np.mean(accs) if accs else 0
            savgs.append(avg)
            row += f" {avg:5.3f} |"
        row += f" {np.std(savgs):.3f}"
        print(row)

    winner = max(mode_overalls, key=mode_overalls.get)
    base = mode_overalls['single']
    print(f"\n  WINNER: {winner} ({mode_overalls[winner]:.3f}, +{mode_overalls[winner]-base:.3f} vs single)")

    # Individual matchups
    print(f"\n  Head-to-head (per config):")
    smart_wins = 0; random_wins = 0; ties = 0
    for tn, _ in TASKS:
        for i in range(len(all_results['smart_burst'][tn])):
            s = all_results['smart_burst'][tn][i][2]
            r = all_results['burst7_random'][tn][i][2]
            if s > r + 0.005: smart_wins += 1
            elif r > s + 0.005: random_wins += 1
            else: ties += 1
    total_matches = smart_wins + random_wins + ties
    print(f"    smart_burst wins: {smart_wins}/{total_matches}")
    print(f"    burst7_random wins: {random_wins}/{total_matches}")
    print(f"    ties: {ties}/{total_matches}")


if __name__ == "__main__":
    main()
