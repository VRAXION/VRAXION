"""
INSTNCT — Pathway Builder v2: rooted loops + trial + background tuning
=======================================================================
Two-layer mutation:
  Layer 1 (rare): Build pathway rooted in existing IO-reachable neurons
    1. Find neuron reachable FROM input (BFS from input zone)
    2. Find neuron reachable TO output (reverse BFS from output zone)
    3. Create loop between them with 3-5 new intermediaries
    4. Trial period: burst-mutate ONLY the loop for N steps
    5. If no improvement → rip it out, try another
  Layer 2 (continuous): Background tuning on whole network
    Normal mutation: theta, decay, rewire, polarity drift

A) single (baseline)
B) smart_burst v1 (random placement)
C) rooted_pathway (v2 — IO-connected + trial period + background)
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
STEPS = 2500; TRIAL_BUDGET = 200  # steps to try a new pathway before ripping


def make_alternating(rng, n=30):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]

def make_cycle3(rng, n=30):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]

TASKS = [("Alt", make_alternating), ("Cyc3", make_cycle3)]


def bfs_reachable(mask, start_nodes, max_hops=4):
    """BFS from start_nodes, return all reachable neurons within max_hops."""
    H = mask.shape[0]
    reached = set(start_nodes)
    frontier = set(start_nodes)
    for _ in range(max_hops):
        next_frontier = set()
        for n in frontier:
            targets = np.where(mask[n])[0]
            for t in targets:
                if t not in reached:
                    reached.add(int(t))
                    next_frontier.add(int(t))
        frontier = next_frontier
        if not frontier:
            break
    return reached


def reverse_bfs_reachable(mask, end_nodes, max_hops=4):
    """Reverse BFS: which neurons can REACH end_nodes within max_hops."""
    H = mask.shape[0]
    reached = set(end_nodes)
    frontier = set(end_nodes)
    for _ in range(max_hops):
        next_frontier = set()
        for n in frontier:
            sources = np.where(mask[:, n])[0]
            for s in sources:
                if s not in reached:
                    reached.add(int(s))
                    next_frontier.add(int(s))
        frontier = next_frontier
        if not frontier:
            break
    return reached


def find_io_anchors(net):
    """Find neurons reachable from input and neurons that reach output."""
    H = net.H
    # Input zone: neurons that get nonzero injection from input_projection
    # (all neurons get some, but high-magnitude ones are effective inputs)
    input_strength = np.abs(net.input_projection).mean(axis=0)  # H-dim
    input_zone = list(np.where(input_strength > np.median(input_strength))[0])

    # Output zone: prediction neurons
    output_zone = [n for n in PRED_NEURONS if n < H]

    # Forward reachable from input
    from_input = bfs_reachable(net.mask, input_zone, max_hops=3)
    # Reverse reachable to output
    to_output = reverse_bfs_reachable(net.mask, output_zone, max_hops=3)

    return from_input, to_output


def build_rooted_pathway(net):
    """Build a loop ROOTED in IO-connected neurons. Returns loop edges or None."""
    H = net.H
    from_input, to_output = find_io_anchors(net)

    # Find anchor pair: one reachable from input, one reaching output
    input_anchors = list(from_input - set(PRED_NEURONS))
    output_anchors = list(to_output - set(PRED_NEURONS))

    if not input_anchors or not output_anchors:
        return None

    anchor_in = random.choice(input_anchors)
    anchor_out = random.choice(output_anchors)

    # Build loop: anchor_in → N1 → N2 → ... → anchor_out → anchor_in
    # Pick 2-3 intermediate neurons (not already in critical paths)
    used = set(PRED_NEURONS) | {anchor_in, anchor_out}
    available = [n for n in range(H) if n not in used]
    if len(available) < 2:
        return None

    n_mid = random.randint(2, min(3, len(available)))
    middles = random.sample(available, n_mid)

    # Chain: anchor_in → mid1 → mid2 → anchor_out → anchor_in (loop back)
    chain = [anchor_in] + middles + [anchor_out]
    edges = []
    for i in range(len(chain) - 1):
        r, c = chain[i], chain[i + 1]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True
            edges.append((r, c))
    # Close loop: anchor_out → anchor_in
    if anchor_out != anchor_in and not net.mask[anchor_out, anchor_in]:
        net.mask[anchor_out, anchor_in] = True
        edges.append((anchor_out, anchor_in))

    if edges:
        # Tune theta down on loop neurons
        for n in chain:
            if net.theta[n] > 1:
                net.theta[n] = np.uint8(max(1, int(net.theta[n]) - 1))
                net._theta_f32[n] = float(net.theta[n])
        net.resync_alive()

    return edges if edges else None


def rip_pathway(net, edges):
    """Remove all edges of a failed pathway."""
    for r, c in edges:
        if net.mask[r, c]:
            net.mask[r, c] = False
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


RANDOM_OPS = ['add', 'add_loop', 'remove', 'rewire', 'theta', 'decay', 'polarity']


def run_rooted(H, task_fn, seed, eval_seqs):
    """Layer 1: rooted pathway with trial + Layer 2: background tuning."""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)

    best = eval_acc(net, eval_seqs); accepts = 0
    active_pathway = None
    trial_start = 0; trial_best = 0
    pathways_built = 0; pathways_ripped = 0

    for step in range(1, STEPS + 1):
        snap = net.save_state()

        # LAYER 1: Pathway lifecycle
        if active_pathway is None:
            # Try to build a new pathway every 100 steps
            if step % 100 == 0:
                pw = build_rooted_pathway(net)
                if pw:
                    active_pathway = pw
                    trial_start = step
                    trial_best = eval_acc(net, eval_seqs)
                    pathways_built += 1

        elif step - trial_start >= TRIAL_BUDGET:
            # Trial period expired: did it help?
            current = eval_acc(net, eval_seqs)
            if current <= trial_best - 0.005:
                # Failed → rip it out
                rip_pathway(net, active_pathway)
                pathways_ripped += 1
            # Either way, pathway trial is over
            active_pathway = None

        # LAYER 2: Background tuning (always)
        net.mutate()
        # Extra: one more random op for variety
        net.mutate(forced_op=random.choice(['theta', 'decay', 'rewire']))

        new = eval_acc(net, eval_seqs)
        if new > best:
            best = new; accepts += 1
        else:
            net.restore_state(snap)

    return best, accepts, pathways_built, pathways_ripped


def run_smart_v1(H, task_fn, seed, eval_seqs):
    """Smart burst v1 (random placement) from previous test."""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    best = eval_acc(net, eval_seqs); accepts = 0; stale = 0

    for step in range(1, STEPS + 1):
        if stale >= 300:
            nodes = random.sample(range(H), min(random.randint(3, 5), H))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0

        snap = net.save_state()
        # Smart burst v1: loop + wire + tune + prune
        loop_len = random.randint(3, min(4, H))
        loop_nodes = random.sample(range(H), loop_len)
        for i in range(loop_len):
            r, c = loop_nodes[i], loop_nodes[(i+1) % loop_len]
            if r != c and not net.mask[r, c]: net.mask[r, c] = True
        for n in loop_nodes:
            if net.theta[n] > 1:
                net.theta[n] = np.uint8(max(1, int(net.theta[n]) - 1))
                net._theta_f32[n] = float(net.theta[n])
        non_loop = [n for n in range(H) if n not in loop_nodes]
        if non_loop:
            src = random.choice(non_loop)
            if src != loop_nodes[0] and not net.mask[src, loop_nodes[0]]:
                net.mask[src, loop_nodes[0]] = True
        target_zone = [n for n in PRED_NEURONS if n < H and n != loop_nodes[-1]]
        if target_zone:
            tgt = random.choice(target_zone)
            if not net.mask[loop_nodes[-1], tgt]:
                net.mask[loop_nodes[-1], tgt] = True
        net.resync_alive()

        new = eval_acc(net, eval_seqs)
        if new > best:
            best = new; accepts += 1; stale = 0
        else:
            net.restore_state(snap); stale += 1

    return best, accepts, 0, 0


def run_single(H, task_fn, seed, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    best = eval_acc(net, eval_seqs); accepts = 0; stale = 0
    for step in range(1, STEPS + 1):
        if stale >= 300:
            nodes = random.sample(range(H), min(random.randint(3, 5), H))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0
        snap = net.save_state(); net.mutate()
        new = eval_acc(net, eval_seqs)
        if new > best: best = new; accepts += 1; stale = 0
        else: net.restore_state(snap); stale += 1
    return best, accepts, 0, 0


def main():
    H_OPTIONS = [32, 64]
    SEEDS = [42, 123]
    MODES = [
        ("single", run_single),
        ("smart_v1", run_smart_v1),
        ("rooted_v2", run_rooted),
    ]

    print("=" * 90)
    print("  Rooted Pathway v2: IO-connected loops + trial period + background")
    print(f"  H: {H_OPTIONS} | Seeds: {SEEDS} | Steps: {STEPS} | Trial: {TRIAL_BUDGET}")
    print("=" * 90)

    task_eval = {tn: [tf(np.random.RandomState(77 + i), 30) for i in range(3)]
                 for tn, tf in TASKS}

    all_results = {m[0]: {tn: [] for tn, _ in TASKS} for m in MODES}
    total = len(H_OPTIONS) * len(SEEDS) * len(TASKS) * len(MODES)
    done = 0

    for H in H_OPTIONS:
        for seed in SEEDS:
            for tn, tf in TASKS:
                for mode_name, mode_fn in MODES:
                    done += 1
                    print(f"  [{done:2d}/{total}] H={H} s={seed} {tn:>5} {mode_name:>10}...",
                          end="", flush=True)
                    t0 = time.time()
                    acc, accepts, built, ripped = mode_fn(H, tf, seed, task_eval[tn])
                    extra = f" built={built} ripped={ripped}" if built > 0 else ""
                    print(f" acc={acc:.3f} acc#={accepts}{extra} ({time.time()-t0:.0f}s)")
                    all_results[mode_name][tn].append((H, seed, acc))

    # Summary
    print(f"\n{'='*90}")
    print(f"  RESULTS")
    print(f"{'='*90}")
    print(f"  {'':>10} |", " | ".join(f"{tn:>7}" for tn, _ in TASKS), "| overall")

    mode_overalls = {}
    for mode_name, _ in MODES:
        row = f"  {mode_name:>10} |"
        tavgs = []
        for tn, _ in TASKS:
            accs = [r[2] for r in all_results[mode_name][tn]]
            avg = np.mean(accs)
            tavgs.append(avg)
            row += f" {avg:7.3f} |"
        overall = np.mean(tavgs)
        mode_overalls[mode_name] = overall
        row += f" {overall:.3f}"
        print(row)

    winner = max(mode_overalls, key=mode_overalls.get)
    print(f"\n  WINNER: {winner} ({mode_overalls[winner]:.3f})")

    # Head to head: rooted_v2 vs smart_v1
    r_wins = 0; s_wins = 0; ties = 0
    for tn, _ in TASKS:
        for i in range(len(all_results['rooted_v2'][tn])):
            r = all_results['rooted_v2'][tn][i][2]
            s = all_results['smart_v1'][tn][i][2]
            if r > s + 0.005: r_wins += 1
            elif s > r + 0.005: s_wins += 1
            else: ties += 1
    print(f"\n  rooted_v2 vs smart_v1: {r_wins}W-{s_wins}L-{ties}T")


if __name__ == "__main__":
    main()
