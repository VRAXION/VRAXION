"""
INSTNCT — Rooted Pathway v2: Adversarial Validation + Deep Telemetry
=====================================================================
Questions to answer:
  1. Is the 63.9% real or a code bug / false positive?
  2. Does the trial period (rip if bad) ever trigger?
  3. What happens on English text (mainline task)?
  4. WHEN do accuracy jumps happen? After pathway insertion or background?
  5. Multiple seeds (5) for statistical confidence
  6. Ablation: is BFS anchoring the key, or just "more edges"?

Tests:
  A) rooted_v2 (full: BFS + loop + trial + background)
  B) random_pathway (same edges but NO BFS — random anchor points)
  C) edges_only (same # of random edges, no loop structure)
  D) single (baseline)

Tasks: Alt, Cyc3, English
Seeds: 5
Deep telemetry: accuracy curve, pathway events, edge count, accept rate
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
H = 64; STEPS = 2500; TRIAL_BUDGET = 200

TEXT = ("a stitch in time saves nine. the early bird catches the worm. "
       "all that glitters is not gold. actions speak louder than words. "
       "fortune favors the bold. knowledge is power. practice makes perfect. ")
TEXT_BYTES = list(np.frombuffer(TEXT.encode('ascii'), dtype=np.uint8))


def make_alternating(rng, n=30):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]

def make_cycle3(rng, n=30):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]

def make_english(rng, n=30):
    start = rng.randint(0, max(1, len(TEXT_BYTES) - n - 1))
    return TEXT_BYTES[start:start + n + 1]

TASKS = [("Alt", make_alternating), ("Cyc3", make_cycle3), ("Eng", make_english)]


def bfs_reachable(mask, start_nodes, max_hops=3):
    H = mask.shape[0]
    reached = set(int(n) for n in start_nodes)
    frontier = set(reached)
    for _ in range(max_hops):
        nxt = set()
        for n in frontier:
            for t in np.where(mask[n])[0]:
                if int(t) not in reached:
                    reached.add(int(t)); nxt.add(int(t))
        frontier = nxt
        if not frontier: break
    return reached

def reverse_bfs(mask, end_nodes, max_hops=3):
    H = mask.shape[0]
    reached = set(int(n) for n in end_nodes)
    frontier = set(reached)
    for _ in range(max_hops):
        nxt = set()
        for n in frontier:
            for s in np.where(mask[:, n])[0]:
                if int(s) not in reached:
                    reached.add(int(s)); nxt.add(int(s))
        frontier = nxt
        if not frontier: break
    return reached


def build_rooted(net):
    """BFS-anchored loop. Returns (edges, loop_nodes) or (None, None)."""
    input_str = np.abs(net.input_projection).mean(axis=0)
    input_zone = list(np.where(input_str > np.median(input_str))[0])
    output_zone = [n for n in PRED_NEURONS if n < H]
    from_in = bfs_reachable(net.mask, input_zone)
    to_out = reverse_bfs(net.mask, output_zone)
    anchors_in = list(from_in - set(PRED_NEURONS))
    anchors_out = list(to_out - set(PRED_NEURONS))
    if not anchors_in or not anchors_out: return None, None
    a_in = random.choice(anchors_in); a_out = random.choice(anchors_out)
    used = set(PRED_NEURONS) | {a_in, a_out}
    avail = [n for n in range(H) if n not in used]
    if len(avail) < 2: return None, None
    mids = random.sample(avail, random.randint(2, min(3, len(avail))))
    chain = [a_in] + mids + [a_out]
    edges = []
    for i in range(len(chain) - 1):
        r, c = chain[i], chain[i+1]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True; edges.append((r, c))
    if a_out != a_in and not net.mask[a_out, a_in]:
        net.mask[a_out, a_in] = True; edges.append((a_out, a_in))
    for n in chain:
        if net.theta[n] > 1:
            net.theta[n] = np.uint8(max(1, int(net.theta[n]) - 1))
            net._theta_f32[n] = float(net.theta[n])
    net.resync_alive()
    return edges if edges else None, chain


def build_random_pathway(net):
    """Same structure as rooted but RANDOM anchors (no BFS). Ablation control."""
    nodes = random.sample(range(H), min(5, H))
    chain = nodes
    edges = []
    for i in range(len(chain) - 1):
        r, c = chain[i], chain[i+1]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True; edges.append((r, c))
    if chain[-1] != chain[0] and not net.mask[chain[-1], chain[0]]:
        net.mask[chain[-1], chain[0]] = True; edges.append((chain[-1], chain[0]))
    for n in chain:
        if net.theta[n] > 1:
            net.theta[n] = np.uint8(max(1, int(net.theta[n]) - 1))
            net._theta_f32[n] = float(net.theta[n])
    net.resync_alive()
    return edges if edges else None


def add_random_edges(net, n_edges=5):
    """Same number of edges but NO structure. Pure edge-count control."""
    added = 0
    for _ in range(n_edges * 3):
        r, c = random.randint(0, H-1), random.randint(0, H-1)
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True; added += 1
            if added >= n_edges: break
    net.resync_alive()
    return added


def rip_edges(net, edges):
    for r, c in edges:
        if net.mask[r, c]: net.mask[r, c] = False
    net.resync_alive()


def eval_acc(net, seqs):
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
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


def run_arm(mode, seed, task_fn, eval_seqs):
    """Run one arm with deep telemetry."""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)

    best = eval_acc(net, eval_seqs); accepts = 0
    active_pw = None; trial_start = 0; trial_best = 0
    pw_built = 0; pw_ripped = 0; pw_kept = 0

    # Telemetry: accuracy every 50 steps
    telem = {'acc_curve': [], 'pw_events': [], 'edge_curve': [], 'accept_curve': []}
    LOG_EVERY = 50

    for step in range(1, STEPS + 1):
        snap = net.save_state()

        # === PATHWAY LIFECYCLE (modes: rooted_v2, random_pw) ===
        if mode in ('rooted_v2', 'random_pw'):
            if active_pw is None and step % 100 == 0:
                if mode == 'rooted_v2':
                    pw, nodes = build_rooted(net)
                else:
                    pw = build_random_pathway(net)
                if pw:
                    active_pw = pw; trial_start = step
                    trial_best = eval_acc(net, eval_seqs)
                    pw_built += 1
                    telem['pw_events'].append((step, 'BUILD', len(pw)))
            elif active_pw is not None and step - trial_start >= TRIAL_BUDGET:
                current = eval_acc(net, eval_seqs)
                if current < trial_best - 0.01:
                    rip_edges(net, active_pw)
                    pw_ripped += 1
                    telem['pw_events'].append((step, 'RIP', len(active_pw)))
                else:
                    pw_kept += 1
                    telem['pw_events'].append((step, 'KEEP', len(active_pw)))
                active_pw = None

        # === EDGE-ONLY CONTROL ===
        if mode == 'edges_only' and step % 100 == 0:
            add_random_edges(net, 5)

        # === BACKGROUND MUTATION (all modes) ===
        net.mutate()
        net.mutate(forced_op=random.choice(['theta', 'decay', 'rewire']))

        new = eval_acc(net, eval_seqs)
        if new > best:
            # Telemetry: record jump
            jump = new - best
            if jump > 0.02:
                has_pw = active_pw is not None
                telem['pw_events'].append((step, f'JUMP+{jump:.3f}', 'pw_active' if has_pw else 'no_pw'))
            best = new; accepts += 1
        else:
            net.restore_state(snap)

        if step % LOG_EVERY == 0:
            telem['acc_curve'].append((step, best))
            telem['edge_curve'].append((step, len(net.alive)))
            telem['accept_curve'].append((step, accepts))

    return {
        'acc': best, 'accepts': accepts,
        'pw_built': pw_built, 'pw_ripped': pw_ripped, 'pw_kept': pw_kept,
        'edges': len(net.alive), 'telem': telem,
    }


def main():
    SEEDS = [42, 123, 456, 789, 999]
    MODES = ['single', 'rooted_v2', 'random_pw', 'edges_only']

    print("=" * 100)
    print("  Adversarial Validation: Rooted Pathway v2")
    print(f"  H={H} | Steps={STEPS} | Seeds={len(SEEDS)} | Trial={TRIAL_BUDGET}")
    print(f"  Modes: {MODES}")
    print(f"  Tasks: Alt, Cyc3, English")
    print("=" * 100)

    task_eval = {tn: [tf(np.random.RandomState(77 + i), 30) for i in range(3)]
                 for tn, tf in TASKS}

    # Results[mode][task] = list of result dicts
    results = {m: {tn: [] for tn, _ in TASKS} for m in MODES}
    total = len(SEEDS) * len(TASKS) * len(MODES)
    done = 0

    for seed in SEEDS:
        for tn, tf in TASKS:
            for mode in MODES:
                done += 1
                print(f"  [{done:3d}/{total}] s={seed:3d} {tn:>5} {mode:>10}...", end="", flush=True)
                t0 = time.time()
                r = run_arm(mode, seed, tf, task_eval[tn])
                pw_str = f" built={r['pw_built']} rip={r['pw_ripped']} keep={r['pw_kept']}" if r['pw_built'] > 0 else ""
                print(f" acc={r['acc']:.3f} acc#={r['accepts']} e={r['edges']}{pw_str} ({time.time()-t0:.0f}s)")
                results[mode][tn].append(r)

    # === SUMMARY ===
    print(f"\n{'='*100}")
    print(f"  RESULTS — 5 seeds averaged")
    print(f"{'='*100}")

    print(f"\n  {'':>10} |", " | ".join(f"{tn:>7}" for tn, _ in TASKS), "| overall | std")
    mode_stats = {}
    for mode in MODES:
        row = f"  {mode:>10} |"
        all_accs = []
        for tn, _ in TASKS:
            accs = [r['acc'] for r in results[mode][tn]]
            avg = np.mean(accs)
            all_accs.extend(accs)
            row += f" {avg:7.3f} |"
        overall = np.mean(all_accs)
        std = np.std(all_accs)
        mode_stats[mode] = (overall, std)
        row += f" {overall:.3f}  | {std:.3f}"
        print(row)

    # === ABLATION: Is BFS the key? ===
    print(f"\n  ABLATION: BFS anchoring")
    rooted = mode_stats['rooted_v2'][0]
    random_pw = mode_stats['random_pw'][0]
    edges = mode_stats['edges_only'][0]
    single = mode_stats['single'][0]
    print(f"    rooted_v2:   {rooted:.3f} (BFS + loop)")
    print(f"    random_pw:   {random_pw:.3f} (random anchor + loop)")
    print(f"    edges_only:  {edges:.3f} (random edges, no loop)")
    print(f"    single:      {single:.3f} (baseline)")
    print(f"    BFS value:   {rooted - random_pw:+.3f} (rooted - random)")
    print(f"    Loop value:  {random_pw - edges:+.3f} (loop - edges)")
    print(f"    Edge value:  {edges - single:+.3f} (edges - single)")

    # === PATHWAY LIFECYCLE ===
    print(f"\n  PATHWAY LIFECYCLE:")
    for mode in ['rooted_v2', 'random_pw']:
        built = sum(r['pw_built'] for tn, _ in TASKS for r in results[mode][tn])
        ripped = sum(r['pw_ripped'] for tn, _ in TASKS for r in results[mode][tn])
        kept = sum(r['pw_kept'] for tn, _ in TASKS for r in results[mode][tn])
        print(f"    {mode}: built={built} kept={kept} ripped={ripped} "
              f"(keep rate: {100*kept/max(built,1):.0f}%)")

    # === DEEP TELEMETRY: accuracy jumps ===
    print(f"\n  ACCURACY JUMPS (>2% gains):")
    for mode in ['rooted_v2', 'random_pw']:
        jumps_with_pw = 0; jumps_without = 0
        for tn, _ in TASKS:
            for r in results[mode][tn]:
                for ev in r['telem']['pw_events']:
                    if 'JUMP' in str(ev[1]):
                        if 'pw_active' in str(ev[2]):
                            jumps_with_pw += 1
                        else:
                            jumps_without += 1
        total_j = jumps_with_pw + jumps_without
        print(f"    {mode}: {jumps_with_pw} during pathway, {jumps_without} during background"
              f" ({100*jumps_with_pw/max(total_j,1):.0f}% pathway-linked)")

    # === STATISTICAL SIGNIFICANCE ===
    print(f"\n  STATISTICAL TEST (rooted_v2 vs single):")
    r_accs = [r['acc'] for tn, _ in TASKS for r in results['rooted_v2'][tn]]
    s_accs = [r['acc'] for tn, _ in TASKS for r in results['single'][tn]]
    diff = np.mean(r_accs) - np.mean(s_accs)
    pooled_std = np.sqrt((np.var(r_accs) + np.var(s_accs)) / 2)
    t_stat = diff / (pooled_std / np.sqrt(len(r_accs))) if pooled_std > 0 else 0
    print(f"    mean diff: {diff:+.3f}")
    print(f"    t-stat: {t_stat:.2f} ({'p<0.05' if abs(t_stat) > 2.0 else 'not significant'})")

    winner = max(mode_stats, key=lambda m: mode_stats[m][0])
    print(f"\n  WINNER: {winner} ({mode_stats[winner][0]:.3f} ± {mode_stats[winner][1]:.3f})")


if __name__ == "__main__":
    main()
