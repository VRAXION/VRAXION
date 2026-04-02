"""
INSTNCT — Scale test: does pathway structure matter at H=256?
==============================================================
At H=64 the search space is small (4K edges) → random finds things.
At H=256 the space is 65K edges → targeted should help more.

Quick test: 3 modes × 2 seeds × Cyc3 only (hardest pattern task).
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
H = 256; STEPS = 2000


def make_cycle3(rng, n=30):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]


def bfs_reach(mask, starts, hops=3):
    reached = set(int(n) for n in starts)
    frontier = set(reached)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            for t in np.where(mask[n])[0]:
                if int(t) not in reached:
                    reached.add(int(t)); nxt.add(int(t))
        frontier = nxt
        if not frontier: break
    return reached

def rev_bfs(mask, ends, hops=3):
    reached = set(int(n) for n in ends)
    frontier = set(reached)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            for s in np.where(mask[:, n])[0]:
                if int(s) not in reached:
                    reached.add(int(s)); nxt.add(int(s))
        frontier = nxt
        if not frontier: break
    return reached


def build_rooted(net):
    inp_str = np.abs(net.input_projection).mean(axis=0)
    inp_zone = list(np.where(inp_str > np.median(inp_str))[0])
    out_zone = [n for n in PRED_NEURONS if n < H]
    fi = bfs_reach(net.mask, inp_zone)
    to = rev_bfs(net.mask, out_zone)
    ai = list(fi - set(PRED_NEURONS)); ao = list(to - set(PRED_NEURONS))
    if not ai or not ao: return None
    a_in = random.choice(ai); a_out = random.choice(ao)
    used = set(PRED_NEURONS) | {a_in, a_out}
    avail = [n for n in range(H) if n not in used]
    if len(avail) < 2: return None
    mids = random.sample(avail, random.randint(2, min(3, len(avail))))
    chain = [a_in] + mids + [a_out]
    edges = []
    for i in range(len(chain)-1):
        r, c = chain[i], chain[i+1]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True; edges.append((r, c))
    if a_out != a_in and not net.mask[a_out, a_in]:
        net.mask[a_out, a_in] = True; edges.append((a_out, a_in))
    for n in chain:
        if net.theta[n] > 1:
            net.theta[n] = np.uint8(max(1, int(net.theta[n])-1))
            net._theta_f32[n] = float(net.theta[n])
    net.resync_alive()
    return edges if edges else None


def add_random_edges(net, n=5):
    added = 0
    for _ in range(n*3):
        r, c = random.randint(0, H-1), random.randint(0, H-1)
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True; added += 1
            if added >= n: break
    net.resync_alive()


def eval_acc(net, seqs):
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
    c = 0; t = 0
    for seq in seqs:
        for i in range(len(seq)-1):
            st, ch = SelfWiringGraph.rollout_token(
                net.input_projection[int(seq[i])], mask=net.mask,
                theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                input_duration=INPUT_DURATION, state=st, charge=ch,
                sparse_cache=sc, polarity=net._polarity_f32,
                refractory=net.refractory, channel=net.channel)
            if int(np.argmax(ch[PRED_NEURONS])) == int(seq[i+1]): c += 1
            t += 1
    return c / t if t else 0.0


def run_arm(mode, seed, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    print(f"      init edges={len(net.alive)}", end="", flush=True)

    best = eval_acc(net, eval_seqs); accepts = 0
    active_pw = None; trial_start = 0; trial_best = 0
    pw_built = 0; pw_kept = 0; pw_ripped = 0
    log_every = STEPS // 8

    for step in range(1, STEPS+1):
        snap = net.save_state()

        if mode == 'rooted':
            if active_pw is None and step % 80 == 0:
                pw = build_rooted(net)
                if pw:
                    active_pw = pw; trial_start = step
                    trial_best = eval_acc(net, eval_seqs)
                    pw_built += 1
            elif active_pw and step - trial_start >= 150:
                cur = eval_acc(net, eval_seqs)
                if cur < trial_best - 0.01:
                    for r, c in active_pw:
                        if net.mask[r, c]: net.mask[r, c] = False
                    net.resync_alive(); pw_ripped += 1
                else:
                    pw_kept += 1
                active_pw = None
        elif mode == 'edges_only' and step % 80 == 0:
            add_random_edges(net, 5)

        net.mutate()
        net.mutate(forced_op=random.choice(['theta', 'decay', 'rewire']))

        new = eval_acc(net, eval_seqs)
        if new > best:
            best = new; accepts += 1
        else:
            net.restore_state(snap)

        if step % log_every == 0:
            print(f"\n      step {step:5d} acc={best:.3f} e={len(net.alive)} acc#={accepts}", end="", flush=True)

    return best, accepts, pw_built, pw_kept, pw_ripped, len(net.alive)


def main():
    SEEDS = [42, 123]
    MODES = ['single', 'rooted', 'edges_only']

    eval_seqs = [make_cycle3(np.random.RandomState(77+i), 30) for i in range(3)]

    print("=" * 80)
    print(f"  Scale Test: H={H} ({H*H} possible edges)")
    print(f"  Task: Cyc3 | Steps: {STEPS} | Seeds: {SEEDS}")
    print(f"  H=64 search space: 4K edges → H=256: 65K edges (16x bigger)")
    print("=" * 80)

    results = {}
    for seed in SEEDS:
        results[seed] = {}
        for mode in MODES:
            print(f"\n  s={seed} {mode}:", end="", flush=True)
            t0 = time.time()
            acc, accepts, built, kept, ripped, edges = run_arm(mode, seed, eval_seqs)
            elapsed = time.time() - t0
            pw_str = f" pw: built={built} kept={kept} rip={ripped}" if built > 0 else ""
            print(f"\n    FINAL: acc={acc:.3f} e={edges} acc#={accepts}{pw_str} ({elapsed:.0f}s)")
            results[seed][mode] = acc

    print(f"\n{'='*80}")
    print(f"  RESULTS — H={H} Cyc3")
    print(f"{'='*80}")
    print(f"  {'':>12} |", " | ".join(f"s={s}" for s in SEEDS), "| avg")
    for mode in MODES:
        row = f"  {mode:>12} |"
        accs = []
        for seed in SEEDS:
            a = results[seed][mode]
            accs.append(a)
            row += f" {a:.3f} |"
        row += f" {np.mean(accs):.3f}"
        print(row)

    # Compare to H=64 results
    print(f"\n  Scale comparison:")
    print(f"    H=64  search space: {64*64:,} positions")
    print(f"    H=256 search space: {256*256:,} positions ({256*256//(64*64)}x bigger)")
    rooted_avg = np.mean([results[s]['rooted'] for s in SEEDS])
    single_avg = np.mean([results[s]['single'] for s in SEEDS])
    edges_avg = np.mean([results[s]['edges_only'] for s in SEEDS])
    print(f"    rooted advantage: {rooted_avg - single_avg:+.3f}")
    print(f"    edges advantage:  {edges_avg - single_avg:+.3f}")
    if rooted_avg - single_avg > edges_avg - single_avg + 0.01:
        print(f"    → STRUCTURE MATTERS at H={H}!")
    else:
        print(f"    → Still edge-count dominant at H={H}")


if __name__ == "__main__":
    main()
