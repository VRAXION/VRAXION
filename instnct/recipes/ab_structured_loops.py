"""
INSTNCT — Structured Loop Network: fixed topology, learnable params
====================================================================
Fixed architecture: N parallel 3-node loops, each wired to IO.
Only parameters mutate: theta, decay, channel, edge weights (int4).

Structure (for 4 loops, H=12 + IO):
  IN → [A→B→C→A] → OUT    (loop 1: neurons 0,1,2)
  IN → [D→E→F→D] → OUT    (loop 2: neurons 3,4,5)
  IN → [G→H→I→G] → OUT    (loop 3: neurons 6,7,8)
  IN → [J→K→L→J] → OUT    (loop 4: neurons 9,10,11)

  Each loop: IN feeds node 0 and 2, OUT reads from 0 and 2

Sweep: 2-8 loops × Alt/Cyc3 task × parameter mutation only.
Also test: int4 edge weights (1-15 strength) vs binary.
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


def make_alternating(rng, n=30):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]

def make_cycle3(rng, n=30):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]

TASKS = [("Alt", make_alternating), ("Cyc3", make_cycle3)]


def build_structured_net(n_loops, seed, use_weights=False):
    """Build a network with fixed loop topology."""
    H = n_loops * 3  # 3 neurons per loop
    if H < max(PRED_NEURONS) + 1:
        H = max(PRED_NEURONS) + 1  # ensure pred neurons exist

    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=0,
                          theta_init=3, decay_init=0.15, seed=seed)
    # Clear any edges from init
    net.mask[:] = False

    # Build loop topology
    for loop_idx in range(n_loops):
        a = loop_idx * 3
        b = loop_idx * 3 + 1
        c = loop_idx * 3 + 2
        if a >= H or b >= H or c >= H:
            continue
        # Internal loop: A→B→C→A
        net.mask[a, b] = True
        net.mask[b, c] = True
        net.mask[c, a] = True

    # Cross-loop connections: each loop's A connects to next loop's A
    for i in range(n_loops - 1):
        src = i * 3
        dst = (i + 1) * 3
        if src < H and dst < H:
            net.mask[src, dst] = True

    # Last loop connects back to first (global loop)
    if n_loops > 1:
        net.mask[(n_loops-1)*3, 0] = True

    net.resync_alive()

    # Edge weights (int4 if enabled)
    edge_weights = None
    if use_weights:
        edge_weights = np.full(len(net.alive), 8, dtype=np.int16)  # start at middle

    return net, edge_weights


def eval_structured(net, seq, edge_weights=None):
    """Eval with optional int4 edge weights."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n = len(rows)
    H = net.H
    st = np.zeros(H, dtype=np.float32)
    ch = np.zeros(H, dtype=np.float32)
    ref = np.zeros(H, dtype=np.int8)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))
    correct = 0; total = 0

    for idx in range(len(seq) - 1):
        inj = net.input_projection[int(seq[idx])]
        act = st.copy(); cur = ch.copy()
        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0:
                cur = np.maximum(cur - 1.0, 0.0)
            if tick < INPUT_DURATION:
                act = act + inj
            if n > 0:
                if edge_weights is not None:
                    w = edge_weights[:n].astype(np.float32) / 15.0  # normalize to 0-1
                    raw = np.zeros(H, dtype=np.float32)
                    np.add.at(raw, cols, act[rows] * w)
                else:
                    raw = np.zeros(H, dtype=np.float32)
                    np.add.at(raw, cols, act[rows])
            else:
                raw = np.zeros(H, dtype=np.float32)
            cur += raw; np.clip(cur, 0.0, 15.0, out=cur)
            eff = np.clip(net._theta_f32 * SelfWiringGraph.WAVE_LUT[net.channel, tick % 8], 1.0, 15.0)
            can = (ref == 0); fired = (cur >= eff) & can
            ref[ref > 0] -= 1; ref[fired] = 1
            act = fired.astype(np.float32) * net._polarity_f32
            cur[fired] = 0.0
        st = act; ch = cur
        if H > max(PRED_NEURONS) and int(np.argmax(ch[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def mutate_params_only(net, edge_weights=None):
    """Mutate ONLY parameters, never topology."""
    H = net.H
    undo = []

    # Theta drift: 1-3 random neurons
    for _ in range(random.randint(1, 3)):
        idx = random.randint(0, H - 1)
        old = int(net.theta[idx])
        new_val = np.clip(old + random.choice([-1, 0, 1]), 1, 15)
        undo.append(('T', idx, old))
        net.theta[idx] = np.uint8(new_val)
        net._theta_f32[idx] = float(new_val)

    # Channel drift: 1 neuron
    if random.random() < 0.3:
        idx = random.randint(0, H - 1)
        old = int(net.channel[idx])
        undo.append(('CH', idx, old))
        net.channel[idx] = np.uint8(random.randint(1, 8))

    # Decay drift: 1 neuron
    if random.random() < 0.3:
        idx = random.randint(0, H - 1)
        old = float(net.decay[idx])
        undo.append(('D', idx, old))
        net.decay[idx] = np.float32(np.clip(old + random.gauss(0, 0.02), 0.01, 0.5))

    # Polarity flip: rare
    if random.random() < 0.1:
        idx = random.randint(0, H - 1)
        old = bool(net.polarity[idx])
        undo.append(('P', idx, old))
        net.polarity[idx] = not old
        net._polarity_f32[idx] = -1.0 if old else 1.0

    # Edge weight drift (if enabled)
    if edge_weights is not None and len(edge_weights) > 0:
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, len(edge_weights) - 1)
            old = int(edge_weights[idx])
            undo.append(('W', idx, old))
            edge_weights[idx] = np.clip(old + random.choice([-1, 0, 1]), 1, 15)

    return undo


def undo_params(net, undo, edge_weights=None):
    for entry in reversed(undo):
        op = entry[0]
        if op == 'T':
            _, idx, old = entry
            net.theta[idx] = np.uint8(old)
            net._theta_f32[idx] = float(old)
        elif op == 'CH':
            _, idx, old = entry
            net.channel[idx] = np.uint8(old)
        elif op == 'D':
            _, idx, old = entry
            net.decay[idx] = np.float32(old)
        elif op == 'P':
            _, idx, old = entry
            net.polarity[idx] = old
            net._polarity_f32[idx] = 1.0 if old else -1.0
        elif op == 'W' and edge_weights is not None:
            _, idx, old = entry
            edge_weights[idx] = old


def run_config(n_loops, use_weights, task_fn, seed, steps, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net, ew = build_structured_net(n_loops, seed, use_weights)
    H = net.H; edges = len(net.alive)

    best = eval_structured(net, eval_seqs[0], ew)
    accepts = 0

    for step in range(1, steps + 1):
        undo = mutate_params_only(net, ew)
        accs = [eval_structured(net, s, ew) for s in eval_seqs]
        new = np.mean(accs)
        if new > best:
            best = new; accepts += 1
        else:
            undo_params(net, undo, ew)

    return best, accepts, H, edges


def main():
    STEPS = 3000; SEEDS = [42, 123]
    LOOP_OPTIONS = [2, 4, 6, 8, 10]

    eval_data = {tn: [tf(np.random.RandomState(77+i), 30) for i in range(3)]
                 for tn, tf in TASKS}

    print("=" * 90)
    print("  Structured Loop Network: fixed topology, learnable params")
    print(f"  Loops: {LOOP_OPTIONS} | Steps: {STEPS} | Seeds: {SEEDS}")
    print(f"  Each loop: A→B→C→A + cross-loop chain + global loop-back")
    print("=" * 90)

    # Also run standard INSTNCT as baseline (same H, topology learns)
    results = []

    for n_loops in LOOP_OPTIONS:
        for tn, tf in TASKS:
            for use_w in [False, True]:
                w_label = "weighted" if use_w else "binary"
                accs = []
                for seed in SEEDS:
                    acc, accepts, H, edges = run_config(
                        n_loops, use_w, tf, seed, STEPS, eval_data[tn])
                    accs.append(acc)
                avg = np.mean(accs)
                results.append((n_loops, tn, w_label, avg, H, edges))
                print(f"  {n_loops:2d} loops H={H:3d} e={edges:3d} {tn:>5} {w_label:>8}: "
                      f"acc={avg:.3f}")

    # Baseline: standard INSTNCT (topology learns) at matching H
    print(f"\n  --- Baselines: standard INSTNCT (topology learns) ---")
    baselines = {}
    for n_loops in [4, 8]:
        H = max(n_loops * 3, max(PRED_NEURONS) + 1)
        for tn, tf in TASKS:
            accs = []
            for seed in SEEDS:
                random.seed(seed); np.random.seed(seed)
                net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                                      theta_init=1, decay_init=0.10, seed=seed)
                ev = eval_data[tn]
                best = np.mean([eval_structured(net, s) for s in ev])
                for step in range(1, STEPS+1):
                    snap = net.save_state(); net.mutate()
                    new = np.mean([eval_structured(net, s) for s in ev])
                    if new > best: best = new
                    else: net.restore_state(snap)
                accs.append(best)
            avg = np.mean(accs)
            baselines[(n_loops, tn)] = avg
            print(f"  standard H={H:3d} {tn:>5}: acc={avg:.3f}")

    # Summary
    print(f"\n{'='*90}")
    print(f"  RESULTS")
    print(f"{'='*90}")
    print(f"  {'Loops':>5} | {'H':>3} | {'Edges':>5} | {'Task':>5} | {'Type':>8} | {'Acc':>5}")
    print(f"  {'-'*5}-+-{'-'*3}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*5}")

    best_overall = 0; best_config = ""
    for n_loops, tn, w_label, avg, H, edges in results:
        marker = ""
        if avg > best_overall:
            best_overall = avg; best_config = f"{n_loops}L {w_label} {tn}"
        print(f"  {n_loops:5d} | {H:3d} | {edges:5d} | {tn:>5} | {w_label:>8} | {avg:5.3f}")

    print(f"\n  Best: {best_config} = {best_overall:.3f}")

    # Structured vs standard at same H
    print(f"\n  Structured vs Standard (same H):")
    for n_loops in [4, 8]:
        for tn, _ in TASKS:
            struct_bin = [r[3] for r in results if r[0]==n_loops and r[1]==tn and r[2]=='binary']
            struct_w = [r[3] for r in results if r[0]==n_loops and r[1]==tn and r[2]=='weighted']
            std = baselines.get((n_loops, tn), 0)
            sb = struct_bin[0] if struct_bin else 0
            sw = struct_w[0] if struct_w else 0
            print(f"    {n_loops}L {tn}: struct_bin={sb:.3f} struct_w={sw:.3f} standard={std:.3f} "
                  f"{'STRUCT' if max(sb,sw) > std else 'STANDARD'}")


if __name__ == "__main__":
    main()
