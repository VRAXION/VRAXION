"""
INSTNCT — Burst7 Guided vs Random: universal sweep
====================================================
Is burst7_guided universally better than burst7_random?

Sweep: 3 network sizes × 4 tasks × 3 seeds = 36 configs.
Both arms get 7 ops per step. Guided targets via stamina+charge.
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
MAX_STAMINA = 15; REGEN_PERIOD = 2
STEPS = 2000; PLATEAU_WINDOW = 300

TEXT = ("a stitch in time saves nine. the early bird catches the worm. "
       "all that glitters is not gold. actions speak louder than words. "
       "fortune favors the bold. knowledge is power. practice makes perfect. ")
TEXT_BYTES = list(np.frombuffer(TEXT.encode('ascii'), dtype=np.uint8))

def make_constant(rng, n=30):
    v = rng.randint(0, 10); return [v] * (n + 1)
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

TASKS = [("Const", make_constant), ("Alt", make_alternating),
         ("Cyc3", make_cycle3), ("Eng", make_english)]
RANDOM_OPS = ['add', 'add_loop', 'remove', 'rewire', 'theta', 'decay', 'polarity']


def probe_state(net, seq):
    """Quick probe: get fire_rate and avg_charge per neuron."""
    H = net.H; net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc; n = len(rows)
    st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
    ref = np.zeros(H, dtype=np.int8)
    stamina = np.full(n, MAX_STAMINA, dtype=np.int16)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))
    fires = np.zeros(H); charges = np.zeros(H); ticks_total = 0

    for idx in range(min(15, len(seq) - 1)):
        inj = net.input_projection[int(seq[idx])]
        act = st.copy(); cur = ch.copy()
        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0: cur = np.maximum(cur - 1.0, 0.0)
            if tick < INPUT_DURATION: act = act + inj
            if n > 0:
                if tick % REGEN_PERIOD == 0:
                    stamina[:] = np.clip(stamina + 1, 0, MAX_STAMINA)
                s_m = stamina.astype(np.float32) / float(MAX_STAMINA)
                raw = np.zeros(H, dtype=np.float32)
                np.add.at(raw, cols, act[rows] * s_m)
            else: raw = np.zeros(H, dtype=np.float32)
            cur += raw; np.clip(cur, 0.0, 15.0, out=cur)
            eff = np.clip(net._theta_f32 * SelfWiringGraph.WAVE_LUT[net.channel, tick % 8], 1.0, 15.0)
            can = (ref == 0); fired = (cur >= eff) & can
            ref[ref > 0] -= 1; ref[fired] = 1
            act = fired.astype(np.float32) * net._polarity_f32
            cur[fired] = 0.0
            if n > 0:
                fs = fired[rows]
                if np.any(fs): stamina[fs] = np.clip(stamina[fs] - 1, 0, MAX_STAMINA)
            fires += fired; charges += cur; ticks_total += 1
        st = act; ch = cur

    fr = fires / max(ticks_total, 1)
    ac = charges / max(ticks_total, 1)
    return fr, ac, stamina


def guided_op(net, fr, ac, stamina):
    """Pick ONE guided op based on diagnostics."""
    H = net.H; alive = net.alive
    r = random.random()

    if r < 0.25 and len(alive) > 0 and len(stamina) == len(alive):
        # REMOVE highest stamina (unused)
        idx = int(np.argmax(stamina[:len(alive)]))
        return 'remove'
    elif r < 0.50:
        # ADD to pressure point
        pressure = ac * (1.0 - fr); pressure[pressure < 0] = 0
        if np.max(pressure) > 0:
            return 'add'
    elif r < 0.70 and len(alive) > 0:
        return 'rewire'
    elif r < 0.85:
        # THETA based on fire rate
        always = np.where(fr > 0.5)[0]
        never = np.where((fr < 0.02) & (ac > 0.05))[0]
        if len(always) > 0: return 'theta'
        elif len(never) > 0: return 'theta'
    else:
        return random.choice(['decay', 'polarity', 'add_loop'])

    return random.choice(RANDOM_OPS)  # fallback


def eval_acc(net, seqs):
    """Standard eval."""
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
    """Train one config."""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    probe_seq = task_fn(np.random.RandomState(seed + 99), 30)

    best = eval_acc(net, eval_seqs); accepts = 0; stale = 0

    for step in range(1, STEPS + 1):
        if stale >= PLATEAU_WINDOW:
            nodes = random.sample(range(H), min(random.randint(3, 5), H))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0

        snap = net.save_state()

        if mode == 'burst7_random':
            for _ in range(7):
                net.mutate(forced_op=random.choice(RANDOM_OPS))
        elif mode == 'burst7_guided':
            fr, ac, stam = probe_state(net, probe_seq)
            for _ in range(7):
                op = guided_op(net, fr, ac, stam)
                net.mutate(forced_op=op)
        elif mode == 'single':
            net.mutate()

        new = eval_acc(net, eval_seqs)
        if new > best:
            best = new; accepts += 1; stale = 0
        else:
            net.restore_state(snap); stale += 1

    return best, accepts


def main():
    H_OPTIONS = [32, 64, 128]
    SEEDS = [42, 123, 999]
    MODES = ['single', 'burst7_random', 'burst7_guided']

    print("=" * 100)
    print("  Burst7 Guided vs Random: universal sweep")
    print(f"  H: {H_OPTIONS} | Seeds: {SEEDS} | Tasks: {len(TASKS)} | Steps: {STEPS}")
    print(f"  Modes: {MODES}")
    print("=" * 100)

    # Pre-generate eval seqs (same for all arms within a task)
    task_eval = {}
    for tn, tf in TASKS:
        task_eval[tn] = [tf(np.random.RandomState(77 + i), 30) for i in range(3)]

    # Results: [mode][task] = list of (H, seed, acc)
    all_results = {m: {tn: [] for tn, _ in TASKS} for m in MODES}

    total = len(H_OPTIONS) * len(SEEDS) * len(TASKS) * len(MODES)
    done = 0

    for H in H_OPTIONS:
        for seed in SEEDS:
            for tn, tf in TASKS:
                for mode in MODES:
                    done += 1
                    print(f"  [{done}/{total}] H={H} s={seed} {tn:>5} {mode:>15}...", end="", flush=True)
                    t0 = time.time()
                    acc, accepts = run_config(H, tf, seed, mode, task_eval[tn])
                    print(f" acc={acc:.3f} acc#={accepts} ({time.time()-t0:.0f}s)")
                    all_results[mode][tn].append((H, seed, acc))

    # Summary per task
    print(f"\n{'='*100}")
    print(f"  RESULTS — per task average")
    print(f"{'='*100}")
    print(f"  {'':>15} |", " | ".join(f"{tn:>7}" for tn, _ in TASKS), "| overall")

    mode_overalls = {}
    for mode in MODES:
        row = f"  {mode:>15} |"
        task_avgs = []
        for tn, _ in TASKS:
            accs = [r[2] for r in all_results[mode][tn]]
            avg = np.mean(accs)
            task_avgs.append(avg)
            row += f" {avg:7.3f} |"
        overall = np.mean(task_avgs)
        mode_overalls[mode] = overall
        row += f" {overall:.3f}"
        print(row)

    # Summary per H
    print(f"\n  Per network size:")
    print(f"  {'':>15} |", " | ".join(f"H={h:>3}" for h in H_OPTIONS), "| avg")
    for mode in MODES:
        row = f"  {mode:>15} |"
        h_avgs = []
        for H in H_OPTIONS:
            accs = [r[2] for r in sum(all_results[mode].values(), []) if r[0] == H]
            avg = np.mean(accs) if accs else 0
            h_avgs.append(avg)
            row += f" {avg:5.3f} |"
        row += f" {np.mean(h_avgs):.3f}"
        print(row)

    # Winner
    winner = max(mode_overalls, key=mode_overalls.get)
    base = mode_overalls['single']
    print(f"\n  WINNER: {winner} ({mode_overalls[winner]:.3f}, +{mode_overalls[winner]-base:.3f} vs single)")

    # Wins per task
    print(f"\n  Per-task winners:")
    for tn, _ in TASKS:
        task_mode_avg = {m: np.mean([r[2] for r in all_results[m][tn]]) for m in MODES}
        w = max(task_mode_avg, key=task_mode_avg.get)
        print(f"    {tn}: {w} ({task_mode_avg[w]:.3f})")


if __name__ == "__main__":
    main()
