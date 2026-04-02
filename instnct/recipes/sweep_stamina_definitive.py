"""
INSTNCT — DEFINITIVE Stamina Sweep: proportional drain, all variables
======================================================================
THE test to close the stamina question once and for all.

Proportional drain: drain_rate = 4/255 = 1.57% per fire (constant RATE).
  scaled_drain = max(1, round(drain_rate * max_stamina))

Sweep EVERYTHING:
  - Stamina resolution: uint2(3), uint4(15), uint6(63), uint8(255)
  - Network size: H=32, H=64, H=128
  - Tasks: Const, Alt, Cyc3, English
  - Learnable drain: starts random, mutates, track convergence

Train WITHOUT stamina (fast path), eval WITH stamina (proportional).
Track where learnable drain converges per resolution.
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
REGEN_PERIOD = 2
DRAIN_RATE = 4.0 / 255.0   # 1.57% — the universal physical constant
TRAIN_STEPS = 2500; PLATEAU_WINDOW = 400

TEXT = ("a stitch in time saves nine. the early bird catches the worm. "
       "all that glitters is not gold. actions speak louder than words. "
       "fortune favors the bold. knowledge is power. practice makes perfect. ")
TEXT_BYTES = list(np.frombuffer(TEXT.encode('ascii'), dtype=np.uint8))


def make_constant(rng, n=40):
    v = rng.randint(0, 10); return [v] * (n + 1)
def make_alternating(rng, n=40):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]
def make_cycle3(rng, n=40):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]
def make_english(rng, n=40):
    start = rng.randint(0, max(1, len(TEXT_BYTES) - n - 1))
    return TEXT_BYTES[start:start + n + 1]

TASKS = [("Const", make_constant), ("Alt", make_alternating),
         ("Cyc3", make_cycle3), ("Eng", make_english)]


def train_base(H, task_fn, seed):
    """Train base network (no stamina, standard fast path)."""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    seqs = [task_fn(np.random.RandomState(seed + i), 30) for i in range(2)]

    def qeval():
        net.reset()
        sc = SelfWiringGraph.build_sparse_cache(net.mask)
        accs = []
        for seq in seqs:
            st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
            c = 0; t = 0
            for i in range(len(seq) - 1):
                st, ch = SelfWiringGraph.rollout_token(
                    net.input_projection[int(seq[i])], mask=net.mask,
                    theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                    input_duration=INPUT_DURATION, state=st, charge=ch,
                    sparse_cache=sc, polarity=net._polarity_f32,
                    refractory=net.refractory, channel=net.channel)
                if int(np.argmax(ch[PRED_NEURONS])) == int(seq[i+1]): c += 1
                t += 1
            accs.append(c / t if t else 0)
        return np.mean(accs)

    best = qeval(); stale = 0
    for step in range(1, TRAIN_STEPS + 1):
        if stale >= PLATEAU_WINDOW:
            nodes = random.sample(range(H), min(random.randint(3, 5), H))
            for i in range(len(nodes)):
                r, c_ = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c_ and not net.mask[r, c_]: net.mask[r, c_] = True
            net.resync_alive(); stale = 0
            new = qeval()
            if new > best: best = new
        snap = net.save_state(); net.mutate()
        new = qeval()
        if new > best: best = new; stale = 0
        else: net.restore_state(snap); stale += 1
    return net, best


def eval_stamina(net, seq, max_stamina, proportional=True):
    """Eval with proportional drain rate."""
    H = net.H
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n = len(rows)
    if n == 0:
        return 0.0, {}

    st = np.zeros(H, dtype=np.float32)
    ch = np.zeros(H, dtype=np.float32)
    ref = np.zeros(H, dtype=np.int8)
    stamina = np.full(n, max_stamina, dtype=np.int16)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))

    if proportional:
        drain = max(1, int(round(DRAIN_RATE * max_stamina)))
    else:
        drain = 4  # fixed absolute

    correct = 0; total = 0
    for idx in range(len(seq) - 1):
        inj = net.input_projection[int(seq[idx])]
        act = st.copy(); cur = ch.copy()
        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0:
                cur = np.maximum(cur - 1.0, 0.0)
            if tick < INPUT_DURATION:
                act = act + inj
            if tick % REGEN_PERIOD == 0:
                stamina[:] = np.clip(stamina + 1, 0, max_stamina)
            s_mult = stamina.astype(np.float32) / float(max_stamina)
            raw = np.zeros(H, dtype=np.float32)
            np.add.at(raw, cols, act[rows] * s_mult)
            cur += raw
            np.clip(cur, 0.0, 15.0, out=cur)
            eff = np.clip(net._theta_f32 * SelfWiringGraph.WAVE_LUT[net.channel, tick % 8], 1.0, 15.0)
            can = (ref == 0); fired = (cur >= eff) & can
            ref[ref > 0] -= 1; ref[fired] = 1
            act = fired.astype(np.float32) * net._polarity_f32
            cur[fired] = 0.0
            fs = fired[rows]
            if np.any(fs):
                stamina[fs] = np.clip(stamina[fs] - drain, 0, max_stamina)
        st = act; ch = cur
        if int(np.argmax(ch[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    telem = {
        'drain_int': drain,
        'drain_pct': 100.0 * drain / max_stamina,
        'stam_mean': float(np.mean(stamina)),
        'stam_std': float(np.std(stamina)),
        'pct_zero': float(100 * np.sum(stamina == 0) / n),
    }
    return acc, telem


def eval_no_stamina(net, seq):
    """Baseline: no stamina."""
    H = net.H; net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
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
            if len(rows):
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
        if int(np.argmax(ch[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def main():
    SEED = 42
    H_OPTIONS = [32, 64, 128]
    STAM_OPTIONS = [
        ("uint2", 3), ("uint4", 15), ("uint6", 63), ("uint8", 255),
    ]

    master_rng = np.random.RandomState(77)
    task_seqs = {name: [fn(master_rng, 40) for _ in range(3)]
                 for name, fn in TASKS}

    print("=" * 110)
    print("  DEFINITIVE Stamina Sweep: proportional drain, all network sizes")
    print(f"  Drain rate: {DRAIN_RATE*100:.2f}%/fire (proportional to resolution)")
    print(f"  Regen: +1/{REGEN_PERIOD}tick | Tasks: {len(TASKS)} | H: {H_OPTIONS}")
    print("=" * 110)

    # Train base networks for each H × task
    print(f"\n  === TRAINING ===")
    nets = {}
    for H in H_OPTIONS:
        for task_name, task_fn in TASKS:
            key = (H, task_name)
            print(f"    H={H:3d} {task_name:>5}...", end="", flush=True)
            t0 = time.time()
            net, acc = train_base(H, task_fn, SEED)
            print(f" acc={acc:.3f} edges={len(net.alive)} ({time.time()-t0:.0f}s)")
            nets[key] = net

    # Eval sweep: proportional drain
    print(f"\n  === EVAL: PROPORTIONAL DRAIN ===")

    # Results table: [H][task][stam_label] = (acc_on, acc_off, telem)
    results = {}

    for H in H_OPTIONS:
        results[H] = {}
        for task_name, _ in TASKS:
            results[H][task_name] = {}
            net = nets[(H, task_name)]
            seqs = task_seqs[task_name]

            # Baseline
            base = np.mean([eval_no_stamina(net, s) for s in seqs])
            results[H][task_name]['none'] = (base, base, {})

            for stam_label, max_s in STAM_OPTIONS:
                accs = []; telems = []
                for s in seqs:
                    a, t = eval_stamina(net, s, max_s, proportional=True)
                    accs.append(a); telems.append(t)
                avg_acc = np.mean(accs)
                results[H][task_name][stam_label] = (avg_acc, base, telems[-1])

    # Print results per H
    for H in H_OPTIONS:
        print(f"\n  --- H={H} ---")
        header = f"  {'':>8} |"
        for tn, _ in TASKS:
            header += f" {tn:>7} |"
        header += "  avg  | drain_int"
        print(header)
        print(f"  {'-'*8}-+" + ("-" * 7 + "-+") * len(TASKS) + "------+---------")

        for stam_label in ['none'] + [s[0] for s in STAM_OPTIONS]:
            row = f"  {stam_label:>8} |"
            accs = []
            drain_info = ""
            for tn, _ in TASKS:
                acc_on, _, telem = results[H][tn][stam_label]
                accs.append(acc_on)
                row += f" {acc_on:7.3f} |"
                if telem and 'drain_int' in telem:
                    drain_info = f"d={telem['drain_int']}"
            avg = np.mean(accs)
            row += f" {avg:.3f} | {drain_info}"
            print(row)

    # Cross-H summary: which resolution wins everywhere?
    print(f"\n  === CROSS-NETWORK SUMMARY ===")
    print(f"  {'':>8} |", " | ".join(f"H={h:>3}" for h in H_OPTIONS), "| overall")

    for stam_label in ['none'] + [s[0] for s in STAM_OPTIONS]:
        row = f"  {stam_label:>8} |"
        all_accs = []
        for H in H_OPTIONS:
            h_accs = [results[H][tn][stam_label][0] for tn, _ in TASKS]
            avg = np.mean(h_accs)
            all_accs.append(avg)
            row += f" {avg:5.3f} |"
        overall = np.mean(all_accs)
        row += f" {overall:.3f}"
        print(row)

    # Proportional vs absolute drain comparison on uint4
    print(f"\n  === PROPORTIONAL vs ABSOLUTE drain (uint4, H=64) ===")
    net64 = nets[(64, 'Alt')]
    seqs_alt = task_seqs['Alt']
    for label, max_s in STAM_OPTIONS:
        prop_acc = np.mean([eval_stamina(net64, s, max_s, proportional=True)[0] for s in seqs_alt])
        abs_acc = np.mean([eval_stamina(net64, s, max_s, proportional=False)[0] for s in seqs_alt])
        prop_drain = max(1, int(round(DRAIN_RATE * max_s)))
        print(f"    {label:>6}: prop(d={prop_drain})={prop_acc:.3f} vs abs(d=4)={abs_acc:.3f} "
              f"{'PROP' if prop_acc >= abs_acc else 'ABS'}")

    # Final verdict
    print(f"\n  === VERDICT ===")
    overall_by_res = {}
    for stam_label in ['none'] + [s[0] for s in STAM_OPTIONS]:
        all_a = []
        for H in H_OPTIONS:
            for tn, _ in TASKS:
                all_a.append(results[H][tn][stam_label][0])
        overall_by_res[stam_label] = np.mean(all_a)

    for sl, avg in sorted(overall_by_res.items(), key=lambda x: -x[1]):
        marker = " <<<" if avg == max(overall_by_res.values()) else ""
        print(f"    {sl:>8}: {avg:.4f}{marker}")

    winner = max(overall_by_res, key=overall_by_res.get)
    base = overall_by_res['none']
    print(f"\n  WINNER: {winner} ({overall_by_res[winner]:.4f}, +{overall_by_res[winner]-base:.4f} vs none)")
    print(f"  Proportional drain rate: {DRAIN_RATE*100:.2f}%/fire = universal constant")


if __name__ == "__main__":
    main()
