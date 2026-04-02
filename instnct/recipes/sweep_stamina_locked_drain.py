"""
INSTNCT — Comprehensive Stamina Sweep: locked drain, all tasks, telemetry
==========================================================================
Drain locked at 4 (converged value). Sweep stamina resolution across
all tasks with deep diagnostics.

Phase 1: Train 4 base networks (no stamina, standard fast path)
Phase 2: Eval-only sweep with stamina configs
Phase 3: Telemetry analysis
"""
import sys, time, random, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256; TICKS = 8; INPUT_DURATION = 2; H = 64
PRED_NEURONS = list(range(0, 10))
LOCKED_DRAIN = 4
REGEN_PERIOD = 2
TRAIN_STEPS = 3000; PLATEAU_WINDOW = 400

TEXT = ("a stitch in time saves nine. the early bird catches the worm. "
       "all that glitters is not gold. actions speak louder than words. "
       "fortune favors the bold. knowledge is power. practice makes perfect. ")
TEXT_BYTES = list(np.frombuffer(TEXT.encode('ascii'), dtype=np.uint8))


# --- Task generators ---
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

TASKS = [
    ("Const", make_constant),
    ("Alt",   make_alternating),
    ("Cyc3",  make_cycle3),
    ("Eng",   make_english),
]


# --- Phase 1: Train base network (no stamina) ---
def train_base(task_name, make_fn, seed):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    train_seqs = [make_fn(np.random.RandomState(seed + i), 40) for i in range(3)]

    def quick_eval():
        net.reset()
        sc = SelfWiringGraph.build_sparse_cache(net.mask)
        accs = []
        for seq in train_seqs:
            state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
            c = 0; t = 0
            for i in range(len(seq) - 1):
                state, charge = SelfWiringGraph.rollout_token(
                    net.input_projection[int(seq[i])], mask=net.mask,
                    theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                    input_duration=INPUT_DURATION, state=state, charge=charge,
                    sparse_cache=sc, polarity=net._polarity_f32,
                    refractory=net.refractory, channel=net.channel)
                if int(np.argmax(charge[PRED_NEURONS])) == int(seq[i+1]): c += 1
                t += 1
            accs.append(c / t if t else 0)
        return np.mean(accs)

    best = quick_eval(); stale = 0; accepts = 0
    for step in range(1, TRAIN_STEPS + 1):
        if stale >= PLATEAU_WINDOW:
            nodes = random.sample(range(H), random.randint(3, 5))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0
            new = quick_eval()
            if new > best: best = new; accepts += 1
        snap = net.save_state(); net.mutate()
        new = quick_eval()
        if new > best: best = new; accepts += 1; stale = 0
        else: net.restore_state(snap); stale += 1

    return net, best, accepts


# --- Phase 2: Eval with stamina (cross-token, telemetry) ---
def eval_with_telemetry(net, seq, stamina_max, init_mode):
    """Returns (accuracy, telemetry_dict)."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))

    if stamina_max is None:
        stamina = None
    else:
        init_val = stamina_max if init_mode == 'full' else stamina_max // 2
        stamina = np.full(n_edges, init_val, dtype=np.int16)

    correct = 0; total = 0
    token_frs = []
    token_charges = []

    for idx in range(len(seq) - 1):
        injected = net.input_projection[int(seq[idx])]
        act = state.copy(); cur_charge = charge.copy()

        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0:
                cur_charge = np.maximum(cur_charge - 1.0, 0.0)
            if tick < INPUT_DURATION:
                act = act + injected
            if stamina is not None and len(rows):
                if tick % REGEN_PERIOD == 0:
                    stamina[:] = np.clip(stamina + 1, 0, stamina_max)
                s_mult = stamina.astype(np.float32) / float(stamina_max)
                raw = np.zeros(H, dtype=np.float32)
                np.add.at(raw, cols, act[rows] * s_mult)
            elif len(rows):
                raw = np.zeros(H, dtype=np.float32)
                np.add.at(raw, cols, act[rows])
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
            if stamina is not None and len(rows):
                fs = fired[rows]
                if np.any(fs):
                    stamina[fs] = np.clip(stamina[fs] - LOCKED_DRAIN, 0, stamina_max)

        state = act; charge = cur_charge
        token_frs.append(float(np.mean(np.abs(state) > 0)))
        token_charges.append(float(np.mean(charge)))
        if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    telem = {
        'acc': acc,
        'fr_mean': float(np.mean(token_frs)),
        'fr_std': float(np.std(token_frs)),
        'charge_mean': float(np.mean(token_charges)),
    }
    if stamina is not None:
        telem['stam_min'] = int(stamina.min())
        telem['stam_max'] = int(stamina.max())
        telem['stam_mean'] = float(np.mean(stamina))
        telem['stam_std'] = float(np.std(stamina))
        telem['pct_zero'] = float(100 * np.sum(stamina == 0) / len(stamina))
        telem['pct_half'] = float(100 * np.sum(stamina < stamina_max * 0.5) / len(stamina))
        telem['n_edges'] = n_edges
    return acc, telem


def main():
    SEED = 42
    master_rng = np.random.RandomState(77)

    # Eval sequences per task (held-out)
    task_eval_seqs = {}
    for name, make_fn in TASKS:
        task_eval_seqs[name] = [make_fn(master_rng, 40) for _ in range(4)]

    # Stamina configs
    configs = [
        ("none",      None,  'full'),
        ("uint2_f",   3,     'full'),
        ("uint2_h",   3,     'half'),
        ("uint4_f",   15,    'full'),
        ("uint4_h",   15,    'half'),
        ("uint5_f",   31,    'full'),
        ("uint6_f",   63,    'full'),
        ("uint8_f",   255,   'full'),
        ("uint4_h",   15,    'half'),
        ("uint8_h",   255,   'half'),
    ]
    # Deduplicate
    seen = set()
    unique_configs = []
    for label, sm, im in configs:
        key = (label, sm, im)
        if key not in seen:
            seen.add(key); unique_configs.append((label, sm, im))
    configs = unique_configs

    print("=" * 100)
    print("  Comprehensive Stamina Sweep: drain locked at %d, all tasks" % LOCKED_DRAIN)
    print(f"  H={H} | Train: {TRAIN_STEPS} steps | Regen: every {REGEN_PERIOD} ticks")
    print(f"  Configs: {len(configs)} | Tasks: {len(TASKS)}")
    print("=" * 100)

    # Phase 1: Train base networks
    print(f"\n  === PHASE 1: Training base networks (no stamina) ===")
    trained_nets = {}
    for name, make_fn in TASKS:
        print(f"    {name}...", end="", flush=True)
        t0 = time.time()
        net, acc, accepts = train_base(name, make_fn, SEED)
        print(f" acc={acc:.3f} edges={len(net.alive)} acc#={accepts} ({time.time()-t0:.0f}s)")
        trained_nets[name] = net

    # Phase 2: Eval sweep
    print(f"\n  === PHASE 2: Stamina sweep (eval-only) ===\n")

    # Header
    col_w = 7
    header = f"  {'Config':>10} |"
    for name, _ in TASKS:
        header += f" {name:>{col_w}} |"
    header += " avg"
    print(header)
    print(f"  {'-'*10}-+" + ("-" * (col_w) + "-+") * len(TASKS) + "----")

    all_telemetry = {}

    for cfg_label, stam_max, init_mode in configs:
        row = f"  {cfg_label:>10} |"
        accs = []
        for task_name, _ in TASKS:
            net = trained_nets[task_name]
            seqs = task_eval_seqs[task_name]
            task_accs = []
            task_telems = []
            for seq in seqs:
                a, t = eval_with_telemetry(net, seq, stam_max, init_mode)
                task_accs.append(a)
                task_telems.append(t)
            avg_acc = np.mean(task_accs)
            accs.append(avg_acc)
            row += f" {avg_acc:{col_w}.3f} |"
            # Store telemetry
            key = f"{cfg_label}_{task_name}"
            all_telemetry[key] = {
                'acc': avg_acc,
                'telem': task_telems[-1],  # last seq telemetry
            }
        avg = np.mean(accs)
        row += f" {avg:.3f}"
        if avg == max(np.mean([all_telemetry.get(f"{c[0]}_{t[0]}", {}).get('acc', 0)
                               for t in TASKS]) for c in configs if f"{c[0]}_{TASKS[0][0]}" in all_telemetry):
            row += " <<<"
        print(row)

    # Phase 3: Telemetry dump
    print(f"\n  === PHASE 3: Telemetry (best config per task) ===\n")
    for task_name, _ in TASKS:
        best_cfg = None; best_acc = -1
        for cfg_label, stam_max, init_mode in configs:
            key = f"{cfg_label}_{task_name}"
            if key in all_telemetry and all_telemetry[key]['acc'] > best_acc:
                best_acc = all_telemetry[key]['acc']
                best_cfg = cfg_label
        key = f"{best_cfg}_{task_name}"
        t = all_telemetry[key]['telem']
        print(f"  {task_name} best={best_cfg} acc={best_acc:.3f}")
        if 'stam_min' in t:
            print(f"    stamina: [{t['stam_min']},{t['stam_max']}] mean={t['stam_mean']:.1f} std={t['stam_std']:.1f}")
            print(f"    depleted: {t['pct_zero']:.1f}% at zero, {t['pct_half']:.1f}% below half")
        print(f"    firing: FR={t['fr_mean']:.3f}±{t['fr_std']:.3f}, charge={t['charge_mean']:.3f}")

    # Final verdict
    print(f"\n  === VERDICT ===\n")
    # Best across ALL tasks
    cfg_avgs = {}
    for cfg_label, _, _ in configs:
        accs = []
        for task_name, _ in TASKS:
            key = f"{cfg_label}_{task_name}"
            if key in all_telemetry:
                accs.append(all_telemetry[key]['acc'])
        if accs:
            cfg_avgs[cfg_label] = np.mean(accs)

    for cfg, avg in sorted(cfg_avgs.items(), key=lambda x: -x[1]):
        marker = " <<<" if avg == max(cfg_avgs.values()) else ""
        print(f"    {cfg:>10}: {avg:.3f}{marker}")

    winner = max(cfg_avgs, key=cfg_avgs.get)
    baseline = cfg_avgs.get('none', 0)
    print(f"\n  Winner: {winner} ({cfg_avgs[winner]:.3f}, +{cfg_avgs[winner]-baseline:.3f} vs baseline)")


if __name__ == "__main__":
    main()
