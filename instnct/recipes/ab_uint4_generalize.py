"""
INSTNCT — Uint4 Stamina Generalization
========================================
Does the winning spec (stamina=uint4, drain=[1-6], regen=2) work
on ALL tasks, not just L2 Alt?

Tasks: Const, Alt, Cyc3, English. Each: with and without stamina.
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
MAX_STAMINA = 15; REGEN_PERIOD = 2; H = 64

TEXT = ("a stitch in time saves nine. the early bird catches the worm. "
       "all that glitters is not gold. actions speak louder than words. "
       "fortune favors the bold. knowledge is power. practice makes perfect. "
       "the pen is mightier than the sword. where there is a will there is a way. ")
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

TASKS = [
    ("Const",   make_constant),
    ("Alt",     make_alternating),
    ("Cyc3",    make_cycle3),
    ("English", make_english),
]


def eval_seq(net, seq, edge_drain, use_stamina):
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n, MAX_STAMINA, dtype=np.int16) if use_stamina else None
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))
    ed = edge_drain[:n] if edge_drain is not None else None
    correct = 0; total = 0

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
                    stamina[:] = np.clip(stamina + 1, 0, MAX_STAMINA)
                s_mult = stamina.astype(np.float32) / float(MAX_STAMINA)
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
            if stamina is not None and ed is not None and len(rows):
                fs = fired[rows]
                if np.any(fs):
                    stamina[fs] = np.clip(stamina[fs] - ed[fs], 0, MAX_STAMINA)
        state = act; charge = cur_charge
        if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def train_and_eval(task_name, make_fn, seed, steps):
    """Train one network per task, then eval with/without stamina."""
    random.seed(seed); np.random.seed(seed)
    master_rng = np.random.RandomState(seed + 77)
    train_seqs = [make_fn(np.random.RandomState(seed + i), 40) for i in range(3)]
    eval_seqs = [make_fn(master_rng, 40) for _ in range(4)]

    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    n_edges = len(net.alive)
    drain = np.random.randint(1, 7, size=n_edges).astype(np.int16)

    def resize():
        nonlocal drain
        n = len(net.alive)
        if n > len(drain):
            drain = np.concatenate([drain, np.random.randint(1, 7, size=n - len(drain)).astype(np.int16)])
        elif n < len(drain):
            drain = drain[:n]

    # Train WITH stamina (the winning config)
    def avg_acc_stamina():
        resize()
        return np.mean([eval_seq(net, s, drain, True) for s in train_seqs])

    best = avg_acc_stamina()
    accepts = 0; stale = 0
    for step in range(1, steps + 1):
        if stale >= 400:
            nodes = random.sample(range(H), random.randint(3, 5))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0
            new = avg_acc_stamina()
            if new > best: best = new; accepts += 1
        snap = net.save_state()
        old_drain = drain.copy()
        net.mutate(); resize()
        for _ in range(random.randint(1, 5)):
            if len(drain) == 0: break
            idx = random.randint(0, len(drain) - 1)
            drain[idx] = np.clip(drain[idx] + random.choice([-1, 0, 1]), 1, 6)
        new = avg_acc_stamina()
        if new > best:
            best = new; accepts += 1; stale = 0
        else:
            net.restore_state(snap); drain = old_drain; resize(); stale += 1

    # Eval on held-out seqs: WITH and WITHOUT stamina
    resize()
    acc_on = np.mean([eval_seq(net, s, drain, True) for s in eval_seqs])
    acc_off = np.mean([eval_seq(net, s, drain, False) for s in eval_seqs])

    return acc_on, acc_off, accepts, np.mean(drain), np.std(drain)


def main():
    STEPS = 3000; SEED = 42

    print("=" * 80)
    print("  Uint4 Stamina Generalization: does it help on ALL tasks?")
    print(f"  Spec: stamina=[0-15], drain=[1-6]/edge, regen=2tick")
    print(f"  H={H} | Steps={STEPS}")
    print("=" * 80)

    results = []
    for task_name, make_fn in TASKS:
        print(f"\n  {task_name}...", end="", flush=True)
        t0 = time.time()
        acc_on, acc_off, accepts, d_mean, d_std = train_and_eval(
            task_name, make_fn, SEED, STEPS)
        elapsed = time.time() - t0
        delta = acc_on - acc_off
        print(f" ON={acc_on:.3f} OFF={acc_off:.3f} delta={delta:+.3f} | "
              f"drain={d_mean:.1f}±{d_std:.1f} | acc#={accepts} | {elapsed:.0f}s")
        results.append((task_name, acc_on, acc_off, delta, d_mean, d_std, accepts))

    print(f"\n{'='*80}")
    print(f"  RESULTS")
    print(f"{'='*80}")
    print(f"  {'Task':>8} | {'Stam ON':>7} | {'Stam OFF':>8} | {'Delta':>6} | {'Drain':>10} | {'Helps?':>6}")
    wins = 0; losses = 0; ties = 0
    for name, on, off, delta, dm, ds, _ in results:
        h = "YES" if delta > 0.005 else ("NO" if delta < -0.005 else "TIE")
        if delta > 0.005: wins += 1
        elif delta < -0.005: losses += 1
        else: ties += 1
        print(f"  {name:>8} | {on:7.3f} | {off:8.3f} | {delta:+6.3f} | {dm:.1f}±{ds:.1f} | {h:>6}")

    print(f"\n  Helps: {wins}/{len(TASKS)} | Hurts: {losses}/{len(TASKS)} | Tie: {ties}/{len(TASKS)}")

    # Drain convergence across tasks
    drains = [r[4] for r in results]
    print(f"\n  Drain across tasks: {[f'{d:.1f}' for d in drains]}")
    print(f"  Mean={np.mean(drains):.1f} Std={np.std(drains):.1f}")
    if np.std(drains) < 0.5:
        print(f"  → CONVERGES: drain≈{np.mean(drains):.0f} universally")
    else:
        print(f"  → VARIES by task")


if __name__ == "__main__":
    main()
