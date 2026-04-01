"""
INSTNCT — Universal Stamina: fixed physics, no tuning
======================================================
The drain/regen is PHYSICS, not a hyperparameter.
One rule: fire = -1, rest = +1 every 6 ticks. Period.

Test across DIFFERENT tasks and sequence lengths to prove
it's universal (not seq-length dependent).

Tasks:
  - L1 Const (trivial, short)
  - L2 Alt (sequential, medium)
  - L3 Cycle3 (harder, long)
  - English text bytes (real, long)

Each task: same stamina physics, compare stamina ON vs OFF.
If stamina helps on ALL tasks → it's universal physics.
If only on some → it's task-dependent tuning (bad).
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
H = 64

# FIXED PHYSICS — not tunable
DRAIN = 1        # -1 per fire (always)
REGEN_PERIOD = 6 # +1 every 6 ticks (always, matches DECAY_PERIOD)


# --- Tasks ---

def make_constant(rng, n):
    v = rng.randint(0, 10)
    return [v] * (n + 1)

def make_alternating(rng, n):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]

def make_cycle3(rng, n):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]

TEXT = ("a stitch in time saves nine. the early bird catches the worm. "
       "all that glitters is not gold. actions speak louder than words. "
       "fortune favors the bold. knowledge is power. practice makes perfect. "
       "the pen is mightier than the sword. where there is a will there is a way. ")
TEXT_BYTES = list(np.frombuffer(TEXT.encode('ascii'), dtype=np.uint8))

def make_english(rng, n):
    start = rng.randint(0, max(1, len(TEXT_BYTES) - n - 1))
    return TEXT_BYTES[start:start + n + 1]

TASKS = [
    ("Const(20)",  make_constant,    20),
    ("Const(60)",  make_constant,    60),
    ("Alt(20)",    make_alternating, 20),
    ("Alt(60)",    make_alternating, 60),
    ("Cyc3(20)",   make_cycle3,      20),
    ("Cyc3(60)",   make_cycle3,      60),
    ("English(30)",make_english,     30),
    ("English(60)",make_english,     60),
]


# --- Eval with cross-token persistent stamina ---

def eval_task(net, seq, use_stamina):
    """One sequence, persistent state + optional persistent stamina."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n_edges, 255, dtype=np.uint8) if use_stamina else None
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))

    correct = 0; total = 0
    for idx in range(len(seq) - 1):
        injected = net.input_projection[int(seq[idx])]
        act = state.copy()
        cur_charge = charge.copy()

        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0:
                cur_charge = np.maximum(cur_charge - 1.0, 0.0)
            if tick < INPUT_DURATION:
                act = act + injected

            if stamina is not None and len(rows):
                if tick % REGEN_PERIOD == 0:
                    stamina[:] = np.clip(stamina.astype(np.int16) + 1, 0, 255).astype(np.uint8)
                s_mult = stamina.astype(np.float32) / 255.0
                raw = np.zeros(H, dtype=np.float32)
                np.add.at(raw, cols, act[rows] * s_mult)
            elif len(rows):
                raw = np.zeros(H, dtype=np.float32)
                np.add.at(raw, cols, act[rows])
            else:
                raw = np.zeros(H, dtype=np.float32)

            cur_charge += raw
            np.clip(cur_charge, 0.0, 15.0, out=cur_charge)
            eff_theta = np.clip(
                net._theta_f32 * SelfWiringGraph.WAVE_LUT[net.channel, tick % 8],
                1.0, 15.0)
            can_fire = (refractory == 0)
            fired = (cur_charge >= eff_theta) & can_fire
            refractory[refractory > 0] -= 1
            refractory[fired] = 1
            act = fired.astype(np.float32) * net._polarity_f32
            cur_charge[fired] = 0.0

            if stamina is not None and len(rows):
                fs = fired[rows]
                if np.any(fs):
                    stamina[fs] = np.clip(
                        stamina[fs].astype(np.int16) - DRAIN, 0, 255).astype(np.uint8)

        state = act; charge = cur_charge
        pred = int(np.argmax(charge[PRED_NEURONS])) if max(PRED_NEURONS) < H else 0
        if pred == int(seq[idx + 1]):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    stam_info = None
    if stamina is not None:
        stam_info = (int(stamina.min()), int(stamina.max()), float(stamina.std()))
    return acc, stam_info


def main():
    SEED = 42; TRAIN_STEPS = 3000; PLATEAU_WINDOW = 400
    master_rng = np.random.RandomState(99)

    print("=" * 90)
    print("  Universal Stamina Test: fixed physics across all tasks")
    print(f"  Physics: drain=-{DRAIN}/fire, regen=+1/{REGEN_PERIOD}tick, mult=stamina/255")
    print(f"  H={H} | Train: {TRAIN_STEPS} steps | Loop injection at plateau")
    print("=" * 90)

    # Train ONE network (no stamina, standard fast path)
    print(f"\n  Training base network...")
    random.seed(SEED); np.random.seed(SEED)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=SEED)

    # Use Alt(30) as training task (middle difficulty)
    train_seqs = [make_alternating(np.random.RandomState(i), 30) for i in range(3)]

    def quick_eval():
        net.reset()
        sc = SelfWiringGraph.build_sparse_cache(net.mask)
        accs = []
        for seq in train_seqs:
            state = np.zeros(H, dtype=np.float32)
            charge = np.zeros(H, dtype=np.float32)
            correct = 0; total = 0
            for i in range(len(seq) - 1):
                state, charge = SelfWiringGraph.rollout_token(
                    net.input_projection[int(seq[i])], mask=net.mask,
                    theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                    input_duration=INPUT_DURATION, state=state, charge=charge,
                    sparse_cache=sc, polarity=net._polarity_f32,
                    refractory=net.refractory, channel=net.channel)
                if int(np.argmax(charge[PRED_NEURONS])) == int(seq[i+1]):
                    correct += 1
                total += 1
            accs.append(correct / total if total else 0)
        return np.mean(accs)

    best = quick_eval(); accepts = 0; stale = 0
    t0 = time.time()
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
    print(f"  Done in {time.time()-t0:.0f}s: acc={best:.3f}, edges={len(net.alive)}")

    # Now eval this SAME network on ALL tasks, with and without stamina
    print(f"\n  Evaluating on all tasks (stamina ON vs OFF)...")
    print(f"  {'Task':>14} | {'OFF':>5} | {'ON':>5} | {'Delta':>6} | {'Stamina':>20}")
    print(f"  {'-'*14}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*20}")

    wins = 0; losses = 0; ties = 0
    for task_name, make_fn, seq_len in TASKS:
        seqs = [make_fn(np.random.RandomState(77 + i), seq_len) for i in range(4)]

        acc_off = np.mean([eval_task(net, s, False)[0] for s in seqs])
        results_on = [eval_task(net, s, True) for s in seqs]
        acc_on = np.mean([r[0] for r in results_on])
        stam = results_on[0][1]

        delta = acc_on - acc_off
        stam_str = f"[{stam[0]},{stam[1]}] std={stam[2]:.1f}" if stam else ""
        marker = "+" if delta > 0.005 else ("-" if delta < -0.005 else "=")
        print(f"  {task_name:>14} | {acc_off:5.3f} | {acc_on:5.3f} | {delta:+6.3f} {marker} | {stam_str}")

        if delta > 0.005: wins += 1
        elif delta < -0.005: losses += 1
        else: ties += 1

    print(f"\n  Stamina helps: {wins}/{len(TASKS)} tasks")
    print(f"  Stamina hurts: {losses}/{len(TASKS)} tasks")
    print(f"  Neutral: {ties}/{len(TASKS)} tasks")

    if wins > losses:
        print(f"\n  VERDICT: UNIVERSAL — stamina helps more than hurts")
    elif losses > wins:
        print(f"\n  VERDICT: NOT UNIVERSAL — stamina hurts more tasks")
    else:
        print(f"\n  VERDICT: INCONCLUSIVE — mixed results")


if __name__ == "__main__":
    main()
