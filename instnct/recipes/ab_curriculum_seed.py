"""
INSTNCT — Curriculum Seed: staged learning then growth
=======================================================
Train seed on simplest possible tasks first, grow only after mastery.

Levels:
  L1: Echo (A → A) — can the network pass signal through?
  L2: Pairs (0→1, 2→3, 4→5...) — can it transform?
  L3: Shift (A → A+1 mod 256) — generalized transform?

Each level: train until mastery (>50% acc), then grow.
Growth: 32 → 64 → 128

Expectation signal active throughout (pain/reward/boredom).
"""
import sys, time, random, argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256  # network vocab (projections); curriculum uses subsets
TICKS = 8
INPUT_DURATION = 2
TARGET_FR = 0.15
EMA_ALPHA = 0.3
MASTERY_THRESHOLD = 0.30   # 30% accuracy to pass level


# --- Curriculum levels ---
# Vocab scales with level: start tiny, grow as network grows.
# L1: 8 symbols (H=32 can handle this)
# L2: 16 symbols (H=64 after growth)
# L3: 32 symbols (H=128 after growth)

def make_echo_pairs(rng, n=32):
    """L1: A → A (vocab=8)"""
    src = rng.randint(0, 8, size=n)
    return list(zip(src, src))


def make_pair_pairs(rng, n=32):
    """L2: 0→1, 2→3... (vocab=16, 8 fixed pairs)"""
    src = rng.randint(0, 8, size=n) * 2  # even: 0,2,4,6,8,10,12,14
    tgt = src + 1
    return list(zip(src, tgt))


def make_shift_pairs(rng, n=32):
    """L3: A → (A+1) mod 32 (vocab=32)"""
    src = rng.randint(0, 32, size=n)
    tgt = (src + 1) % 32
    return list(zip(src, tgt))


LEVELS = [
    ('L1:Echo8',   make_echo_pairs),
    ('L2:Pair16',  make_pair_pairs),
    ('L3:Shift32', make_shift_pairs),
]


# --- Expectation tracker ---

class ExpTracker:
    def __init__(self):
        self.expected_fr = TARGET_FR
        self.expected_sat = 0.25

    def update(self, fr, sat):
        err_fr = fr - self.expected_fr
        err_sat = sat - self.expected_sat
        self.expected_fr += EMA_ALPHA * err_fr
        self.expected_sat += EMA_ALPHA * err_sat
        return err_fr - err_sat

    def copy(self):
        t = ExpTracker()
        t.expected_fr = self.expected_fr
        t.expected_sat = self.expected_sat
        return t


def compute_sat(charge, edge_count):
    if edge_count < 1:
        return 0.0
    return min(float(np.sum(charge)) / max(float(edge_count), 1.0), 1.0)


# --- Eval with expectation signal ---

def eval_pairs(net, pairs, tracker):
    """Evaluate on byte pairs with expectation signal."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    trk = tracker.copy()
    correct = 0

    for a, b in pairs:
        data_inj = net.input_projection[int(a)]
        # Phase 1: data (2 ticks)
        state, charge = SelfWiringGraph.rollout_token(
            data_inj, mask=net.mask, theta=net._theta_f32, decay=net.decay,
            ticks=2, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sc,
            polarity=net._polarity_f32, refractory=net.refractory,
            channel=net.channel,
        )
        # Expectation signal
        sat = compute_sat(charge, len(net.alive))
        fr = float(np.mean(np.abs(state) > 0))
        delta = trk.update(fr, sat)
        sig = np.zeros(H, dtype=np.float32)
        if delta < -0.001:
            sig[0] = min(abs(delta) * 10.0, 3.0)
        elif delta > 0.001:
            sig[1] = min(delta * 10.0, 3.0)
        else:
            sig[0] = 0.5
        # Phase 2: signal + propagation (6 ticks)
        state, charge = SelfWiringGraph.rollout_token(
            sig, mask=net.mask, theta=net._theta_f32, decay=net.decay,
            ticks=TICKS - 2, input_duration=1,
            state=state, charge=charge, sparse_cache=sc,
            polarity=net._polarity_f32, refractory=net.refractory,
            channel=net.channel,
        )
        logits = charge @ net.output_projection
        if int(np.argmax(logits)) == int(b):
            correct += 1

    acc = correct / len(pairs) if pairs else 0.0
    homeo = -abs(fr - TARGET_FR)
    return acc, homeo, fr


def eval_no_signal(net, pairs):
    """Evaluate without expectation signal (clean task eval)."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0

    for a, b in pairs:
        data_inj = net.input_projection[int(a)]
        state, charge = SelfWiringGraph.rollout_token(
            data_inj, mask=net.mask, theta=net._theta_f32, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sc,
            polarity=net._polarity_f32, refractory=net.refractory,
            channel=net.channel,
        )
        logits = charge @ net.output_projection
        if int(np.argmax(logits)) == int(b):
            correct += 1

    return correct / len(pairs) if pairs else 0.0


# --- Growth ---

def grow(old_net, new_H, rng):
    old_H = old_net.H
    new_net = SelfWiringGraph(
        vocab=VOCAB, hidden=new_H, density=4,
        theta_init=1, decay_init=0.10,
        seed=int(rng.randint(0, 2**31)),
    )
    new_net.mask[:old_H, :old_H] = old_net.mask[:old_H, :old_H]
    new_net.theta[:old_H] = old_net.theta
    new_net._theta_f32[:old_H] = old_net._theta_f32
    new_net.decay[:old_H] = old_net.decay
    new_net.polarity[:old_H] = old_net.polarity
    new_net._polarity_f32[:old_H] = old_net._polarity_f32
    new_net.channel[:old_H] = old_net.channel
    new_net.resync_alive()
    new_net.reset()
    return new_net


# --- Training loop ---

OPS = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'remove']


def train_level(net, level_name, make_pairs_fn, tracker, rng,
                max_steps=10000, eval_every=200):
    """Train on one curriculum level until mastery or budget exhausted."""
    # Multiple eval sets for stability
    eval_sets = [make_pairs_fn(rng, n=32) for _ in range(3)]

    def avg_acc():
        accs = [eval_pairs(net, pairs, tracker)[0] for pairs in eval_sets]
        return np.mean(accs)

    best_acc = avg_acc()
    best_ho = 0.0
    accepts = 0

    for step in range(1, max_steps + 1):
        op = OPS[(step - 1) % len(OPS)]
        if op in ('rewire', 'remove', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)

        # Eval with signal
        accs = []
        for pairs in eval_sets:
            a, h, f = eval_pairs(net, pairs, tracker)
            accs.append(a)
        new_acc = np.mean(accs)

        if new_acc > best_acc:
            best_acc = new_acc
            best_ho = h
            accepts += 1
        else:
            net.restore_state(state_snap)

        if step % eval_every == 0:
            print(f"    {level_name} step {step:5d} | acc={best_acc:.3f} "
                  f"| FR={f:.3f} | edges={len(net.alive)} | acc#={accepts}")

        if best_acc >= MASTERY_THRESHOLD:
            print(f"    >>> MASTERY at step {step}: acc={best_acc:.3f}")
            return best_acc, accepts, step, True

    print(f"    >>> Budget exhausted: acc={best_acc:.3f}")
    return best_acc, accepts, max_steps, False


def run_curriculum(label, init_H, growth_schedule, steps_per_level, seed):
    """Run full curriculum: level up, grow, repeat."""
    rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)
    tracker = ExpTracker()

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=init_H, density=4,
        theta_init=1, decay_init=0.10, seed=seed,
    )

    print(f"\n  [{label}] Starting at H={init_H}, edges={len(net.alive)}")
    results = []

    for li, (lname, make_fn) in enumerate(LEVELS):
        print(f"\n  --- {lname} (H={net.H}) ---")
        acc, accs, steps, mastered = train_level(
            net, lname, make_fn, tracker, rng,
            max_steps=steps_per_level,
        )
        results.append({
            'level': lname, 'acc': acc, 'mastered': mastered,
            'steps': steps, 'accepts': accs, 'H': net.H,
            'edges': len(net.alive),
        })

        # Grow after mastery (if there's a next size)
        if mastered and li < len(LEVELS) - 1:
            gh = growth_schedule.get(li)
            if gh and gh > net.H:
                print(f"\n  >>> GROWTH: {net.H} → {gh}")
                net = grow(net, gh, rng)
                print(f"      edges={len(net.alive)}")

    return net, results


def run_baseline(label, H, steps_per_level, seed):
    """Baseline: random init at full size, same curriculum, no growth."""
    rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)
    tracker = ExpTracker()

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=H, density=4,
        theta_init=1, decay_init=0.10, seed=seed,
    )

    print(f"\n  [{label}] Starting at H={H}, edges={len(net.alive)}")
    results = []

    for li, (lname, make_fn) in enumerate(LEVELS):
        print(f"\n  --- {lname} (H={net.H}) ---")
        acc, accs, steps, mastered = train_level(
            net, lname, make_fn, tracker, rng,
            max_steps=steps_per_level,
        )
        results.append({
            'level': lname, 'acc': acc, 'mastered': mastered,
            'steps': steps, 'accepts': accs, 'H': net.H,
            'edges': len(net.alive),
        })

    return net, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-h", type=int, default=32)
    ap.add_argument("--target-h", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10000, help="Steps per level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="checkpoints")
    args = ap.parse_args()

    CKPT_DIR = Path(args.out)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    growth = {0: 64, 1: args.target_h}  # grow after L1, L2

    print("=" * 90)
    print("  Curriculum Seed: train core to master Echo → Pairs → Shift")
    print(f"  Growth: {args.seed_h} → 64 → {args.target_h}")
    print(f"  Mastery threshold: {MASTERY_THRESHOLD*100:.0f}% | Steps/level: {args.steps}")
    print("=" * 90)

    t0 = time.time()
    net, results = run_curriculum(
        "SEED", args.seed_h, growth, args.steps, args.seed
    )
    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*90}")
    print(f"  RESULTS — Curriculum Seed Training")
    print(f"{'='*90}")
    print(f"\n  {'Level':>8} | {'H':>4} | {'Acc':>5} | {'Mastered':>8} | {'Steps':>5} | {'Edges':>5} | {'Accepts':>7}")
    total_mastered = 0
    for r in results:
        m = 'YES' if r['mastered'] else 'no'
        if r['mastered']:
            total_mastered += 1
        print(f"  {r['level']:>8} | {r['H']:4d} | {r['acc']:5.3f} | {m:>8} | {r['steps']:5d} | {r['edges']:5d} | {r['accepts']:7d}")

    # Clean eval on each level (no expectation signal)
    test_rng = np.random.RandomState(999)
    print(f"\n  Clean eval (no signal):")
    for lname, make_fn in LEVELS:
        pairs = make_fn(test_rng, n=64)
        clean_acc = eval_no_signal(net, pairs)
        print(f"    {lname}: {clean_acc:.4f}")

    # Neuron connectivity report
    p0 = int(np.sum(net.mask[0, :]))
    p1 = int(np.sum(net.mask[1, :]))
    print(f"\n  Pain neuron (0) outgoing: {p0}")
    print(f"  Reward neuron (1) outgoing: {p1}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Levels mastered: {total_mastered}/{len(LEVELS)}")

    # Save checkpoint
    ckpt_path = CKPT_DIR / "curriculum_seed_best.npz"
    net.save(str(ckpt_path))
    print(f"  Saved: {ckpt_path}")

    if total_mastered == len(LEVELS):
        print(f"\n  FULLY MATURE — all levels mastered!")
    elif total_mastered > 0:
        print(f"\n  PARTIAL — {total_mastered} level(s) mastered, keep training")
    else:
        print(f"\n  NOT YET — no levels mastered, needs tuning")


if __name__ == "__main__":
    main()
