"""
INSTNCT — Pattern Matching Seed
================================
No pain, no reward, no target FR. Just patterns.

Feed repeating sequences, ask "what comes next?"
Start trivial, advance only when mastered.

Levels:
  L1: Constant  — "5 5 5 5 5 ?" → 5
  L2: Alternating — "3 7 3 7 3 ?" → 7
  L3: Cycle-3   — "1 5 7 1 5 7 1 5 ?" → 7
  L4: Cycle-4   — "2 8 4 1 2 8 4 ?" → 1

Vocab: 0-9 (10 symbols). Tiny, learnable.
Growth: 32 → 64 → 128 after mastery.
"""
import sys, time, random, argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256    # network projection size (standard)
TICKS = 8
INPUT_DURATION = 2
MASTERY = 0.50  # 50% accuracy to pass (random = 10% for 10 symbols)


# --- Pattern generators ---

def make_constant(rng, n=48):
    """5 5 5 5 5... predict next = 5"""
    val = rng.randint(0, 10)
    seq = [val] * (n + 1)
    return seq

def make_alternating(rng, n=48):
    """3 7 3 7 3 7... predict next"""
    a, b = rng.randint(0, 10, size=2)
    while b == a:
        b = rng.randint(0, 10)
    seq = [a if i % 2 == 0 else b for i in range(n + 1)]
    return seq

def make_cycle3(rng, n=48):
    """1 5 7 1 5 7... predict next"""
    vals = list(rng.choice(10, size=3, replace=False))
    seq = [vals[i % 3] for i in range(n + 1)]
    return seq

def make_cycle4(rng, n=48):
    """2 8 4 1 2 8 4 1... predict next"""
    vals = list(rng.choice(10, size=4, replace=False))
    seq = [vals[i % 4] for i in range(n + 1)]
    return seq

LEVELS = [
    ('L1:Const',  make_constant),
    ('L2:Alt',    make_alternating),
    ('L3:Cyc3',   make_cycle3),
    ('L4:Cyc4',   make_cycle4),
]


# --- Eval: feed sequence, count correct next-predictions ---

def eval_sequences(net, seqs):
    """Feed sequences, measure prediction accuracy.

    For each sequence [a, b, c, d, ...]:
      inject a → predict b, inject b → predict c, etc.
    Score = fraction correct.
    """
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    correct = 0
    total = 0

    for seq in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)

        for i in range(len(seq) - 1):
            inp = int(seq[i])
            target = int(seq[i + 1])

            injected = net.input_projection[inp]
            state, charge = SelfWiringGraph.rollout_token(
                injected, mask=net.mask, theta=net._theta_f32,
                decay=net.decay, ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )

            logits = charge @ net.output_projection
            pred = int(np.argmax(logits))
            if pred == target:
                correct += 1
            total += 1

    return correct / total if total else 0.0


# --- Training ---

OPS = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'remove']


def train_level(net, level_name, make_fn, rng, max_steps=10000):
    """Train on one level until mastery or budget."""
    # Generate multiple eval sets for stability
    eval_sets = [make_fn(rng) for _ in range(5)]

    def avg_acc():
        return np.mean([eval_sequences(net, [s]) for s in eval_sets])

    best_acc = avg_acc()
    accepts = 0
    log_every = max(200, max_steps // 30)

    for step in range(1, max_steps + 1):
        op = OPS[(step - 1) % len(OPS)]
        if op in ('rewire', 'remove', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)
        new_acc = avg_acc()

        if new_acc > best_acc:
            best_acc = new_acc
            accepts += 1
        else:
            net.restore_state(state_snap)

        if step % log_every == 0:
            fr = _quick_fr(net, eval_sets[0])
            print(f"    {level_name} step {step:5d} | acc={best_acc:.3f} | edges={len(net.alive)} "
                  f"| FR={fr:.3f} | acc#={accepts}")

        if best_acc >= MASTERY:
            print(f"    >>> MASTERY at step {step}: acc={best_acc:.3f} ({accepts} accepts)")
            return best_acc, accepts, step, True

    print(f"    >>> Budget exhausted: acc={best_acc:.3f} ({accepts} accepts)")
    return best_acc, accepts, max_steps, False


def _quick_fr(net, seq):
    """Quick firing rate check."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    for i in range(min(10, len(seq) - 1)):
        injected = net.input_projection[int(seq[i])]
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=net.mask, theta=net._theta_f32,
            decay=net.decay, ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sc,
            polarity=net._polarity_f32, refractory=net.refractory,
            channel=net.channel,
        )
    return float(np.mean(np.abs(state) > 0))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-h", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10000, help="Steps per level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="checkpoints")
    args = ap.parse_args()

    CKPT_DIR = Path(args.out)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Growth: double after each mastered level
    growth_sizes = [args.seed_h, args.seed_h * 2, args.seed_h * 4, args.seed_h * 8]

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=args.seed_h, density=4,
        theta_init=1, decay_init=0.10, seed=args.seed,
    )

    print("=" * 80)
    print("  Pattern Matching Seed")
    print(f"  Vocab: 0-9 | Mastery: {MASTERY*100:.0f}% | Steps/level: {args.steps}")
    print(f"  Levels: Const → Alt → Cycle3 → Cycle4")
    print(f"  Start: H={args.seed_h}, edges={len(net.alive)}")
    print("=" * 80)

    results = []
    for li, (lname, make_fn) in enumerate(LEVELS):
        print(f"\n  --- {lname} (H={net.H}, edges={len(net.alive)}) ---")
        acc, accepts, steps, mastered = train_level(
            net, lname, make_fn, rng, max_steps=args.steps,
        )
        results.append({
            'level': lname, 'acc': acc, 'mastered': mastered,
            'steps': steps, 'accepts': accepts, 'H': net.H,
            'edges': len(net.alive),
        })

        # Save checkpoint after each level
        ckpt = CKPT_DIR / f"pattern_{lname.lower().replace(':', '_')}.npz"
        net.save(str(ckpt))

        # Grow if mastered and there's a bigger size
        if mastered and li + 1 < len(growth_sizes):
            new_H = growth_sizes[li + 1]
            if new_H > net.H:
                print(f"\n  >>> GROWTH: {net.H} → {new_H}")
                net = grow(net, new_H, rng)
                print(f"      edges={len(net.alive)}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  RESULTS")
    print(f"{'='*80}")
    print(f"  {'Level':>10} | {'H':>4} | {'Acc':>5} | {'Master':>6} | {'Steps':>5} | {'Edges':>5}")
    total_mastered = 0
    for r in results:
        m = 'YES' if r['mastered'] else 'no'
        if r['mastered']:
            total_mastered += 1
        print(f"  {r['level']:>10} | {r['H']:4d} | {r['acc']:5.3f} | {m:>6} | {r['steps']:5d} | {r['edges']:5d}")

    print(f"\n  Mastered: {total_mastered}/{len(LEVELS)}")
    print(f"  Random baseline: 10.0% (1/10 symbols)")
    if total_mastered == len(LEVELS):
        print(f"  ALL LEVELS MASTERED")
    elif total_mastered > 0:
        last = results[total_mastered - 1]
        print(f"  Stuck at level {total_mastered + 1}")


if __name__ == "__main__":
    main()
