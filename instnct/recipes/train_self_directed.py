"""
INSTNCT — Self-Directed Evolution
==================================
The network outputs THREE things:
  1. Prediction (neurons 0-9): what symbol comes next?
  2. Confidence (neuron 10): how sure am I?
  3. Mutation vote (neurons 11-15): what should change about me?

Nobody tells the network what these neurons mean.
The fitness pressure teaches it:
  - Confidence: rewarded for calibrated uncertainty
  - Mutation: we DO whatever it votes for → good votes get selected

Patterns: Const → Alt → Cycle3 (same curriculum as pattern seed)
"""
import sys, time, random, argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256
TICKS = 8
INPUT_DURATION = 2
MASTERY = 0.45

# Readout neuron assignments (by charge value after rollout)
PRED_NEURONS = list(range(0, 10))     # prediction: 10 symbols
CONF_NEURON = 10                       # confidence
MUT_NEURONS = list(range(11, 16))      # mutation vote
MUT_OPS = ['add', 'remove', 'rewire', 'theta', 'decay']


# --- Pattern generators (same as pattern seed) ---

def make_constant(rng, n=48):
    val = rng.randint(0, 10)
    return [val] * (n + 1)

def make_alternating(rng, n=48):
    a, b = rng.randint(0, 10, size=2)
    while b == a:
        b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]

def make_cycle3(rng, n=48):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]

LEVELS = [
    ('L1:Const', make_constant),
    ('L2:Alt',   make_alternating),
    ('L3:Cyc3',  make_cycle3),
]


# --- Core eval: prediction + confidence scoring ---

def eval_with_confidence(net, seqs):
    """Evaluate prediction accuracy with confidence calibration.

    Reads charge directly from hidden neurons after rollout:
      - PRED_NEURONS[0:10]: argmax = predicted symbol
      - CONF_NEURON: confidence (normalized charge)
      - MUT_NEURONS[0:5]: mutation vote

    Fitness = prediction_score + calibration_score
    """
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    pred_correct = 0
    calibration_score = 0.0
    total = 0
    mut_votes = np.zeros(len(MUT_OPS), dtype=np.float32)

    for seq in seqs:
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

            # Read prediction from charge of pred neurons
            pred_charges = charge[PRED_NEURONS] if max(PRED_NEURONS) < H else np.zeros(10)
            pred = int(np.argmax(pred_charges))
            correct = (pred == target)
            if correct:
                pred_correct += 1

            # Read confidence from charge of conf neuron
            if CONF_NEURON < H:
                raw_conf = float(charge[CONF_NEURON])
                conf = raw_conf / SelfWiringGraph.MAX_CHARGE  # normalize 0-1
            else:
                conf = 0.5

            # Calibration: reward honest confidence
            if correct:
                calibration_score += conf       # right + confident = good
            else:
                calibration_score += (1.0 - conf)  # wrong + humble = also good

            # Accumulate mutation votes
            if max(MUT_NEURONS) < H:
                for mi, mn in enumerate(MUT_NEURONS):
                    mut_votes[mi] += float(charge[mn])

            total += 1

    acc = pred_correct / total if total else 0.0
    cal = calibration_score / total if total else 0.0

    # Composite fitness: prediction matters most, calibration is bonus
    score = 0.7 * acc + 0.3 * cal

    # Determine voted mutation
    voted_op = MUT_OPS[int(np.argmax(mut_votes))]

    return score, acc, cal, voted_op, conf


# --- Training ---

def train_level(net, level_name, make_fn, rng, max_steps=10000):
    """Train with full default mutate() — network's own mutation drive decides."""
    eval_sets = [make_fn(rng) for _ in range(5)]

    def avg_eval():
        scores, accs, cals = [], [], []
        for s in eval_sets:
            sc, ac, ca, vo, co = eval_with_confidence(net, [s])
            scores.append(sc); accs.append(ac); cals.append(ca)
        return np.mean(scores), np.mean(accs), np.mean(cals)

    best_score, best_acc, best_cal = avg_eval()
    accepts = 0
    log_every = max(200, max_steps // 25)

    for step in range(1, max_steps + 1):
        # Use full default mutate() — includes loss_pct drift, drive drift,
        # theta, polarity, channel, and adaptive add/remove/rewire
        state_snap = net.save_state()
        undo = net.mutate()  # <-- full default, no forced_op

        new_score, new_acc, new_cal = avg_eval()

        if new_score > best_score:
            best_score = new_score
            best_acc = new_acc
            best_cal = new_cal
            accepts += 1
        else:
            net.restore_state(state_snap)

        if step % log_every == 0:
            drive = int(net.mutation_drive)
            drive_str = f"+{drive}" if drive > 0 else str(drive)
            print(f"    {level_name} step {step:5d} | acc={best_acc:.3f} | cal={best_cal:.3f} "
                  f"| score={best_score:.3f} | edges={len(net.alive)} | drive={drive_str} "
                  f"| acc#={accepts}")

        if best_acc >= MASTERY:
            print(f"    >>> MASTERY at step {step}: acc={best_acc:.3f} cal={best_cal:.3f} "
                  f"drive={int(net.mutation_drive)} edges={len(net.alive)}")
            return best_acc, best_cal, accepts, step, True

    print(f"    >>> Budget exhausted: acc={best_acc:.3f} cal={best_cal:.3f}")
    return best_acc, best_cal, accepts, max_steps, False


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

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=args.seed_h, density=4,
        theta_init=1, decay_init=0.10, seed=args.seed,
    )

    growth_sizes = [args.seed_h * 2, args.seed_h * 4]

    print("=" * 90)
    print("  Self-Directed Evolution")
    print(f"  H={args.seed_h} | Neurons 0-9=pred, 10=confidence, 11-15=mutation vote")
    print(f"  The network votes on its own mutations!")
    print(f"  Mastery: {MASTERY*100:.0f}% | Steps/level: {args.steps}")
    print("=" * 90)

    results = []
    for li, (lname, make_fn) in enumerate(LEVELS):
        print(f"\n  --- {lname} (H={net.H}, edges={len(net.alive)}) ---")
        acc, cal, accepts, steps, mastered = train_level(
            net, lname, make_fn, rng, max_steps=args.steps,
        )
        results.append({
            'level': lname, 'acc': acc, 'cal': cal, 'mastered': mastered,
            'steps': steps, 'accepts': accepts, 'H': net.H,
            'edges': len(net.alive), 'drive': int(net.mutation_drive),
        })

        ckpt = CKPT_DIR / f"self_directed_{lname.lower().replace(':', '_')}.npz"
        net.save(str(ckpt))

        if mastered and li < len(growth_sizes):
            new_H = growth_sizes[li]
            if new_H > net.H:
                print(f"\n  >>> GROWTH: {net.H} → {new_H}")
                net = grow(net, new_H, rng)
                print(f"      edges={len(net.alive)}")

    # Summary
    print(f"\n{'='*90}")
    print(f"  RESULTS — Self-Directed Evolution")
    print(f"{'='*90}")
    print(f"  {'Level':>10} | {'H':>4} | {'Acc':>5} | {'Cal':>5} | {'Master':>6} | {'Steps':>5} | {'Accepts':>7}")
    total_mastered = 0
    for r in results:
        m = 'YES' if r['mastered'] else 'no'
        if r['mastered']:
            total_mastered += 1
        print(f"  {r['level']:>10} | {r['H']:4d} | {r['acc']:5.3f} | {r['cal']:5.3f} | {m:>6} | {r['steps']:5d} | {r['accepts']:7d}")

    # Drive analysis
    print(f"\n  Mutation drive (learned):")
    for r in results:
        d = r['drive']
        mode = "ADD" if d > 0 else ("REMOVE" if d < 0 else "REWIRE")
        print(f"    {r['level']}: drive={d:+d} ({mode})")

    print(f"\n  Mastered: {total_mastered}/{len(LEVELS)}")


if __name__ == "__main__":
    main()
