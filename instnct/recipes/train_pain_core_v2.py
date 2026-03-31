"""
INSTNCT — Pain Core v2: Internal Pain Sensing
==============================================
The network SENSES its own saturation as an input signal and learns
to self-regulate. Pain is not an external fitness function — it's
an input the network can read and react to.

Architecture:
  - Hidden neurons [0..H-1]
  - Neuron 0: "pain receptor" — receives saturation signal each tick
  - Neuron 1: "data input" — receives projected byte
  - Output: readout from charge @ output_projection
  - Saturation = global_voltage / ceiling(edge_count)

Training target (minimal):
  Feed byte pairs (A, B). After injecting A + pain, the network
  should predict B. But the REAL test: does the network use the
  pain channel to self-regulate, or does it ignore it?

Usage:
    python instnct/recipes/train_pain_core_v2.py
    python instnct/recipes/train_pain_core_v2.py --hidden 32 --steps 30000
"""
import sys, time, random, argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# --- Defaults ---
VOCAB = 256
TICKS = 8
INPUT_DURATION = 2
N_PAIRS = 16             # byte pairs per eval round
N_EVAL_SETS = 3          # averaged for stability
TARGET_FR = 0.15
MAX_CHARGE = SelfWiringGraph.MAX_CHARGE  # 15.0

# Mutation schedule
OPS_EXPLORE = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'remove']
OPS_REFINE = ['rewire', 'theta', 'theta', 'decay', 'rewire', 'theta', 'decay', 'rewire']


def compute_saturation(charge, edge_count):
    """How full is the network relative to its capacity?

    ceiling = edge_count: if every source neuron spiked, this many
    charge units would flow in one tick. Global voltage / ceiling =
    saturation ratio (0.0 = empty, 1.0+ = overloaded).
    """
    if edge_count < 1:
        return 0.0
    global_voltage = float(np.sum(charge))
    ceiling = float(edge_count)
    return min(global_voltage / ceiling, 1.0)


def inject_pain(hidden_charge, edge_count, H, pain_scale=3.0):
    """Create pain injection vector: saturation signal into neuron 0.

    Neuron 0 = pain receptor. Gets charge proportional to saturation.
    Higher saturation → stronger pain signal → neuron 0 fires more →
    (hopefully) the network learns to throttle itself.
    """
    pain_vec = np.zeros(H, dtype=np.float32)
    sat = compute_saturation(hidden_charge, edge_count)
    pain_vec[0] = sat * pain_scale  # scale so it can actually trigger firing
    return pain_vec, sat


def rollout_with_pain(net, data_injected, sparse_cache, state, charge):
    """One token rollout with pain signal injected mid-tick.

    Tick 0-1: data input (normal)
    Tick 2: pain signal injected based on current charge state
    Tick 3-7: free propagation

    We do this by running two mini-rollouts:
      1) ticks 0-2: data input (duration=2)
      2) inject pain based on charge after data
      3) ticks 2-7: continue with pain added to charge
    """
    H = net.H
    edge_count = len(net.alive)

    # Phase 1: inject data for 2 ticks
    state, charge = SelfWiringGraph.rollout_token(
        data_injected,
        mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=2, input_duration=INPUT_DURATION,
        state=state, charge=charge, sparse_cache=sparse_cache,
        polarity=net._polarity_f32, refractory=net.refractory,
        channel=net.channel,
    )

    # Measure saturation after data injection
    pain_vec, saturation = inject_pain(charge, edge_count, H)

    # Phase 2: inject pain + continue propagation for remaining ticks
    state, charge = SelfWiringGraph.rollout_token(
        pain_vec,
        mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=TICKS - 2, input_duration=1,  # pain injected for 1 tick
        state=state, charge=charge, sparse_cache=sparse_cache,
        polarity=net._polarity_f32, refractory=net.refractory,
        channel=net.channel,
    )

    return state, charge, saturation


def eval_network(net, byte_pairs, output_projection):
    """Evaluate: prediction accuracy + internal health metrics.

    For each (A, B) pair:
      1. Inject A + pain signal
      2. Read output, see if it predicts B
      3. Track saturation, firing rate, charge

    Returns composite score and detailed metrics.
    """
    net.reset()
    H = net.H
    sparse_cache = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    correct = 0
    total = 0
    saturations = []
    firing_rates = []
    mean_charges = []

    for a_byte, b_byte in byte_pairs:
        # Inject byte A through projection
        data_inj = net.input_projection[int(a_byte)]

        # Rollout with internal pain sensing
        state, charge, sat = rollout_with_pain(
            net, data_inj, sparse_cache, state, charge
        )

        saturations.append(sat)
        fr = float(np.mean(np.abs(state) > 0))
        firing_rates.append(fr)
        mean_charges.append(float(np.mean(charge)))

        # Readout: does output predict B?
        logits = charge @ output_projection
        pred = int(np.argmax(logits))
        if pred == int(b_byte):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    avg_fr = np.mean(firing_rates)
    avg_ch = np.mean(mean_charges)
    avg_sat = np.mean(saturations)

    # Composite fitness:
    # 1. Prediction accuracy (minimal — just learning to use inputs)
    # 2. Homeostasis (FR near target)
    # 3. Saturation health (not too high, not dead)
    #    Sweet spot: 0.1-0.4 saturation (active but not overloaded)
    homeo = -abs(avg_fr - TARGET_FR)
    sat_health = -abs(avg_sat - 0.25)  # target 25% saturation
    metab = avg_fr / (avg_ch + 0.01)
    metab = min(metab, 2.0)

    score = (
        0.5 * acc +          # can it predict at all?
        1.0 * homeo +        # alive and breathing?
        0.5 * sat_health +   # healthy saturation?
        0.2 * metab          # efficient?
    )

    return score, acc, homeo, sat_health, metab, avg_fr, avg_ch, avg_sat


def main():
    ap = argparse.ArgumentParser(description="Pain Core v2: internal pain sensing")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--density", type=float, default=4)
    ap.add_argument("--theta", type=int, default=1)
    ap.add_argument("--decay", type=float, default=0.10)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--out", type=str, default="checkpoints")
    args = ap.parse_args()

    STEPS = args.steps
    SEED = args.seed
    CKPT_DIR = Path(args.out)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Fixed eval byte pairs (A→B) — multiple sets for stability
    eval_sets = []
    for _ in range(N_EVAL_SETS):
        pairs = [(rng.randint(0, 256), rng.randint(0, 256)) for _ in range(N_PAIRS)]
        eval_sets.append(pairs)

    def eval_multi(net):
        """Average over multiple pair sets."""
        results = [eval_network(net, pairs, net.output_projection) for pairs in eval_sets]
        return tuple(np.mean([r[i] for r in results]) for i in range(8))

    # Init
    if args.resume:
        net = SelfWiringGraph.load(args.resume)
        print(f"Resumed from {args.resume} (H={net.H}, edges={len(net.alive)})")
    else:
        net = SelfWiringGraph(
            vocab=VOCAB, hidden=args.hidden, density=args.density,
            theta_init=args.theta, decay_init=args.decay, seed=SEED,
        )

    best_score, best_acc, best_ho, best_sat, best_me, best_fr, best_ch, best_sv = eval_multi(net)
    best_ever = best_score
    accepts = 0
    stale = 0
    phase = 'EXPLORE'

    print(f"=== Pain Core v2: Internal Pain Sensing ===")
    print(f"H={net.H} | density={args.density}% | theta={args.theta}")
    print(f"Steps={STEPS} | pairs={N_PAIRS}x{N_EVAL_SETS}")
    print(f"Neuron 0 = pain receptor (saturation input)")
    print(f"Target: FR={TARGET_FR}, saturation=0.25")
    print()
    hdr = (f"{'step':>6} | {'ph':>3} | {'edges':>5} | {'acc':>5} | {'FR':>5} "
           f"| {'sat':>5} | {'chg':>5} | {'homeo':>7} | {'sat_h':>6} "
           f"| {'score':>7} | {'acc#':>4} | {'stale':>5}")
    print(hdr)
    print("-" * 100)

    t0 = time.time()
    log_every = max(1, STEPS // 40)

    for step in range(1, STEPS + 1):
        if phase == 'EXPLORE' and best_ho > -0.03:
            phase = 'REFINE'

        ops = OPS_EXPLORE if phase == 'EXPLORE' else OPS_REFINE
        op = ops[(step - 1) % len(ops)]
        if op in ('rewire', 'remove', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)
        sc, ac, ho, sh, me, fr, ch, sv = eval_multi(net)

        if sc > best_score:
            best_score, best_acc, best_ho, best_sat = sc, ac, ho, sh
            best_me, best_fr, best_ch, best_sv = me, fr, ch, sv
            accepts += 1
            stale = 0
            if sc > best_ever:
                best_ever = sc
        else:
            net.restore_state(state_snap)
            stale += 1

        # Stale → save + rotate
        if stale == 2000:
            path = CKPT_DIR / f"pain_v2_stale_{step}.npz"
            net.save(str(path))
            print(f"  >>> Stale at step {step}, saved {path.name}")
            eval_sets.append([(rng.randint(0, 256), rng.randint(0, 256))
                              for _ in range(N_PAIRS)])
            if len(eval_sets) > 5:
                eval_sets.pop(0)
            best_score, best_acc, best_ho, best_sat, best_me, best_fr, best_ch, best_sv = eval_multi(net)
            stale = 0

        if step % log_every == 0 or step == 1:
            phs = 'EXP' if phase == 'EXPLORE' else 'REF'
            print(f"{step:6d} | {phs:>3} | {len(net.alive):5d} | {best_acc:5.3f} | {best_fr:5.3f} "
                  f"| {best_sv:5.3f} | {best_ch:5.3f} | {best_ho:7.4f} | {best_sat:6.4f} "
                  f"| {best_score:7.4f} | {accepts:4d} | {stale:5d}")

    elapsed = time.time() - t0

    # Save
    final_path = CKPT_DIR / "pain_v2_best.npz"
    net.save(str(final_path))

    print(f"\n{'='*100}")
    print(f"  DONE in {elapsed:.1f}s ({elapsed/STEPS*1000:.1f}ms/step)")
    print(f"  Saved: {final_path}")
    print(f"{'='*100}")

    print(f"\n--- Core v2 Quality Report ---")
    print(f"  Accuracy:     {best_acc:.4f}  (random = 1/256 = 0.004)")
    print(f"  Firing rate:  {best_fr:.4f}  (target: {TARGET_FR})")
    print(f"  Saturation:   {best_sv:.4f}  (target: 0.25)")
    print(f"  Homeostasis:  {best_ho:+.4f}")
    print(f"  Mean charge:  {best_ch:.4f}")
    print(f"  Edges:        {len(net.alive)}  ({100*len(net.alive)/(net.H*net.H):.1f}%)")
    print(f"  Accepts:      {accepts}/{STEPS} ({100*accepts/STEPS:.1f}%)")

    # Check: does neuron 0 (pain receptor) have incoming edges from itself?
    # And outgoing edges to others? (sign it's being used)
    pain_in = int(np.sum(net.mask[:, 0]))   # edges INTO neuron 0
    pain_out = int(np.sum(net.mask[0, :]))  # edges FROM neuron 0
    print(f"\n  Pain receptor (neuron 0):")
    print(f"    Incoming edges: {pain_in}")
    print(f"    Outgoing edges: {pain_out}")
    used = pain_out > 0
    print(f"    Connected to network: {'YES — pain signal propagates' if used else 'NO — isolated'}")

    mature = best_ho > -0.02 and best_fr > 0.05 and best_sv < 0.5
    print(f"\n  Maturity: {'MATURE' if mature else 'NOT YET'}")


if __name__ == "__main__":
    main()
