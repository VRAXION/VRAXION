"""
INSTNCT — Pain Core Training
=============================
Train a tiny seed network to maximum pain fitness.
Goal: mature core that has mastered homeostasis, metabolic efficiency,
and insight dynamics — ready for transplant into larger networks.

Saves checkpoint every plateau + final best.
No task data, no bigram — purely internal dynamics.

Usage:
    python instnct/recipes/train_pain_core.py
    python instnct/recipes/train_pain_core.py --hidden 16 --steps 50000
    python instnct/recipes/train_pain_core.py --resume checkpoints/pain_core_best.npz
"""
import sys, os, time, random, argparse
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
TARGET_FR = 0.15
N_PROBE_TOKENS = 16       # more probes = stabler signal
W_HOMEO = 1.0
W_METAB = 0.3
W_INSIGHT = 0.5

# Mutation ops: cycle through these. Rewire-heavy since we start with density.
OPS = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'remove']


def compute_pain(net, probe_bytes):
    """Internal fitness: homeostasis + metabolic + insight."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    frs, chs = [], []
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    for bv in probe_bytes:
        injected = net.input_projection[int(bv)]
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=net.mask, theta=net._theta_f32, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sc,
            polarity=net._polarity_f32, refractory=net.refractory,
            channel=net.channel,
        )
        frs.append(float(np.mean(np.abs(state) > 0)))
        chs.append(float(np.mean(charge)))

    avg_fr = np.mean(frs)
    avg_ch = np.mean(chs)
    homeo = -abs(avg_fr - TARGET_FR)
    # Metabolic: efficient = high firing per unit charge. Cap to avoid div-by-zero explosion.
    metab = min(avg_fr / (avg_ch + 0.01), 2.0)
    deltas = [(frs[t] - frs[t-1]) - (chs[t] - chs[t-1]) for t in range(1, len(frs))]
    insight = float(np.mean(deltas)) if deltas else 0.0
    score = W_HOMEO * homeo + W_METAB * metab + W_INSIGHT * insight
    return score, homeo, metab, insight, avg_fr, avg_ch


def main():
    ap = argparse.ArgumentParser(description="Train pain core seed")
    ap.add_argument("--hidden", type=int, default=32, help="Hidden neurons (default: 32)")
    ap.add_argument("--density", type=float, default=4, help="Initial density %% (default: 4)")
    ap.add_argument("--theta", type=int, default=2, help="Theta init (default: 2)")
    ap.add_argument("--decay", type=float, default=0.10, help="Decay init (default: 0.10)")
    ap.add_argument("--steps", type=int, default=20000, help="Total steps (default: 20000)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    ap.add_argument("--out", type=str, default="checkpoints", help="Output dir (default: checkpoints)")
    args = ap.parse_args()

    H = args.hidden
    STEPS = args.steps
    SEED = args.seed
    CKPT_DIR = Path(args.out)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Fixed probes — multiple sets to reduce noise
    probe_sets = [rng.randint(0, 256, size=N_PROBE_TOKENS) for _ in range(3)]

    def eval_multi(net):
        """Average pain over multiple probe sets for stability."""
        scores, homeos, metabs, insights, frs, chs = [], [], [], [], [], []
        for pb in probe_sets:
            s, h, m, i, fr, ch = compute_pain(net, pb)
            scores.append(s); homeos.append(h); metabs.append(m)
            insights.append(i); frs.append(fr); chs.append(ch)
        return (np.mean(scores), np.mean(homeos), np.mean(metabs),
                np.mean(insights), np.mean(frs), np.mean(chs))

    # Init or resume
    if args.resume:
        net = SelfWiringGraph.load(args.resume)
        print(f"Resumed from {args.resume} (H={net.H}, edges={len(net.alive)})")
    else:
        net = SelfWiringGraph(
            vocab=VOCAB, hidden=H, density=args.density,
            theta_init=args.theta, decay_init=args.decay, seed=SEED,
        )

    best_score, best_ho, best_me, best_in, best_fr, best_ch = eval_multi(net)
    best_ever = best_score
    accepts = 0
    stale = 0
    stale_limit = 2000    # save checkpoint if stale this long
    phase = 'EXPLORE'     # EXPLORE → REFINE when close to target

    print(f"=== Pain Core Training ===")
    print(f"H={net.H} | density={args.density}% | theta={args.theta} | decay={args.decay}")
    print(f"Steps={STEPS} | probes={N_PROBE_TOKENS}x{len(probe_sets)} | target FR={TARGET_FR}")
    print(f"Checkpoint dir: {CKPT_DIR}")
    print()
    print(f"{'step':>6} | {'phase':>7} | {'edges':>5} | {'FR':>6} | {'charge':>6} "
          f"| {'homeo':>7} | {'metab':>6} | {'insght':>6} | {'score':>7} "
          f"| {'acc':>5} | {'stale':>5} | {'acc%':>5}")
    print("-" * 105)

    t0 = time.time()
    log_every = max(1, STEPS // 40)  # ~40 log lines total

    for step in range(1, STEPS + 1):
        # Phase switching: once homeostasis is decent, shift to refine
        if phase == 'EXPLORE' and best_ho > -0.03:
            phase = 'REFINE'

        # Op selection
        if phase == 'EXPLORE':
            op = OPS[(step - 1) % len(OPS)]
        else:
            # Refine: more theta/decay tuning, less structural change
            refine_ops = ['rewire', 'theta', 'theta', 'decay', 'rewire', 'theta', 'decay', 'rewire']
            op = refine_ops[(step - 1) % len(refine_ops)]

        if op in ('rewire', 'remove', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)
        new_score, ho, me, ins, fr, ch = eval_multi(net)

        if new_score > best_score:
            best_score = new_score
            best_ho, best_me, best_in, best_fr, best_ch = ho, me, ins, fr, ch
            accepts += 1
            stale = 0
            if new_score > best_ever:
                best_ever = new_score
        else:
            net.restore_state(state_snap)
            stale += 1

        # Stale checkpoint: save and rotate probes
        if stale == stale_limit:
            path = CKPT_DIR / f"pain_core_stale_{step}.npz"
            net.save(str(path))
            print(f"  >>> Stale plateau at step {step}, saved {path.name}")
            # Rotate probe sets to escape local optima
            probe_sets.append(rng.randint(0, 256, size=N_PROBE_TOKENS))
            if len(probe_sets) > 5:
                probe_sets.pop(0)
            best_score, best_ho, best_me, best_in, best_fr, best_ch = eval_multi(net)
            stale = 0

        # Log
        if step % log_every == 0 or step == 1:
            rate = 100.0 * accepts / step
            print(f"{step:6d} | {phase:>7} | {len(net.alive):5d} | {best_fr:6.3f} | {best_ch:6.3f} "
                  f"| {best_ho:7.4f} | {best_me:6.3f} | {best_in:6.4f} | {best_score:7.4f} "
                  f"| {accepts:5d} | {stale:5d} | {rate:4.1f}%")

    elapsed = time.time() - t0

    # Save final
    final_path = CKPT_DIR / "pain_core_best.npz"
    net.save(str(final_path))

    print(f"\n{'='*105}")
    print(f"  DONE in {elapsed:.1f}s ({elapsed/STEPS*1000:.1f}ms/step)")
    print(f"  Final: H={net.H} | edges={len(net.alive)} | FR={best_fr:.4f} | score={best_score:.4f}")
    print(f"  Accepts: {accepts}/{STEPS} ({100*accepts/STEPS:.1f}%)")
    print(f"  Saved: {final_path}")
    print(f"{'='*105}")

    # Quality report
    print(f"\n--- Core Quality Report ---")
    print(f"  Homeostasis:  {best_ho:+.4f}  (0 = perfect, target FR={TARGET_FR})")
    print(f"  Firing rate:  {best_fr:.4f}  (target: {TARGET_FR})")
    print(f"  Metabolic:    {best_me:.4f}  (higher = more efficient)")
    print(f"  Insight:      {best_in:+.4f}  (positive = tension releasing)")
    print(f"  Mean charge:  {best_ch:.4f}  (lower = less wasted energy)")
    print(f"  Edge count:   {len(net.alive)}  (density: {100*len(net.alive)/(net.H*net.H):.1f}%)")

    # Maturity check
    mature = best_ho > -0.02 and best_fr > 0.05 and best_me > 0.1
    print(f"\n  Maturity: {'MATURE — ready for transplant' if mature else 'NOT YET — needs more steps or tuning'}")
    if not mature:
        if best_ho <= -0.02:
            print(f"    - Homeostasis not converged (need > -0.02, got {best_ho:.4f})")
        if best_fr <= 0.05:
            print(f"    - Firing rate too low (need > 5%, got {best_fr:.4f})")
        if best_me <= 0.1:
            print(f"    - Metabolic efficiency low (need > 0.1, got {best_me:.4f})")


if __name__ == "__main__":
    main()
