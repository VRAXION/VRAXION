"""
INSTNCT — Pain Seed Experiment
==============================
Can internal charge/firing dynamics alone (no task fitness) produce
structured graphs that are better starting points than random init?

Three internal fitness signals:
  1. Homeostasis:  -|firing_rate - target|   (alive and breathing?)
  2. Metabolic:    firing_rate / mean_charge  (efficient energy use?)
  3. Insight:      delta_firing - delta_charge (tension releasing?)

Growth schedule: 32 → 64 → 128 neurons.
No text data, no bigram — random byte probes only.
"""
import sys, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# --- Config ---
VOCAB = 256
INIT_HIDDEN = 32
THETA_INIT = 2
DECAY_INIT = 0.10
TICKS = 8
INPUT_DURATION = 2
TARGET_FIRING_RATE = 0.15
N_PROBE_TOKENS = 8
TOTAL_STEPS = 2000
GROWTH_SCHEDULE = {801: 64, 1401: 128}
# Phase-aware: heavy add early, balanced later
MUTATION_SCHEDULE_SEED = ['add', 'add', 'add', 'add', 'rewire', 'add', 'theta', 'decay']
MUTATION_SCHEDULE_GROW = ['add', 'add', 'rewire', 'theta', 'decay', 'add', 'rewire', 'decay']
PRINT_EVERY = 100
SEED = 42

# Composite weights
W_HOMEO = 1.0
W_METAB = 0.3
W_INSIGHT = 0.5


def compute_pain(net, probe_bytes):
    """Feed random bytes, measure internal dynamics. No task, no accuracy."""
    net.reset()
    H = net.H
    sparse_cache = SelfWiringGraph.build_sparse_cache(net.mask)

    firing_rates = []
    mean_charges = []
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    for byte_val in probe_bytes:
        injected = net.input_projection[int(byte_val)]
        state, charge = SelfWiringGraph.rollout_token(
            injected,
            mask=net.mask,
            theta=net._theta_f32,
            decay=net.decay,
            ticks=TICKS,
            input_duration=INPUT_DURATION,
            state=state,
            charge=charge,
            sparse_cache=sparse_cache,
            polarity=net._polarity_f32,
            refractory=net.refractory,
            channel=net.channel,
        )
        firing_rates.append(float(np.mean(np.abs(state) > 0)))
        mean_charges.append(float(np.mean(charge)))

    # 1. Homeostasis
    avg_fr = np.mean(firing_rates)
    homeo = -abs(avg_fr - TARGET_FIRING_RATE)

    # 2. Metabolic efficiency
    avg_chg = np.mean(mean_charges)
    metab = avg_fr / (avg_chg + 1e-6)

    # 3. Insight: tension release across consecutive tokens
    deltas = []
    for t in range(1, len(firing_rates)):
        d_fire = firing_rates[t] - firing_rates[t - 1]
        d_charge = mean_charges[t] - mean_charges[t - 1]
        deltas.append(d_fire - d_charge)
    insight = float(np.mean(deltas)) if deltas else 0.0

    score = W_HOMEO * homeo + W_METAB * metab + W_INSIGHT * insight
    return score, homeo, metab, insight, avg_fr, avg_chg


def grow_network(old_net, new_H, rng):
    """Transplant old graph into top-left corner of a larger empty network."""
    old_H = old_net.H
    new_net = SelfWiringGraph(
        vocab=VOCAB, hidden=new_H,
        theta_init=THETA_INIT, decay_init=DECAY_INIT,
        seed=int(rng.randint(0, 2**31)),
    )
    # Clear mask, transplant old topology
    new_net.mask[:] = False
    new_net.mask[:old_H, :old_H] = old_net.mask[:old_H, :old_H]

    # Copy per-neuron params for old neurons
    new_net.theta[:old_H] = old_net.theta
    new_net._theta_f32[:old_H] = old_net._theta_f32
    new_net.decay[:old_H] = old_net.decay
    new_net.polarity[:old_H] = old_net.polarity
    new_net._polarity_f32[:old_H] = old_net._polarity_f32
    new_net.channel[:old_H] = old_net.channel

    # Bridge edges: proportional to new neuron count
    # old→new and new→old so new neurons can both receive and send
    n_new = new_H - old_H
    n_bridges = max(4, n_new // 2)
    for _ in range(n_bridges):
        src = int(rng.randint(0, old_H))
        dst = int(rng.randint(old_H, new_H))
        new_net.mask[src, dst] = True
        # Reverse bridge: new→old (so new neurons can fire back)
        src2 = int(rng.randint(old_H, new_H))
        dst2 = int(rng.randint(0, old_H))
        new_net.mask[src2, dst2] = True

    new_net.resync_alive()
    new_net.reset()
    return new_net


def print_header():
    print(f"{'step':>5} | {'H':>4} | {'edges':>5} | {'fire_rt':>7} | {'mean_ch':>7} "
          f"| {'homeo':>7} | {'metab':>7} | {'insght':>7} | {'score':>7} "
          f"| {'acc':>4} | {'rate':>5}")
    print("-" * 95)


def print_row(step, net, score, homeo, metab, insight, avg_fr, avg_chg, accepts, total):
    rate = 100.0 * accepts / total if total else 0
    print(f"{step:5d} | {net.H:4d} | {len(net.alive):5d} | {avg_fr:7.3f} | {avg_chg:7.2f} "
          f"| {homeo:7.4f} | {metab:7.4f} | {insight:7.4f} | {score:7.4f} "
          f"| {accepts:4d} | {rate:4.1f}%")


def main():
    rng = np.random.RandomState(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Fixed probe bytes for consistent evaluation
    probe_bytes = rng.randint(0, 256, size=N_PROBE_TOKENS)

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=INIT_HIDDEN, density=4,
        theta_init=THETA_INIT, decay_init=DECAY_INIT, seed=SEED,
    )

    best_score, homeo, metab, insight, avg_fr, avg_chg = compute_pain(net, probe_bytes)
    accepts = 0

    print("=== INSTNCT Pain Seed Experiment ===")
    print(f"Growth schedule: {INIT_HIDDEN}", end="")
    for s, h in sorted(GROWTH_SCHEDULE.items()):
        print(f" → {h} (step {s})", end="")
    print(f"\nProbe tokens: {N_PROBE_TOKENS}, Ticks: {TICKS}, Target FR: {TARGET_FIRING_RATE}")
    print(f"Weights: homeo={W_HOMEO} metab={W_METAB} insight={W_INSIGHT}")
    print()
    print_header()

    t0 = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        # --- Growth check ---
        if step in GROWTH_SCHEDULE:
            new_H = GROWTH_SCHEDULE[step]
            old_H = net.H
            net = grow_network(net, new_H, rng)
            best_score, homeo, metab, insight, avg_fr, avg_chg = compute_pain(net, probe_bytes)
            print(f"\n{'='*95}")
            print(f"  GROWTH: {old_H} → {new_H} neurons | {len(net.alive)} edges | baseline score: {best_score:.4f}")
            print(f"{'='*95}")
            print_header()

        # --- Mutation ---
        sched = MUTATION_SCHEDULE_SEED if net.H <= INIT_HIDDEN else MUTATION_SCHEDULE_GROW
        op = sched[(step - 1) % len(sched)]
        if op in ('rewire', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)

        # --- Evaluate ---
        new_score, h, m, i, fr, ch = compute_pain(net, probe_bytes)

        # --- Accept / Reject ---
        if new_score > best_score:
            best_score = new_score
            homeo, metab, insight, avg_fr, avg_chg = h, m, i, fr, ch
            accepts += 1
        else:
            net.restore_state(state_snap)

        # --- Log ---
        if step % PRINT_EVERY == 0:
            print_row(step, net, best_score, homeo, metab, insight, avg_fr, avg_chg, accepts, step)

    elapsed = time.time() - t0
    print(f"\n{'='*95}")
    print(f"  DONE in {elapsed:.1f}s | {accepts}/{TOTAL_STEPS} accepted ({100*accepts/TOTAL_STEPS:.1f}%)")
    print(f"  Final: H={net.H}, edges={len(net.alive)}, score={best_score:.4f}")
    print(f"{'='*95}")

    # --- Comparison: evolved vs fresh random at same size AND same edge count ---
    evolved_edges = len(net.alive)
    # Match edge density: evolved_edges / (H*H) as density fraction
    matched_density = evolved_edges / (net.H * net.H) * 100  # as percentage

    print(f"\n--- Comparison: Pain-evolved vs Fresh Random (H={net.H}) ---\n")
    print(f"  A) Same density (4%): random gets more edges naturally")
    fresh_4 = SelfWiringGraph(
        vocab=VOCAB, hidden=net.H, density=4,
        theta_init=THETA_INIT, decay_init=DECAY_INIT,
        seed=int(rng.randint(0, 2**31)),
    )
    fs4, fh4, fm4, fi4, ffr4, fch4 = compute_pain(fresh_4, probe_bytes)

    print(f"  B) Matched edges (~{evolved_edges}): fair comparison")
    fresh_m = SelfWiringGraph(
        vocab=VOCAB, hidden=net.H, density=matched_density,
        theta_init=THETA_INIT, decay_init=DECAY_INIT,
        seed=int(rng.randint(0, 2**31)),
    )
    fsm, fhm, fmm, fim, ffrm, fchm = compute_pain(fresh_m, probe_bytes)

    hdr = f"  {'':>16} | {'score':>7} | {'homeo':>7} | {'metab':>7} | {'insght':>7} | {'fire_rt':>7} | {'mean_ch':>7} | {'edges':>5}"
    print(hdr)
    print(f"  {'Pain-evolved':>16} | {best_score:7.4f} | {homeo:7.4f} | {metab:7.4f} | {insight:7.4f} | {avg_fr:7.3f} | {avg_chg:7.2f} | {evolved_edges:5d}")
    print(f"  {'Random (4%)':>16} | {fs4:7.4f} | {fh4:7.4f} | {fm4:7.4f} | {fi4:7.4f} | {ffr4:7.3f} | {fch4:7.2f} | {len(fresh_4.alive):5d}")
    print(f"  {'Random (matched)':>16} | {fsm:7.4f} | {fhm:7.4f} | {fmm:7.4f} | {fim:7.4f} | {ffrm:7.3f} | {fchm:7.2f} | {len(fresh_m.alive):5d}")
    diff_fair = best_score - fsm
    print(f"\n  Fair delta (evolved - matched random): {diff_fair:+.4f} {'(EVOLVED WINS)' if diff_fair > 0 else '(RANDOM WINS)' if diff_fair < 0 else '(TIE)'}")


if __name__ == "__main__":
    main()
