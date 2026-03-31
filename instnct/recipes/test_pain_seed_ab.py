"""
INSTNCT — Pain Seed A/B: Direct vs Grown
=========================================
A) Pain signal directly on H=128, 2000 steps (no growth)
B) Pain seed 32→64→128 growth pipeline, 2000 steps total

Same total budget, same fitness, same probe bytes. Which is better?
"""
import sys, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# --- Shared config ---
VOCAB = 256
THETA_INIT = 2
DECAY_INIT = 0.10
DENSITY = 4
TICKS = 8
INPUT_DURATION = 2
TARGET_FIRING_RATE = 0.15
N_PROBE_TOKENS = 8
TOTAL_STEPS = 2000
MUTATION_SCHEDULE = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'rewire']
PRINT_EVERY = 200
SEED = 42

W_HOMEO = 1.0
W_METAB = 0.3
W_INSIGHT = 0.5


def compute_pain(net, probe_bytes):
    """Feed random bytes, measure internal dynamics."""
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
            injected, mask=net.mask, theta=net._theta_f32, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sparse_cache,
            polarity=net._polarity_f32, refractory=net.refractory,
            channel=net.channel,
        )
        firing_rates.append(float(np.mean(np.abs(state) > 0)))
        mean_charges.append(float(np.mean(charge)))

    avg_fr = np.mean(firing_rates)
    homeo = -abs(avg_fr - TARGET_FIRING_RATE)
    avg_chg = np.mean(mean_charges)
    metab = avg_fr / (avg_chg + 1e-6)
    deltas = []
    for t in range(1, len(firing_rates)):
        deltas.append((firing_rates[t] - firing_rates[t-1]) - (mean_charges[t] - mean_charges[t-1]))
    insight = float(np.mean(deltas)) if deltas else 0.0
    score = W_HOMEO * homeo + W_METAB * metab + W_INSIGHT * insight
    return score, homeo, metab, insight, avg_fr, avg_chg


def grow_network(old_net, new_H, rng):
    old_H = old_net.H
    new_net = SelfWiringGraph(
        vocab=VOCAB, hidden=new_H, density=DENSITY,
        theta_init=THETA_INIT, decay_init=DECAY_INIT,
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


def run_pain(label, init_H, growth_schedule, probe_bytes, seed):
    """Run pain evolution. Returns (final_net, best_score, details, history)."""
    rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=init_H, density=DENSITY,
        theta_init=THETA_INIT, decay_init=DECAY_INIT, seed=seed,
    )
    best_score, homeo, metab, insight, avg_fr, avg_chg = compute_pain(net, probe_bytes)
    accepts = 0
    history = []

    for step in range(1, TOTAL_STEPS + 1):
        if step in growth_schedule:
            new_H = growth_schedule[step]
            net = grow_network(net, new_H, rng)
            best_score, homeo, metab, insight, avg_fr, avg_chg = compute_pain(net, probe_bytes)

        op = MUTATION_SCHEDULE[(step - 1) % len(MUTATION_SCHEDULE)]
        if op in ('rewire', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)
        new_score, h, m, i, fr, ch = compute_pain(net, probe_bytes)

        if new_score > best_score:
            best_score = new_score
            homeo, metab, insight, avg_fr, avg_chg = h, m, i, fr, ch
            accepts += 1
        else:
            net.restore_state(state_snap)

        if step % PRINT_EVERY == 0:
            history.append((step, net.H, len(net.alive), avg_fr, avg_chg, homeo, metab, insight, best_score, accepts))

    return net, best_score, (homeo, metab, insight, avg_fr, avg_chg, accepts), history


def main():
    # Generate probe bytes BEFORE seeding the runs (same probes for both)
    master_rng = np.random.RandomState(99)
    probe_bytes = master_rng.randint(0, 256, size=N_PROBE_TOKENS)

    print("=" * 100)
    print("  INSTNCT Pain Seed A/B: Direct H=128 vs Seed 32→64→128")
    print("=" * 100)

    # --- A) Direct: pain on H=128, 2000 steps, no growth ---
    print(f"\n>>> A) DIRECT: H=128, {TOTAL_STEPS} steps, no growth")
    t0 = time.time()
    net_a, score_a, (ho_a, me_a, in_a, fr_a, ch_a, acc_a), hist_a = run_pain(
        "Direct-128", 128, {}, probe_bytes, seed=SEED
    )
    ta = time.time() - t0
    print(f"    Done in {ta:.1f}s")

    for step, H, edges, fr, ch, ho, me, ins, sc, acc in hist_a:
        print(f"    step {step:5d} | H={H:4d} | edges={edges:5d} | FR={fr:.3f} | score={sc:.4f} | acc={acc}")

    # --- B) Grown: 32→64→128, 2000 steps total ---
    print(f"\n>>> B) GROWN: 32→64→128, {TOTAL_STEPS} steps total")
    t0 = time.time()
    net_b, score_b, (ho_b, me_b, in_b, fr_b, ch_b, acc_b), hist_b = run_pain(
        "Grown-32→128", 32, {801: 64, 1401: 128}, probe_bytes, seed=SEED
    )
    tb = time.time() - t0
    print(f"    Done in {tb:.1f}s")

    for step, H, edges, fr, ch, ho, me, ins, sc, acc in hist_b:
        print(f"    step {step:5d} | H={H:4d} | edges={edges:5d} | FR={fr:.3f} | score={sc:.4f} | acc={acc}")

    # --- C) Baseline: fresh random H=128, no evolution ---
    fresh = SelfWiringGraph(
        vocab=VOCAB, hidden=128, density=DENSITY,
        theta_init=THETA_INIT, decay_init=DECAY_INIT,
        seed=int(master_rng.randint(0, 2**31)),
    )
    score_c, ho_c, me_c, in_c, fr_c, ch_c = compute_pain(fresh, probe_bytes)

    # --- Summary ---
    print(f"\n{'='*100}")
    print(f"  RESULTS (all at H=128)")
    print(f"{'='*100}")
    hdr = f"  {'':>20} | {'score':>7} | {'homeo':>7} | {'metab':>7} | {'insght':>7} | {'FR':>7} | {'charge':>7} | {'edges':>5} | {'acc':>4}"
    print(hdr)
    print(f"  {'A) Direct 128':>20} | {score_a:7.4f} | {ho_a:7.4f} | {me_a:7.4f} | {in_a:7.4f} | {fr_a:7.3f} | {ch_a:7.2f} | {len(net_a.alive):5d} | {acc_a:4d}")
    print(f"  {'B) Grown 32→128':>20} | {score_b:7.4f} | {ho_b:7.4f} | {me_b:7.4f} | {in_b:7.4f} | {fr_b:7.3f} | {ch_b:7.2f} | {len(net_b.alive):5d} | {acc_b:4d}")
    print(f"  {'C) Fresh random':>20} | {score_c:7.4f} | {ho_c:7.4f} | {me_c:7.4f} | {in_c:7.4f} | {fr_c:7.3f} | {ch_c:7.2f} | {len(fresh.alive):5d} | {'–':>4}")

    winner = "A) DIRECT" if score_a >= score_b else "B) GROWN"
    print(f"\n  Winner: {winner} (delta: {abs(score_a - score_b):.4f})")
    print(f"  Both beat random? A: {score_a > score_c}  B: {score_b > score_c}")


if __name__ == "__main__":
    main()
