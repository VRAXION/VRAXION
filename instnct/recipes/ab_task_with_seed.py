"""
INSTNCT — Task Training A/B/C: Does the seed help on real tasks?
================================================================
A) Random init H=128, task training only (baseline)
B) Expectation seed H=32 → grow H=128, then task training
C) Random init H=128 with live expectation signal during task training

Task: synthetic bigram prediction — deterministic byte→byte mapping.
Same task budget for all three. Measures convergence speed.
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
TARGET_FR = 0.15
EMA_ALPHA = 0.3

# --- Synthetic bigram: each byte has a deterministic "most likely next byte" ---
# This is a learnable task: given byte A, predict byte B = BIGRAM_MAP[A]

def build_bigram_map(seed=123):
    """Create a deterministic byte→byte mapping as learning target."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=256).astype(np.uint8)

BIGRAM_MAP = build_bigram_map()

# Test sequences: random bytes, target = BIGRAM_MAP[byte]
def make_test_seqs(n_seqs, seq_len, rng):
    seqs = []
    for _ in range(n_seqs):
        src = rng.randint(0, 256, size=seq_len).astype(np.uint8)
        seqs.append(src)
    return seqs


def compute_saturation(charge, edge_count):
    if edge_count < 1:
        return 0.0
    return min(float(np.sum(charge)) / max(float(edge_count), 1.0), 1.0)


class ExpectationTracker:
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
        t = ExpectationTracker()
        t.expected_fr = self.expected_fr
        t.expected_sat = self.expected_sat
        return t


def eval_task(net, test_seqs, use_expectation=False, tracker=None):
    """Evaluate bigram prediction accuracy.

    For each byte in sequence, inject it, rollout, read output, check
    if argmax(output) == BIGRAM_MAP[byte].
    """
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    trk = tracker.copy() if tracker else ExpectationTracker()

    correct = 0
    total = 0
    frs = []

    for seq in test_seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)

        for i in range(len(seq)):
            byte_val = int(seq[i])
            target = int(BIGRAM_MAP[byte_val])
            data_inj = net.input_projection[byte_val]

            if use_expectation:
                # Phase 1: data (2 ticks)
                state, charge = SelfWiringGraph.rollout_token(
                    data_inj, mask=net.mask, theta=net._theta_f32, decay=net.decay,
                    ticks=2, input_duration=INPUT_DURATION,
                    state=state, charge=charge, sparse_cache=sc,
                    polarity=net._polarity_f32, refractory=net.refractory,
                    channel=net.channel,
                )
                # Expectation signal
                sat = compute_saturation(charge, len(net.alive))
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
            else:
                # Standard rollout (no internal signal)
                state, charge = SelfWiringGraph.rollout_token(
                    data_inj, mask=net.mask, theta=net._theta_f32, decay=net.decay,
                    ticks=TICKS, input_duration=INPUT_DURATION,
                    state=state, charge=charge, sparse_cache=sc,
                    polarity=net._polarity_f32, refractory=net.refractory,
                    channel=net.channel,
                )

            logits = charge @ net.output_projection
            pred = int(np.argmax(logits))
            if pred == target:
                correct += 1
            total += 1
            frs.append(float(np.mean(np.abs(state) > 0)))

    acc = correct / total if total else 0.0
    avg_fr = np.mean(frs)
    return acc, avg_fr


# --- Expectation pre-training (for arm B) ---

def pretrain_expectation(net, steps, rng):
    """Pre-train with expectation signal only (no task)."""
    probe_bytes = rng.randint(0, 256, size=16)
    tracker = ExpectationTracker()
    best_score = -999
    accepts = 0
    ops = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'remove']

    for step in range(1, steps + 1):
        # Quick eval: run probes, measure health
        net.reset()
        H = net.H
        sc = SelfWiringGraph.build_sparse_cache(net.mask)
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        trk = tracker.copy()
        frs = []

        for bv in probe_bytes:
            data_inj = net.input_projection[int(bv)]
            state, charge = SelfWiringGraph.rollout_token(
                data_inj, mask=net.mask, theta=net._theta_f32, decay=net.decay,
                ticks=2, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )
            sat = compute_saturation(charge, len(net.alive))
            fr = float(np.mean(np.abs(state) > 0))
            delta = trk.update(fr, sat)
            sig = np.zeros(H, dtype=np.float32)
            if delta < -0.001:
                sig[0] = min(abs(delta) * 10.0, 3.0)
            elif delta > 0.001:
                sig[1] = min(delta * 10.0, 3.0)
            else:
                sig[0] = 0.5
            state, charge = SelfWiringGraph.rollout_token(
                sig, mask=net.mask, theta=net._theta_f32, decay=net.decay,
                ticks=TICKS - 2, input_duration=1,
                state=state, charge=charge, sparse_cache=sc,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )
            frs.append(fr)

        avg_fr = np.mean(frs)
        homeo = -abs(avg_fr - TARGET_FR)
        metab = min(avg_fr / (np.mean(charge) + 0.01), 2.0)
        score = 1.0 * homeo + 0.2 * metab

        if step == 1:
            best_score = score

        op = ops[(step - 1) % len(ops)]
        if op in ('rewire', 'remove', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)

        # Re-eval after mutation
        net.reset()
        sc2 = SelfWiringGraph.build_sparse_cache(net.mask)
        st2 = np.zeros(H, dtype=np.float32)
        ch2 = np.zeros(H, dtype=np.float32)
        trk2 = tracker.copy()
        frs2 = []
        for bv in probe_bytes:
            data_inj = net.input_projection[int(bv)]
            st2, ch2 = SelfWiringGraph.rollout_token(
                data_inj, mask=net.mask, theta=net._theta_f32, decay=net.decay,
                ticks=2, input_duration=INPUT_DURATION,
                state=st2, charge=ch2, sparse_cache=sc2,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )
            sat2 = compute_saturation(ch2, len(net.alive))
            fr2 = float(np.mean(np.abs(st2) > 0))
            trk2.update(fr2, sat2)
            sig2 = np.zeros(H, dtype=np.float32)
            sig2[0] = 0.5
            st2, ch2 = SelfWiringGraph.rollout_token(
                sig2, mask=net.mask, theta=net._theta_f32, decay=net.decay,
                ticks=TICKS - 2, input_duration=1,
                state=st2, charge=ch2, sparse_cache=sc2,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )
            frs2.append(fr2)

        avg_fr2 = np.mean(frs2)
        homeo2 = -abs(avg_fr2 - TARGET_FR)
        metab2 = min(avg_fr2 / (np.mean(ch2) + 0.01), 2.0)
        new_score = 1.0 * homeo2 + 0.2 * metab2

        if new_score > best_score:
            best_score = new_score
            accepts += 1
        else:
            net.restore_state(state_snap)

    return accepts


def grow_network(old_net, new_H, rng):
    """Transplant evolved core into larger network."""
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


def task_train(net, test_seqs, task_steps, use_expectation=False, tracker=None):
    """Train on bigram task with mutation+selection."""
    ops = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'remove']
    best_acc, best_fr = eval_task(net, test_seqs, use_expectation, tracker)
    accepts = 0
    history = []
    log_every = max(1, task_steps // 20)

    for step in range(1, task_steps + 1):
        op = ops[(step - 1) % len(ops)]
        if op in ('rewire', 'remove', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        old_tracker = tracker.copy() if tracker else None
        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)

        acc, fr = eval_task(net, test_seqs, use_expectation, tracker)

        if acc > best_acc or (acc == best_acc and abs(fr - TARGET_FR) < abs(best_fr - TARGET_FR)):
            best_acc = acc
            best_fr = fr
            accepts += 1
        else:
            net.restore_state(state_snap)
            if old_tracker:
                tracker.__dict__.update(old_tracker.__dict__)

        if step % log_every == 0:
            history.append({
                'step': step, 'acc': best_acc, 'fr': best_fr,
                'edges': len(net.alive), 'accepts': accepts,
            })

    return best_acc, best_fr, accepts, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-h", type=int, default=128, help="Target hidden size")
    ap.add_argument("--seed-h", type=int, default=32, help="Seed hidden size (arm B)")
    ap.add_argument("--pretrain", type=int, default=5000, help="Pre-training steps (arm B)")
    ap.add_argument("--task-steps", type=int, default=5000, help="Task training steps (all arms)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    master_rng = np.random.RandomState(99)
    test_seqs = make_test_seqs(n_seqs=3, seq_len=32, rng=master_rng)

    print("=" * 100)
    print("  Task Training A/B/C: Does the expectation seed help?")
    print(f"  Target: H={args.target_h} | Task: synthetic bigram (256→256 map)")
    print(f"  Pre-train: {args.pretrain} steps | Task train: {args.task_steps} steps")
    print("=" * 100)

    # === A) Random init, task training only ===
    print(f"\n>>> A) RANDOM INIT H={args.target_h}, task training only")
    random.seed(args.seed); np.random.seed(args.seed)
    net_a = SelfWiringGraph(vocab=VOCAB, hidden=args.target_h, density=4,
                            theta_init=1, decay_init=0.10, seed=args.seed)
    t0 = time.time()
    acc_a0, fr_a0 = eval_task(net_a, test_seqs)
    print(f"    Start: acc={acc_a0:.4f} FR={fr_a0:.3f} edges={len(net_a.alive)}")
    acc_a, fr_a, accepts_a, hist_a = task_train(net_a, test_seqs, args.task_steps)
    ta = time.time() - t0
    print(f"    Done in {ta:.1f}s")
    for h in hist_a:
        print(f"    step {h['step']:5d} | acc={h['acc']:.4f} | FR={h['fr']:.3f} | edges={h['edges']} | acc#={h['accepts']}")

    # === B) Expectation seed → grow → task training ===
    print(f"\n>>> B) EXPECTATION SEED H={args.seed_h} → grow H={args.target_h} → task training")
    random.seed(args.seed); np.random.seed(args.seed)
    rng_b = np.random.RandomState(args.seed)
    net_b_seed = SelfWiringGraph(vocab=VOCAB, hidden=args.seed_h, density=4,
                                 theta_init=1, decay_init=0.10, seed=args.seed)
    t0 = time.time()
    print(f"    Pre-training seed (H={args.seed_h}, {args.pretrain} steps)...")
    pre_acc = pretrain_expectation(net_b_seed, args.pretrain, rng_b)
    print(f"    Pre-train done: {pre_acc} accepts, edges={len(net_b_seed.alive)}")
    net_b = grow_network(net_b_seed, args.target_h, rng_b)
    acc_b0, fr_b0 = eval_task(net_b, test_seqs)
    print(f"    After growth: acc={acc_b0:.4f} FR={fr_b0:.3f} edges={len(net_b.alive)}")
    acc_b, fr_b, accepts_b, hist_b = task_train(net_b, test_seqs, args.task_steps)
    tb = time.time() - t0
    print(f"    Done in {tb:.1f}s (incl. pre-training)")
    for h in hist_b:
        print(f"    step {h['step']:5d} | acc={h['acc']:.4f} | FR={h['fr']:.3f} | edges={h['edges']} | acc#={h['accepts']}")

    # === C) Random init + live expectation signal during task training ===
    print(f"\n>>> C) RANDOM INIT H={args.target_h} + LIVE EXPECTATION during task training")
    random.seed(args.seed); np.random.seed(args.seed)
    net_c = SelfWiringGraph(vocab=VOCAB, hidden=args.target_h, density=4,
                            theta_init=1, decay_init=0.10, seed=args.seed)
    tracker_c = ExpectationTracker()
    t0 = time.time()
    acc_c0, fr_c0 = eval_task(net_c, test_seqs, use_expectation=True, tracker=tracker_c)
    print(f"    Start: acc={acc_c0:.4f} FR={fr_c0:.3f} edges={len(net_c.alive)}")
    acc_c, fr_c, accepts_c, hist_c = task_train(
        net_c, test_seqs, args.task_steps,
        use_expectation=True, tracker=tracker_c,
    )
    tc = time.time() - t0
    print(f"    Done in {tc:.1f}s")
    for h in hist_c:
        print(f"    step {h['step']:5d} | acc={h['acc']:.4f} | FR={h['fr']:.3f} | edges={h['edges']} | acc#={h['accepts']}")

    # === Summary ===
    print(f"\n{'='*100}")
    print(f"  RESULTS — Bigram accuracy after {args.task_steps} task-training steps")
    print(f"{'='*100}")
    print(f"  {'':>30} | {'start':>6} | {'final':>6} | {'gain':>6} | {'FR':>5} | {'edges':>5} | {'accs':>4} | {'time':>5}")
    print(f"  {'A) Random only':>30} | {acc_a0:6.4f} | {acc_a:6.4f} | {acc_a-acc_a0:+6.4f} | {fr_a:5.3f} | {len(net_a.alive):5d} | {accepts_a:4d} | {ta:5.1f}s")
    print(f"  {'B) Seed→grow→task':>30} | {acc_b0:6.4f} | {acc_b:6.4f} | {acc_b-acc_b0:+6.4f} | {fr_b:5.3f} | {len(net_b.alive):5d} | {accepts_b:4d} | {tb:5.1f}s")
    print(f"  {'C) Random+live expectation':>30} | {acc_c0:6.4f} | {acc_c:6.4f} | {acc_c-acc_c0:+6.4f} | {fr_c:5.3f} | {len(net_c.alive):5d} | {accepts_c:4d} | {tc:5.1f}s")

    random_baseline = 1.0 / 256
    print(f"\n  Random baseline: {random_baseline:.4f}")
    best_label = 'A' if acc_a >= max(acc_b, acc_c) else ('B' if acc_b >= acc_c else 'C')
    print(f"  Winner: {best_label}")

    if acc_b > acc_a:
        print(f"  Seed helps: +{acc_b-acc_a:.4f} accuracy over random init")
    else:
        print(f"  Seed does NOT help: {acc_b-acc_a:+.4f} vs random")

    if acc_c > acc_a:
        print(f"  Live expectation helps: +{acc_c-acc_a:.4f} accuracy over baseline")
    else:
        print(f"  Live expectation does NOT help: {acc_c-acc_a:+.4f} vs baseline")


if __name__ == "__main__":
    main()
