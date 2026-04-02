"""
INSTNCT — Expectation-Based Signal A/B
=======================================
C) Prediction error: one unified signal from expectation mismatch.
   worse than expected = pain, better = reward, match = boredom.

vs A) Pain only, B) Pain+Novelty from previous experiment.

This mirrors dopamine prediction error in biology.
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
N_PAIRS = 16
N_EVAL_SETS = 3
TARGET_FR = 0.15
EMA_ALPHA = 0.3           # exponential moving average smoothing


def compute_saturation(charge, edge_count):
    if edge_count < 1:
        return 0.0
    return min(float(np.sum(charge)) / float(edge_count), 1.0)


class ExpectationTracker:
    """Tracks running expectation of network state metrics."""

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.expected_fr = TARGET_FR     # start at target
        self.expected_sat = 0.25         # start at healthy baseline
        self.expected_charge = 0.05

    def update(self, fr, sat, charge):
        """Update expectations, return prediction errors."""
        # Prediction errors (reality - expectation)
        err_fr = fr - self.expected_fr
        err_sat = sat - self.expected_sat
        err_charge = charge - self.expected_charge

        # Update EMA
        self.expected_fr = self.expected_fr + self.alpha * err_fr
        self.expected_sat = self.expected_sat + self.alpha * err_sat
        self.expected_charge = self.expected_charge + self.alpha * err_charge

        # Combined prediction error:
        # FR above expected = good (more active), sat below expected = good (less bloated)
        # So: positive = "better than expected", negative = "worse"
        delta = err_fr - err_sat  # firing up + saturation down = reward

        return delta

    def copy(self):
        t = ExpectationTracker(self.alpha)
        t.expected_fr = self.expected_fr
        t.expected_sat = self.expected_sat
        t.expected_charge = self.expected_charge
        return t


# ---- Mode A: Pain only (saturation into neuron 0) ----

def rollout_pain_only(net, data_inj, sc, state, charge):
    H = net.H
    edge_count = len(net.alive)
    state, charge = SelfWiringGraph.rollout_token(
        data_inj, mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=2, input_duration=INPUT_DURATION,
        state=state, charge=charge, sparse_cache=sc,
        polarity=net._polarity_f32, refractory=net.refractory,
        channel=net.channel,
    )
    sat = compute_saturation(charge, edge_count)
    sig = np.zeros(H, dtype=np.float32)
    sig[0] = sat * 3.0
    state, charge = SelfWiringGraph.rollout_token(
        sig, mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=TICKS - 2, input_duration=1,
        state=state, charge=charge, sparse_cache=sc,
        polarity=net._polarity_f32, refractory=net.refractory,
        channel=net.channel,
    )
    return state, charge, sat


# ---- Mode C: Expectation-based unified signal ----

def rollout_expectation(net, data_inj, sc, state, charge, tracker):
    """Rollout with prediction-error signal.

    Neuron 0: |negative delta| (pain — worse than expected)
    Neuron 1: positive delta (reward — better than expected)
    When delta ≈ 0: small pain on neuron 0 (boredom)
    """
    H = net.H
    edge_count = len(net.alive)

    # Phase 1: data
    state, charge = SelfWiringGraph.rollout_token(
        data_inj, mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=2, input_duration=INPUT_DURATION,
        state=state, charge=charge, sparse_cache=sc,
        polarity=net._polarity_f32, refractory=net.refractory,
        channel=net.channel,
    )

    # Measure reality
    sat = compute_saturation(charge, edge_count)
    fr = float(np.mean(np.abs(state) > 0))
    ch = float(np.mean(charge))

    # Prediction error
    delta = tracker.update(fr, sat, ch)

    # Build signal
    sig = np.zeros(H, dtype=np.float32)
    if delta < -0.001:
        # Worse than expected → pain (neuron 0)
        sig[0] = min(abs(delta) * 10.0, 3.0)
    elif delta > 0.001:
        # Better than expected → reward (neuron 1)
        sig[1] = min(delta * 10.0, 3.0)
    else:
        # Boredom — matches expectation exactly → mild pain
        sig[0] = 0.5

    # Phase 2: signal + propagation
    state, charge = SelfWiringGraph.rollout_token(
        sig, mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=TICKS - 2, input_duration=1,
        state=state, charge=charge, sparse_cache=sc,
        polarity=net._polarity_f32, refractory=net.refractory,
        channel=net.channel,
    )

    return state, charge, sat, delta


def eval_mode(net, byte_pairs, mode, tracker=None):
    """Evaluate network under a specific mode."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    if tracker is not None:
        trk = tracker.copy()  # don't pollute the real tracker during eval
    else:
        trk = ExpectationTracker()

    correct = 0
    total = 0
    sats, frs, chs, deltas = [], [], [], []

    for a_byte, b_byte in byte_pairs:
        data_inj = net.input_projection[int(a_byte)]

        if mode == 'pain':
            state, charge, sat = rollout_pain_only(net, data_inj, sc, state, charge)
            delta = 0.0
        elif mode == 'expectation':
            state, charge, sat, delta = rollout_expectation(
                net, data_inj, sc, state, charge, trk
            )

        sats.append(sat)
        fr = float(np.mean(np.abs(state) > 0))
        frs.append(fr)
        chs.append(float(np.mean(charge)))
        deltas.append(delta)

        logits = charge @ net.output_projection
        pred = int(np.argmax(logits))
        if pred == int(b_byte):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    avg_fr = np.mean(frs)
    avg_ch = np.mean(chs)
    avg_sat = np.mean(sats)
    avg_delta = np.mean(deltas)

    homeo = -abs(avg_fr - TARGET_FR)
    sat_health = -abs(avg_sat - 0.25)
    metab = min(avg_fr / (avg_ch + 0.01), 2.0)

    score = (
        0.5 * acc +
        1.0 * homeo +
        0.5 * sat_health +
        0.2 * metab
    )

    return score, acc, homeo, sat_health, metab, avg_fr, avg_ch, avg_sat, avg_delta


OPS_EXPLORE = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'remove']
OPS_REFINE = ['rewire', 'theta', 'theta', 'decay', 'rewire', 'theta', 'decay', 'rewire']


def run_arm(label, mode, hidden, steps, seed, eval_sets):
    rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=hidden, density=4,
        theta_init=1, decay_init=0.10, seed=seed,
    )
    tracker = ExpectationTracker() if mode == 'expectation' else None

    def eval_multi(net):
        results = [eval_mode(net, pairs, mode, tracker) for pairs in eval_sets]
        return tuple(np.mean([r[i] for r in results]) for i in range(9))

    best = eval_multi(net)
    best_score = best[0]
    accepts = 0
    stale = 0
    phase = 'EXPLORE'
    history = []
    log_every = max(1, steps // 20)

    for step in range(1, steps + 1):
        if phase == 'EXPLORE' and best[2] > -0.03:  # homeo
            phase = 'REFINE'

        ops = OPS_EXPLORE if phase == 'EXPLORE' else OPS_REFINE
        op = ops[(step - 1) % len(ops)]
        if op in ('rewire', 'remove', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        # Save tracker state too for undo
        old_tracker = tracker.copy() if tracker else None
        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)
        res = eval_multi(net)

        if res[0] > best_score:
            best = res
            best_score = res[0]
            accepts += 1
            stale = 0
        else:
            net.restore_state(state_snap)
            if old_tracker:
                tracker = old_tracker
            stale += 1

        if step % log_every == 0:
            history.append({
                'step': step, 'acc': best[1], 'fr': best[5],
                'sat': best[7], 'homeo': best[2], 'score': best_score,
                'delta': best[8], 'edges': len(net.alive), 'accepts': accepts,
            })

    return {
        'label': label, 'net': net,
        'score': best_score, 'acc': best[1], 'fr': best[5],
        'sat': best[7], 'homeo': best[2], 'metab': best[4],
        'charge': best[6], 'delta': best[8],
        'edges': len(net.alive), 'accepts': accepts,
        'history': history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    master_rng = np.random.RandomState(99)
    eval_sets = [
        [(master_rng.randint(0, 256), master_rng.randint(0, 256)) for _ in range(N_PAIRS)]
        for _ in range(N_EVAL_SETS)
    ]

    print("=" * 110)
    print("  Expectation A/B: Pain-only vs Prediction-Error Signal")
    print("=" * 110)

    # A) Pain only
    print(f"\n>>> A) PAIN ONLY")
    t0 = time.time()
    res_a = run_arm("Pain only", 'pain', args.hidden, args.steps, args.seed, eval_sets)
    print(f"    Done in {time.time()-t0:.1f}s")
    for h in res_a['history']:
        print(f"    step {h['step']:5d} | acc={h['acc']:.3f} | FR={h['fr']:.3f} | sat={h['sat']:.3f} | score={h['score']:.4f} | acc#={h['accepts']}")

    # C) Expectation-based
    print(f"\n>>> C) EXPECTATION (prediction error: pain/reward/boredom)")
    t0 = time.time()
    res_c = run_arm("Expectation", 'expectation', args.hidden, args.steps, args.seed, eval_sets)
    print(f"    Done in {time.time()-t0:.1f}s")
    for h in res_c['history']:
        print(f"    step {h['step']:5d} | acc={h['acc']:.3f} | FR={h['fr']:.3f} | sat={h['sat']:.3f} | delta={h['delta']:+.4f} | score={h['score']:.4f} | acc#={h['accepts']}")

    # Summary
    print(f"\n{'='*110}")
    print(f"  RESULTS (H={args.hidden}, {args.steps} steps)")
    print(f"{'='*110}")
    hdr = f"  {'':>14} | {'score':>7} | {'acc':>5} | {'FR':>5} | {'sat':>5} | {'homeo':>7} | {'metab':>5} | {'chg':>5} | {'delta':>7} | {'edges':>5} | {'accs':>4}"
    print(hdr)
    for r in [res_a, res_c]:
        d = r.get('delta', 0)
        print(f"  {r['label']:>14} | {r['score']:7.4f} | {r['acc']:5.3f} | {r['fr']:5.3f} "
              f"| {r['sat']:5.3f} | {r['homeo']:7.4f} | {r['metab']:5.3f} | {r['charge']:5.3f} "
              f"| {d:+7.4f} | {r['edges']:5d} | {r['accepts']:4d}")

    # Neuron connectivity
    for r in [res_a, res_c]:
        net = r['net']
        p0 = int(np.sum(net.mask[0, :]))
        p1 = int(np.sum(net.mask[1, :]))
        print(f"\n  {r['label']}:")
        print(f"    Neuron 0 (pain/neg) outgoing: {p0}")
        print(f"    Neuron 1 (reward/pos) outgoing: {p1}")

    # Winner
    winner = "C) EXPECTATION" if res_c['acc'] > res_a['acc'] else "A) PAIN ONLY" if res_a['acc'] > res_c['acc'] else "TIE on accuracy"
    print(f"\n  Accuracy winner: {winner}")
    print(f"  Homeostasis: {'A' if res_a['homeo'] > res_c['homeo'] else 'C'} better")

    both_fr_ok = res_a['fr'] > 0.05 and res_c['fr'] > 0.05
    print(f"  Both alive: {both_fr_ok}")


if __name__ == "__main__":
    main()
