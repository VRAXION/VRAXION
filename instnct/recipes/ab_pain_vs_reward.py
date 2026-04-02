"""
INSTNCT — Pain+Reward A/B: Does novelty reward help?
=====================================================
A) Pain only (neuron 0 = saturation)
B) Pain + Novelty reward (neuron 0 = saturation, neuron 1 = novelty)

Same budget, same eval, same seed. Which trains better?
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
NOVELTY_WINDOW = 64       # remember last N output patterns


def compute_saturation(charge, edge_count):
    if edge_count < 1:
        return 0.0
    return min(float(np.sum(charge)) / float(edge_count), 1.0)


def rollout_with_signals(net, data_injected, sparse_cache, state, charge,
                         recent_patterns, use_novelty=False):
    """Rollout with pain (always) and optional novelty reward.

    Tick 0-1: data input
    Tick 2: pain + reward injected based on current state
    Tick 3-7: free propagation
    """
    H = net.H
    edge_count = len(net.alive)

    # Phase 1: data injection (2 ticks)
    state, charge = SelfWiringGraph.rollout_token(
        data_injected,
        mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=2, input_duration=INPUT_DURATION,
        state=state, charge=charge, sparse_cache=sparse_cache,
        polarity=net._polarity_f32, refractory=net.refractory,
        channel=net.channel,
    )

    # Compute signals
    saturation = compute_saturation(charge, edge_count)

    # Build injection vector
    signal_vec = np.zeros(H, dtype=np.float32)
    signal_vec[0] = saturation * 3.0  # pain into neuron 0

    novelty = 0.0
    if use_novelty:
        # Hash the current firing pattern
        pat = tuple(np.where(np.abs(state) > 0)[0])
        if pat and pat not in recent_patterns:
            novelty = 1.0
            recent_patterns.add(pat)
            # Keep window bounded
            if len(recent_patterns) > NOVELTY_WINDOW:
                # Remove oldest (sets aren't ordered, so remove random)
                recent_patterns.pop()
        signal_vec[1] = novelty * 3.0  # reward into neuron 1

    # Phase 2: signal injection + propagation (6 ticks)
    state, charge = SelfWiringGraph.rollout_token(
        signal_vec,
        mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=TICKS - 2, input_duration=1,
        state=state, charge=charge, sparse_cache=sparse_cache,
        polarity=net._polarity_f32, refractory=net.refractory,
        channel=net.channel,
    )

    return state, charge, saturation, novelty


def eval_network(net, byte_pairs, use_novelty=False):
    """Eval with pain + optional novelty."""
    net.reset()
    H = net.H
    sparse_cache = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    recent_patterns = set()

    correct = 0
    total = 0
    sats, frs, chs, novs = [], [], [], []

    for a_byte, b_byte in byte_pairs:
        data_inj = net.input_projection[int(a_byte)]
        state, charge, sat, nov = rollout_with_signals(
            net, data_inj, sparse_cache, state, charge,
            recent_patterns, use_novelty=use_novelty,
        )
        sats.append(sat)
        fr = float(np.mean(np.abs(state) > 0))
        frs.append(fr)
        chs.append(float(np.mean(charge)))
        novs.append(nov)

        logits = charge @ net.output_projection
        pred = int(np.argmax(logits))
        if pred == int(b_byte):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    avg_fr = np.mean(frs)
    avg_ch = np.mean(chs)
    avg_sat = np.mean(sats)
    avg_nov = np.mean(novs)

    homeo = -abs(avg_fr - TARGET_FR)
    sat_health = -abs(avg_sat - 0.25)
    metab = min(avg_fr / (avg_ch + 0.01), 2.0)

    # Novelty bonus in score: reward diverse output patterns
    score = (
        0.5 * acc +
        1.0 * homeo +
        0.5 * sat_health +
        0.2 * metab +
        (0.3 * avg_nov if use_novelty else 0.0)
    )

    return score, acc, homeo, sat_health, metab, avg_fr, avg_ch, avg_sat, avg_nov


OPS_EXPLORE = ['rewire', 'rewire', 'add', 'rewire', 'theta', 'decay', 'rewire', 'remove']
OPS_REFINE = ['rewire', 'theta', 'theta', 'decay', 'rewire', 'theta', 'decay', 'rewire']


def run_experiment(label, use_novelty, hidden, steps, seed, eval_sets):
    """Run one arm of the A/B."""
    rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)

    net = SelfWiringGraph(
        vocab=VOCAB, hidden=hidden, density=4,
        theta_init=1, decay_init=0.10, seed=seed,
    )

    def eval_multi(net):
        results = [eval_network(net, pairs, use_novelty=use_novelty) for pairs in eval_sets]
        return tuple(np.mean([r[i] for r in results]) for i in range(9))

    best_score, best_acc, best_ho, best_sat, best_me, best_fr, best_ch, best_sv, best_nov = eval_multi(net)
    accepts = 0
    stale = 0
    phase = 'EXPLORE'
    history = []
    log_every = max(1, steps // 20)

    for step in range(1, steps + 1):
        if phase == 'EXPLORE' and best_ho > -0.03:
            phase = 'REFINE'

        ops = OPS_EXPLORE if phase == 'EXPLORE' else OPS_REFINE
        op = ops[(step - 1) % len(ops)]
        if op in ('rewire', 'remove', 'theta', 'decay') and len(net.alive) == 0:
            op = 'add'

        state_snap = net.save_state()
        undo = net.mutate(forced_op=op)
        sc, ac, ho, sh, me, fr, ch, sv, nv = eval_multi(net)

        if sc > best_score:
            best_score, best_acc, best_ho, best_sat = sc, ac, ho, sh
            best_me, best_fr, best_ch, best_sv, best_nov = me, fr, ch, sv, nv
            accepts += 1
            stale = 0
        else:
            net.restore_state(state_snap)
            stale += 1

        if step % log_every == 0:
            history.append({
                'step': step, 'acc': best_acc, 'fr': best_fr,
                'sat': best_sv, 'homeo': best_ho, 'score': best_score,
                'nov': best_nov, 'edges': len(net.alive), 'accepts': accepts,
            })

    return {
        'label': label, 'net': net,
        'score': best_score, 'acc': best_acc, 'fr': best_fr,
        'sat': best_sv, 'homeo': best_ho, 'metab': best_me,
        'charge': best_ch, 'novelty': best_nov,
        'edges': len(net.alive), 'accepts': accepts,
        'history': history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Same eval sets for both arms
    master_rng = np.random.RandomState(99)
    eval_sets = [
        [(master_rng.randint(0, 256), master_rng.randint(0, 256)) for _ in range(N_PAIRS)]
        for _ in range(N_EVAL_SETS)
    ]

    print("=" * 105)
    print("  Pain+Reward A/B: Does novelty reward help?")
    print("=" * 105)

    # A) Pain only
    print(f"\n>>> A) PAIN ONLY (neuron 0 = saturation)")
    t0 = time.time()
    res_a = run_experiment("Pain only", use_novelty=False,
                           hidden=args.hidden, steps=args.steps,
                           seed=args.seed, eval_sets=eval_sets)
    ta = time.time() - t0
    print(f"    Done in {ta:.1f}s")
    for h in res_a['history']:
        print(f"    step {h['step']:5d} | acc={h['acc']:.3f} | FR={h['fr']:.3f} | sat={h['sat']:.3f} | score={h['score']:.4f} | acc#={h['accepts']}")

    # B) Pain + Novelty
    print(f"\n>>> B) PAIN + NOVELTY (neuron 0 = saturation, neuron 1 = novelty)")
    t0 = time.time()
    res_b = run_experiment("Pain+Novelty", use_novelty=True,
                           hidden=args.hidden, steps=args.steps,
                           seed=args.seed, eval_sets=eval_sets)
    tb = time.time() - t0
    print(f"    Done in {tb:.1f}s")
    for h in res_b['history']:
        print(f"    step {h['step']:5d} | acc={h['acc']:.3f} | FR={h['fr']:.3f} | sat={h['sat']:.3f} | nov={h['nov']:.3f} | score={h['score']:.4f} | acc#={h['accepts']}")

    # Summary
    print(f"\n{'='*105}")
    print(f"  RESULTS (H={args.hidden}, {args.steps} steps)")
    print(f"{'='*105}")
    hdr = f"  {'':>16} | {'score':>7} | {'acc':>5} | {'FR':>5} | {'sat':>5} | {'homeo':>7} | {'metab':>5} | {'edges':>5} | {'accs':>4}"
    print(hdr)
    for r in [res_a, res_b]:
        print(f"  {r['label']:>16} | {r['score']:7.4f} | {r['acc']:5.3f} | {r['fr']:5.3f} "
              f"| {r['sat']:5.3f} | {r['homeo']:7.4f} | {r['metab']:5.3f} | {r['edges']:5d} | {r['accepts']:4d}")

    # Pain receptor analysis
    for r in [res_a, res_b]:
        net = r['net']
        p0_out = int(np.sum(net.mask[0, :]))
        p1_out = int(np.sum(net.mask[1, :])) if args.hidden > 1 else 0
        print(f"\n  {r['label']}:")
        print(f"    Neuron 0 (pain) outgoing: {p0_out}")
        if r['label'] == 'Pain+Novelty':
            print(f"    Neuron 1 (reward) outgoing: {p1_out}")

    # Winner
    # Compare on accuracy (task-relevant) since scores have different novelty bonus
    winner = "A) PAIN ONLY" if res_a['acc'] >= res_b['acc'] else "B) PAIN+NOVELTY"
    delta_acc = abs(res_a['acc'] - res_b['acc'])
    delta_ho = res_a['homeo'] - res_b['homeo']
    print(f"\n  Accuracy winner: {winner} (delta: {delta_acc:.3f})")
    print(f"  Homeostasis: {'A better' if delta_ho > 0 else 'B better'} (delta: {abs(delta_ho):.4f})")
    print(f"  Verdict: {'NOVELTY HELPS' if res_b['acc'] > res_a['acc'] else 'PAIN ALONE SUFFICIENT' if res_a['acc'] >= res_b['acc'] else 'TIE'}")


if __name__ == "__main__":
    main()
