"""
Self-directed mutation sweep
=============================
The network's output neurons control which mutation to apply.
3 methods tested in parallel:

A) RANDOM (control): random mutation type each step
B) ARGMAX: dedicated neurons 0-7, argmax selects mutation type
C) CHARGE_PATTERN: total charge parity/ranges select mutation type

All use try→keep/revert. Small network (H=64) for speed.
Uses simple pattern task (not English) for fast eval.
"""
import sys, os, numpy as np, random, time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph

H = 64; V = 10; TICKS = 8; INPUT_DUR = 2; SEED = 42
STEPS = 5000; REPORT = 500; THRESHOLD = 0.00001

# Simple pattern tasks for fast eval
# Task: predict next in cycle. Input X → output (X+1) % V
CYCLE_MAP = {i: (i + 1) % V for i in range(V)}

MUTATION_OPS = ['add', 'remove', 'rewire', 'theta', 'decay', 'polarity', 'add_loop', 'channel']

def make_net(seed):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V, hidden=H, density=10, theta_init=2, seed=seed)
    return net

def eval_score(net):
    """Eval on cycle prediction task."""
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    correct = 0; total = 0
    net.reset()
    # Run 50 tokens through the cycle
    for i in range(50):
        inp = i % V
        target = CYCLE_MAP[inp]
        one_hot = np.zeros(H, dtype=np.float32)
        one_hot[inp] = 3.0  # inject into first V neurons
        st, ch = SelfWiringGraph.rollout_token(
            one_hot, mask=net.mask, theta=tf, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pl, refractory=net.refractory)
        net.state[:] = st; net.charge[:] = ch
        # Read output from first V neurons
        pred = int(np.argmax(ch[:V]))
        if pred == target: correct += 1
        total += 1
    return correct / total

def select_op_random(net, rng):
    """Random mutation selection (control)."""
    return rng.choice(MUTATION_OPS)

def select_op_argmax(net, rng):
    """Network's charge on neurons 0-7 selects mutation type."""
    charges = net.charge[:len(MUTATION_OPS)]
    idx = int(np.argmax(charges))
    return MUTATION_OPS[idx]

def select_op_charge_pattern(net, rng):
    """Charge statistics select mutation type."""
    total_charge = float(net.charge.sum())
    n_active = int(np.sum(net.charge > 0))
    # Use charge patterns to index into ops
    idx = int(total_charge * 7 + n_active) % len(MUTATION_OPS)
    return MUTATION_OPS[idx]

def run_arm(args):
    label, selector_name, seed = args
    net = make_net(seed)

    if selector_name == 'random':
        selector = select_op_random
    elif selector_name == 'argmax':
        selector = select_op_argmax
    elif selector_name == 'charge_pattern':
        selector = select_op_charge_pattern

    rng = random.Random(seed + 100)
    np_rng = np.random.RandomState(seed + 100)
    log = []
    accepts = 0
    op_accepts = {op: 0 for op in MUTATION_OPS}
    op_tries = {op: 0 for op in MUTATION_OPS}
    t0 = time.time()

    for step in range(1, STEPS + 1):
        saved = net.save_state()
        old_score = eval_score(net)

        # Select mutation type
        op = selector(net, rng)
        op_tries[op] += 1

        # Apply mutation
        try:
            undo = net.mutate(forced_op=op)
        except Exception:
            net.restore_state(saved)
            continue

        new_score = eval_score(net)
        if new_score - old_score > THRESHOLD:
            accepts += 1
            op_accepts[op] += 1
        else:
            net.restore_state(saved)

        if step % REPORT == 0:
            score = eval_score(net)
            edges = net.count_connections()
            elapsed = time.time() - t0
            sps = step / elapsed
            log.append({'step': step, 'score': round(score, 4), 'edges': edges,
                        'accepts': accepts, 'sps': round(sps, 1)})
            # Op distribution
            top_ops = sorted(op_accepts.items(), key=lambda x: -x[1])[:3]
            top_str = ' '.join(f'{k}={v}' for k, v in top_ops)
            print(f'  [{step:5d}] {label}: {score*100:.1f}% e={edges} acc={accepts} '
                  f'{sps:.0f}sps | {top_str}', flush=True)

    final = eval_score(net)
    return {
        'label': label, 'selector': selector_name, 'seed': seed,
        'final_score': round(final, 4), 'edges': net.count_connections(),
        'accepts': accepts, 'op_accepts': op_accepts, 'op_tries': op_tries,
        'log': log
    }


if __name__ == '__main__':
    print('SELF-DIRECTED MUTATION SWEEP')
    print(f'H={H}, V={V}, task=cycle prediction, steps={STEPS}')
    print(f'3 methods × 2 seeds = 6 arms, parallel on 3 cores')
    print()

    arms = []
    for seed in [42, 777]:
        arms.append((f'random_s{seed}', 'random', seed))
        arms.append((f'argmax_s{seed}', 'argmax', seed))
        arms.append((f'charge_s{seed}', 'charge_pattern', seed))

    with Pool(3) as pool:
        results = pool.map(run_arm, arms)

    # Summary
    print(f'\n{"="*60}')
    print(f'  RESULTS')
    print(f'{"="*60}')
    for r in results:
        print(f'  {r["label"]:20s}: {r["final_score"]*100:.1f}% '
              f'e={r["edges"]} acc={r["accepts"]}')

    # Group by method
    print(f'\n  By method (avg of 2 seeds):')
    for method in ['random', 'argmax', 'charge_pattern']:
        scores = [r['final_score'] for r in results if r['selector'] == method]
        accs = [r['accepts'] for r in results if r['selector'] == method]
        print(f'    {method:15s}: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}% '
              f'avg_accepts={np.mean(accs):.0f}')

    # Op distribution for argmax (what did the network "choose"?)
    print(f'\n  Argmax op distribution (what the network chose):')
    for r in results:
        if r['selector'] == 'argmax':
            print(f'    {r["label"]}:')
            for op in MUTATION_OPS:
                tries = r['op_tries'].get(op, 0)
                accs = r['op_accepts'].get(op, 0)
                rate = accs / tries * 100 if tries > 0 else 0
                bar = '#' * (tries * 30 // STEPS) if tries > 0 else ''
                print(f'      {op:10s}: {tries:5d} tries, {accs:4d} acc ({rate:5.1f}%) {bar}')

    print(f'\n{"="*60}')
