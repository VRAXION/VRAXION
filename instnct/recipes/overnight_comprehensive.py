"""
Overnight comprehensive sweep
==============================
Test all promising approaches systematically on cycle task.
5 seeds per method for statistical significance.

Methods:
1. single mutate (baseline)
2. burst ×7 full_mutate
3. burst ×7 random forced_op
4. self-schedule neurons
5. godlike compound (random motif: loop/chain/bidir/hub)

All: H=64, V=10, cycle prediction, 5000 steps, 2 eval per step.
5 methods × 5 seeds = 25 arms, batched 3 at a time.
"""
import sys, os, numpy as np, random, time, json
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph

H = 64; V = 10; TICKS = 8; INPUT_DUR = 2
STEPS = 5000; REPORT = 1000; THRESHOLD = 0.00001
SEEDS = [42, 777, 123, 999, 314]
CYCLE_MAP = {i: (i + 1) % V for i in range(V)}

MUTATION_OPS = ['add', 'remove', 'rewire', 'theta', 'decay', 'polarity', 'add_loop']
N_SCHED = 8
DEFAULT_SCHED = {'add': 3, 'remove': 1, 'rewire': 2, 'theta': 1, 'decay': 1, 'polarity': 0, 'add_loop': 1, 'channel': 0}

def eval_cycle(net, offset=0):
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    ok = 0; tot = 0; net.reset()
    for i in range(50):
        inp = i % V; target = CYCLE_MAP[inp]
        oh = np.zeros(H, dtype=np.float32); oh[offset + inp] = 3.0
        st, ch = SelfWiringGraph.rollout_token(
            oh, mask=net.mask, theta=tf, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pl, refractory=net.refractory)
        net.state[:] = st; net.charge[:] = ch
        if int(np.argmax(ch[offset:offset + V])) == target: ok += 1
        tot += 1
    return ok / tot

def motif_mutation(net):
    """Random compound motif: loop, chain, bidir, hub-out, or hub-in."""
    choice = random.randint(0, 4)
    undo = []
    if choice == 0:  # LOOP-3
        nodes = random.sample(range(H), 3)
        for i in range(3):
            r, c = nodes[i], nodes[(i + 1) % 3]
            if r != c and not net.mask[r, c]:
                net.mask[r, c] = True; undo.append(('A', r, c))
    elif choice == 1:  # CHAIN-4
        nodes = random.sample(range(H), 4)
        for i in range(3):
            r, c = nodes[i], nodes[i + 1]
            if r != c and not net.mask[r, c]:
                net.mask[r, c] = True; undo.append(('A', r, c))
    elif choice == 2:  # BIDIR
        a, b = random.sample(range(H), 2)
        if not net.mask[a, b]: net.mask[a, b] = True; undo.append(('A', a, b))
        if not net.mask[b, a]: net.mask[b, a] = True; undo.append(('A', b, a))
    elif choice == 3:  # HUB-OUT
        hub = random.randint(0, H - 1)
        targets = random.sample([x for x in range(H) if x != hub], min(3, H - 1))
        for t in targets:
            if not net.mask[hub, t]: net.mask[hub, t] = True; undo.append(('A', hub, t))
    elif choice == 4:  # HUB-IN
        hub = random.randint(0, H - 1)
        sources = random.sample([x for x in range(H) if x != hub], min(3, H - 1))
        for s in sources:
            if not net.mask[s, hub]: net.mask[s, hub] = True; undo.append(('A', s, hub))
    # Also mutate 1 theta
    idx = random.randint(0, H - 1)
    old = int(net.theta[idx])
    net.theta[idx] = np.uint8(random.randint(1, 15))
    net._theta_f32[idx] = float(net.theta[idx])
    undo.append(('T', idx, old))
    net.resync_alive()
    return undo

def self_schedule_step(net):
    """Read schedule from neurons 0-7, apply mutations."""
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    net.reset()
    inj = np.zeros(H, dtype=np.float32)
    ops = list(DEFAULT_SCHED.keys())
    for i, op in enumerate(ops):
        inj[i] = float(DEFAULT_SCHED[op])
    st, ch = SelfWiringGraph.rollout_token(
        inj, mask=net.mask, theta=tf, decay=net.decay,
        ticks=TICKS, input_duration=INPUT_DUR,
        state=net.state, charge=net.charge,
        sparse_cache=sc, polarity=pl, refractory=net.refractory)
    net.state[:] = st; net.charge[:] = ch
    for i, op in enumerate(ops):
        count = int(np.clip(round(ch[i]), 0, 15))
        for _ in range(count):
            try: net.mutate(forced_op=op)
            except: pass

def run_arm(args):
    label, method, seed = args
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V, hidden=H, density=10, theta_init=2, seed=seed)
    log = []; accepts = 0; t0 = time.time()
    io_offset = N_SCHED if method == 'self_schedule' else 0

    for step in range(1, STEPS + 1):
        saved = net.save_state()
        old_score = eval_cycle(net, io_offset)

        if method == 'single':
            net.mutate()
        elif method == 'burst7_full':
            for _ in range(7): net.mutate()
        elif method == 'burst7_random':
            for _ in range(7):
                op = random.choice(MUTATION_OPS)
                try: net.mutate(forced_op=op)
                except: pass
        elif method == 'self_schedule':
            self_schedule_step(net)
        elif method == 'motif':
            motif_mutation(net)

        new_score = eval_cycle(net, io_offset)
        if new_score - old_score > THRESHOLD:
            accepts += 1
        else:
            net.restore_state(saved)

        if step % REPORT == 0:
            score = eval_cycle(net, io_offset)
            edges = net.count_connections()
            tm = float(net.theta.mean())
            sps = step / (time.time() - t0)
            log.append({'step': step, 'score': round(score, 4), 'edges': edges,
                        'theta': round(tm, 2), 'accepts': accepts})
            print(f'  [{step:5d}] {label}: {score*100:.1f}% e={edges} '
                  f'θ={tm:.2f} acc={accepts} {sps:.0f}sps', flush=True)

    final = eval_cycle(net, io_offset)
    return {'label': label, 'method': method, 'seed': seed,
            'final': round(final, 4), 'edges': net.count_connections(),
            'accepts': accepts, 'log': log}

if __name__ == '__main__':
    print('OVERNIGHT COMPREHENSIVE SWEEP', flush=True)
    print(f'H={H}, V={V}, steps={STEPS}, 5 methods × 5 seeds = 25 arms', flush=True)
    print()

    METHODS = ['single', 'burst7_full', 'burst7_random', 'self_schedule', 'motif']
    arms = [(f'{m}_s{s}', m, s) for m in METHODS for s in SEEDS]

    all_results = []
    for batch_start in range(0, len(arms), 3):
        batch = arms[batch_start:batch_start + 3]
        labels = [a[0] for a in batch]
        print(f'--- Batch {batch_start // 3 + 1}: {labels} ---', flush=True)
        with Pool(3) as pool:
            results = pool.map(run_arm, batch)
        all_results.extend(results)
        # Save intermediate
        with open(str(ROOT / 'recipes' / 'overnight_comprehensive_partial.json'), 'w') as f:
            json.dump([{'label': r['label'], 'method': r['method'], 'seed': r['seed'],
                        'final': r['final'], 'accepts': r['accepts']} for r in all_results], f)

    sep = '=' * 60
    print(f'\n{sep}', flush=True)
    print('  COMPREHENSIVE RESULTS (5 seeds each)', flush=True)
    print(sep, flush=True)

    for method in METHODS:
        scores = [r['final'] for r in all_results if r['method'] == method]
        accs = [r['accepts'] for r in all_results if r['method'] == method]
        individual = ' '.join(f'{s*100:.0f}' for s in scores)
        print(f'  {method:17s}: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}% '
              f'acc={np.mean(accs):.0f}  [{individual}]')

    # Statistical significance: is the best significantly better than single?
    single_scores = [r['final'] for r in all_results if r['method'] == 'single']
    best_method = max(METHODS, key=lambda m: np.mean([r['final'] for r in all_results if r['method'] == m]))
    best_scores = [r['final'] for r in all_results if r['method'] == best_method]
    diff = np.mean(best_scores) - np.mean(single_scores)
    pooled_std = np.sqrt((np.std(single_scores)**2 + np.std(best_scores)**2) / 2)
    t_stat = diff / (pooled_std * np.sqrt(2/5)) if pooled_std > 0 else 0
    print(f'\n  Best: {best_method} (t={t_stat:.2f} vs single, {"p<0.05" if abs(t_stat) > 2.13 else "not significant"})')
    print(sep, flush=True)

    out_path = ROOT / 'recipes' / 'overnight_comprehensive_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults: {out_path}')
