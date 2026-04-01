"""
Self-scheduling mutation via dedicated neurons
================================================
Neurons 0-7 control the mutation schedule. Their charge after forward
pass determines how many of each mutation type to apply.

Default charge values = default schedule. If the network builds edges
TO/FROM these neurons, it modifies its own mutation schedule.

A) CONTROL: fixed schedule (burst 7× full_mutate)
B) SELF-SCHEDULE: neurons 0-7 control mutation counts

Cycle task (H=64, V=10) × 2 seeds. Fast iteration.
"""
import sys, os, numpy as np, random, time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph

H = 64; V = 10; TICKS = 8; INPUT_DUR = 2
STEPS = 5000; REPORT = 500; THRESHOLD = 0.00001

MUTATION_OPS = ['add', 'remove', 'rewire', 'theta', 'decay', 'polarity', 'add_loop', 'channel']
N_SCHED = len(MUTATION_OPS)  # 8 neurons for schedule

# Default schedule: what a "good" burst looks like
DEFAULT_SCHEDULE = {
    'add': 3,
    'remove': 1,
    'rewire': 2,
    'theta': 1,
    'decay': 1,
    'polarity': 0,
    'add_loop': 1,
    'channel': 0,
}
# Total = 9 mutations per step (similar to burst 7-10)

CYCLE_MAP = {i: (i + 1) % V for i in range(V)}

def eval_cycle(net):
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    ok = 0; tot = 0; net.reset()
    for i in range(50):
        inp = i % V; target = CYCLE_MAP[inp]
        oh = np.zeros(H, dtype=np.float32); oh[N_SCHED + inp] = 3.0  # offset by schedule neurons
        st, ch = SelfWiringGraph.rollout_token(
            oh, mask=net.mask, theta=tf, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pl, refractory=net.refractory)
        net.state[:] = st; net.charge[:] = ch
        # Read output from neurons after schedule neurons
        if int(np.argmax(ch[N_SCHED:N_SCHED + V])) == target: ok += 1
        tot += 1
    return ok / tot

def read_schedule(net):
    """Read mutation schedule from neurons 0-7 charge after a probe pass."""
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)

    # Inject default schedule values into neurons 0-7
    net.reset()
    inj = np.zeros(H, dtype=np.float32)
    for i, op in enumerate(MUTATION_OPS):
        inj[i] = float(DEFAULT_SCHEDULE[op])

    # Run forward — network can modify these via internal edges
    st, ch = SelfWiringGraph.rollout_token(
        inj, mask=net.mask, theta=tf, decay=net.decay,
        ticks=TICKS, input_duration=INPUT_DUR,
        state=net.state, charge=net.charge,
        sparse_cache=sc, polarity=pl, refractory=net.refractory)
    net.state[:] = st; net.charge[:] = ch

    # Read schedule from charge[0:8], clamp to int [0-15]
    schedule = {}
    for i, op in enumerate(MUTATION_OPS):
        val = int(np.clip(round(ch[i]), 0, 15))
        schedule[op] = val
    return schedule

def apply_schedule(net, schedule):
    """Apply the mutation schedule: N times each op."""
    all_undos = []
    for op, count in schedule.items():
        for _ in range(count):
            try:
                undo = net.mutate(forced_op=op)
                all_undos.append(undo)
            except:
                pass
    return all_undos

def run_arm(args):
    label, method, seed = args
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V, hidden=H, density=10, theta_init=2, seed=seed)

    log = []; accepts = 0; t0 = time.time()
    schedule_history = []

    for step in range(1, STEPS + 1):
        saved = net.save_state()
        old_score = eval_cycle(net)

        if method == 'fixed_burst':
            # Control: fixed 7× full_mutate
            for _ in range(7):
                net.mutate()
        elif method == 'self_schedule':
            # Read schedule from neurons, apply it
            schedule = read_schedule(net)
            apply_schedule(net, schedule)
            if step % REPORT == 0:
                schedule_history.append(schedule.copy())

        new_score = eval_cycle(net)
        if new_score - old_score > THRESHOLD:
            accepts += 1
        else:
            net.restore_state(saved)

        if step % REPORT == 0:
            score = eval_cycle(net)
            edges = net.count_connections()
            tm = float(net.theta.mean())
            sps = step / (time.time() - t0)

            # Check: do any edges connect TO/FROM schedule neurons?
            sched_in = int(np.sum(net.mask[:, :N_SCHED]))   # edges INTO sched neurons
            sched_out = int(np.sum(net.mask[:N_SCHED, :]))   # edges FROM sched neurons

            sched_str = ''
            if method == 'self_schedule' and schedule_history:
                last = schedule_history[-1]
                total_ops = sum(last.values())
                top3 = sorted(last.items(), key=lambda x: -x[1])[:3]
                sched_str = f' sched=[{",".join(f"{k[0]}={v}" for k,v in top3)}] total={total_ops}'
                sched_str += f' wired_in={sched_in} wired_out={sched_out}'

            log.append({'step': step, 'score': round(score, 4), 'edges': edges,
                        'theta': round(tm, 2), 'accepts': accepts})
            print(f'  [{step:5d}] {label}: {score*100:.1f}% e={edges} θ={tm:.2f} '
                  f'acc={accepts} {sps:.0f}sps{sched_str}', flush=True)

    final = eval_cycle(net)

    # Final schedule analysis for self_schedule
    final_sched = None
    if method == 'self_schedule':
        final_sched = read_schedule(net)

    return {'label': label, 'method': method, 'seed': seed,
            'final': round(final, 4), 'edges': net.count_connections(),
            'accepts': accepts, 'log': log, 'final_schedule': final_sched}

if __name__ == '__main__':
    print('SELF-SCHEDULING MUTATION NEURONS', flush=True)
    print(f'H={H}, V={V}, neurons 0-7 = mutation schedule', flush=True)
    print(f'Default schedule: {DEFAULT_SCHEDULE}', flush=True)
    print(f'Total default ops/step: {sum(DEFAULT_SCHEDULE.values())}', flush=True)
    print()

    arms = []
    for seed in [42, 777]:
        arms.append((f'fixed_burst_s{seed}', 'fixed_burst', seed))
        arms.append((f'self_sched_s{seed}', 'self_schedule', seed))

    # Run 2 at a time (2 methods per seed)
    with Pool(2) as pool:
        r1 = pool.map(run_arm, arms[:2])
    with Pool(2) as pool:
        r2 = pool.map(run_arm, arms[2:])
    results = r1 + r2

    sep = '=' * 60
    print(f'\n{sep}', flush=True)
    print('  RESULTS', flush=True)
    print(sep, flush=True)

    for r in results:
        print(f'  {r["label"]:22s}: {r["final"]*100:.1f}% e={r["edges"]} acc={r["accepts"]}')

    print(f'\n  By method:', flush=True)
    for method in ['fixed_burst', 'self_schedule']:
        scores = [r['final'] for r in results if r['method'] == method]
        print(f'    {method:15s}: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}%')

    # Show final schedules
    print(f'\n  Self-schedule final mutation counts:', flush=True)
    for r in results:
        if r['final_schedule']:
            total = sum(r['final_schedule'].values())
            print(f'    {r["label"]}: total={total} {r["final_schedule"]}')

    print(sep, flush=True)
