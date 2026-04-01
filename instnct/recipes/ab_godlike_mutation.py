"""
Godlike mutation A/B test
==========================
One informed compound mutation vs burst ×7 vs single-step.

Godlike: probe network → diagnose → targeted compound mutation → eval
Burst:   eval → 7× random mutate() → eval
Single:  eval → 1× mutate() → eval

All use try→keep/revert. Same eval, same budget (2 evals per step).
Cycle task (H=64, V=10) for fast iteration + English bigram (H=256) for real task.
"""
import sys, os, numpy as np, random, time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph

# ── Configs ───────────────────────────────────────────────────────────────────
TICKS = 8; INPUT_DUR = 2; SEED = 42
REPORT = 500; THRESHOLD = 0.00001

TASKS = {
    'cycle': {'H': 64, 'V': 10, 'steps': 5000, 'density': 10, 'theta': 2},
    'english': {'H': 256, 'V': 256, 'steps': 3000, 'density': 5, 'theta': 1},
}

# ── Probe + Godlike mutation ──────────────────────────────────────────────────
def probe_network(net, task_data, task_type):
    """Run probe sequences, collect per-neuron diagnostics."""
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    rows_sp, cols_sp = sc

    fire_counts = np.zeros(H, dtype=np.float32)
    charge_acc = np.zeros(H, dtype=np.float32)
    n_ticks = 0

    # Edge usage tracking (simple stamina proxy)
    edge_usage = np.zeros(len(rows_sp), dtype=np.float32)

    net.reset()
    seq = task_data[:30]  # short probe
    for i in range(len(seq) - 1):
        inj = np.zeros(H, dtype=np.float32)
        if task_type == 'cycle':
            inj[seq[i]] = 3.0
        else:
            inj = task_data['bp'][seq[i]] @ net.input_projection
        st, ch = SelfWiringGraph.rollout_token(
            inj, mask=net.mask, theta=tf, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pl, refractory=net.refractory)
        net.state[:] = st; net.charge[:] = ch

        fired = np.abs(st) > 0
        fire_counts += fired.astype(np.float32)
        charge_acc += ch
        n_ticks += 1

        # Track which edges were used (source fired)
        if len(rows_sp):
            edge_usage += fired[rows_sp].astype(np.float32)

    fire_rate = fire_counts / max(n_ticks, 1)
    avg_charge = charge_acc / max(n_ticks, 1)

    return {
        'fire_rate': fire_rate,
        'avg_charge': avg_charge,
        'edge_usage': edge_usage,
        'rows': rows_sp,
        'cols': cols_sp,
        'n_edges': len(rows_sp),
    }

def godlike_mutate(net, probe_info):
    """Informed compound mutation based on probe diagnostics."""
    H = net.H
    fr = probe_info['fire_rate']
    ch = probe_info['avg_charge']
    eu = probe_info['edge_usage']
    rows = probe_info['rows']
    cols = probe_info['cols']

    undo_list = []

    # 1. HIGH CHARGE + LOW FIRE → needs more input, add edges targeting here
    pressure = ch * (1.0 - fr)
    hot = np.where(pressure > np.percentile(pressure, 90))[0]
    if len(hot) >= 3:
        # Add a loop through hot neurons
        loop_nodes = np.random.choice(hot, size=min(3, len(hot)), replace=False).tolist()
        for i in range(len(loop_nodes)):
            r, c = loop_nodes[i], loop_nodes[(i + 1) % len(loop_nodes)]
            if r != c and not net.mask[r, c]:
                net.mask[r, c] = True
                undo_list.append(('A', r, c))

    # 2. ALWAYS FIRING → theta too low, raise it
    always_on = np.where(fr > 0.8)[0]
    for idx in always_on[:3]:
        if net.theta[idx] < 15:
            old = int(net.theta[idx])
            net.theta[idx] = np.uint8(min(15, old + 1))
            net._theta_f32[idx] = float(net.theta[idx])
            undo_list.append(('T', idx, old))

    # 3. NEVER FIRING + HAS CHARGE → theta too high, lower it
    dead_charged = np.where((fr < 0.01) & (ch > 0.1))[0]
    for idx in dead_charged[:3]:
        if net.theta[idx] > 1:
            old = int(net.theta[idx])
            net.theta[idx] = np.uint8(max(1, old - 1))
            net._theta_f32[idx] = float(net.theta[idx])
            undo_list.append(('T', idx, old))

    # 4. UNUSED EDGES → remove (edge_usage == 0)
    if len(eu) > 0:
        unused = np.where(eu == 0)[0]
        if len(unused) > 0:
            to_remove = np.random.choice(unused, size=min(3, len(unused)), replace=False)
            for eidx in to_remove:
                r, c = int(rows[eidx]), int(cols[eidx])
                if net.mask[r, c]:
                    net.mask[r, c] = False
                    undo_list.append(('R', r, c, True))

    # 5. MOST USED EDGES → rewire to distribute load
    if len(eu) > 5:
        hottest = np.argsort(eu)[-2:]
        for eidx in hottest:
            r, c = int(rows[eidx]), int(cols[eidx])
            nc = random.randint(0, H - 1)
            if nc != r and nc != c and not net.mask[r, nc] and net.mask[r, c]:
                net.mask[r, c] = False
                net.mask[r, nc] = True
                undo_list.append(('W', r, c, nc))

    net.resync_alive()
    return undo_list

# ── Eval functions ────────────────────────────────────────────────────────────
def eval_cycle(net, V):
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    ok = 0; tot = 0; net.reset()
    for i in range(50):
        inp = i % V; target = (inp + 1) % V
        oh = np.zeros(net.H, dtype=np.float32); oh[inp] = 3.0
        st, ch = SelfWiringGraph.rollout_token(
            oh, mask=net.mask, theta=tf, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pl, refractory=net.refractory)
        net.state[:] = st; net.charge[:] = ch
        if int(np.argmax(ch[:V])) == target: ok += 1
        tot += 1
    return ok / tot

def eval_english(net, seqs, bp, bigram):
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    tot = 0.0
    for seq in seqs:
        net.reset(); ss = 0.0; n = 0
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ net.input_projection
            st, ch = SelfWiringGraph.rollout_token(
                inj, mask=net.mask, theta=tf, decay=net.decay,
                ticks=TICKS, input_duration=INPUT_DUR,
                state=net.state, charge=net.charge,
                sparse_cache=sc, polarity=pl, refractory=net.refractory)
            net.state[:] = st; net.charge[:] = ch
            out = ch @ net.output_projection
            on = out / (np.linalg.norm(out) + 1e-8)
            sims = on @ pn.T; e = np.exp(sims - sims.max()); pred = e / e.sum()
            tgt = bigram[seq[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            ss += cos; n += 1
        tot += ss / n if n else 0.0
    return tot / len(seqs)

# ── Arm runner ────────────────────────────────────────────────────────────────
def run_arm(args):
    label, method, task_name, task_cfg, seed, extra = args
    H = task_cfg['H']; V = task_cfg['V']; steps = task_cfg['steps']

    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V, hidden=H, density=task_cfg['density'],
                          theta_init=task_cfg['theta'], seed=seed)

    # English data
    bp = bigram = seqs = None
    if task_name == 'english':
        bp = extra['bp']; bigram = extra['bigram']; seqs = extra['seqs']

    def eval_fn():
        if task_name == 'cycle':
            return eval_cycle(net, V)
        else:
            return eval_english(net, seqs, bp, bigram)

    log = []; accepts = 0; t0 = time.time()

    for step in range(1, steps + 1):
        saved = net.save_state()
        old_score = eval_fn()

        if method == 'single':
            net.mutate()
        elif method == 'burst7':
            for _ in range(7): net.mutate()
        elif method == 'godlike':
            if task_name == 'cycle':
                seq_data = [i % V for i in range(30)]
                probe = probe_network(net, seq_data, 'cycle')
            else:
                probe = probe_network(net, extra, 'english')
            godlike_mutate(net, probe)

        new_score = eval_fn()
        if new_score - old_score > THRESHOLD:
            accepts += 1
        else:
            net.restore_state(saved)

        if step % REPORT == 0:
            score = eval_fn()
            edges = net.count_connections()
            tm = float(net.theta.mean())
            sps = step / (time.time() - t0)
            log.append({'step': step, 'score': round(score, 4),
                        'edges': edges, 'theta': round(tm, 2), 'accepts': accepts})
            print(f'  [{step:5d}] {label}: {score*100:.1f}% e={edges} '
                  f'θ={tm:.2f} acc={accepts} {sps:.0f}sps', flush=True)

    final = eval_fn()
    return {'label': label, 'method': method, 'task': task_name, 'seed': seed,
            'final': round(final, 4), 'edges': net.count_connections(),
            'accepts': accepts, 'log': log}

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('GODLIKE vs BURST vs SINGLE MUTATION', flush=True)

    # Cycle task first (fast)
    print(f'\n=== CYCLE TASK (H=64, V=10) ===', flush=True)
    cycle_cfg = TASKS['cycle']
    arms_cycle = []
    for seed in [42, 777]:
        arms_cycle.append((f'single_s{seed}', 'single', 'cycle', cycle_cfg, seed, {}))
        arms_cycle.append((f'burst7_s{seed}', 'burst7', 'cycle', cycle_cfg, seed, {}))
        arms_cycle.append((f'godlike_s{seed}', 'godlike', 'cycle', cycle_cfg, seed, {}))

    with Pool(3) as pool:
        r1 = pool.map(run_arm, arms_cycle[:3])
    with Pool(3) as pool:
        r2 = pool.map(run_arm, arms_cycle[3:])
    cycle_results = r1 + r2

    # Summary cycle
    print(f'\n  CYCLE RESULTS:', flush=True)
    for method in ['single', 'burst7', 'godlike']:
        scores = [r['final'] for r in cycle_results if r['method'] == method]
        print(f'    {method:10s}: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}%')

    # English task
    print(f'\n=== ENGLISH BIGRAM (H=256, V=256) ===', flush=True)
    from lib.data import load_fineweb_bytes
    all_data = load_fineweb_bytes()
    bigram_path = ROOT / 'recipes' / 'data' / 'bigram_table.npy'
    bigram = np.load(bigram_path) if bigram_path.exists() else None
    if bigram is None:
        os.makedirs(bigram_path.parent, exist_ok=True)
        counts = np.zeros((256, 256), dtype=np.float64)
        for i in range(len(all_data) - 1):
            counts[all_data[i], all_data[i + 1]] += 1
        rs = counts.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        bigram = (counts / rs).astype(np.float32)
        np.save(bigram_path, bigram)
    bp_rng = np.random.RandomState(12345)
    bp = bp_rng.randn(256, 256).astype(np.float32)
    bp /= np.linalg.norm(bp, axis=1, keepdims=True)
    eval_rng = np.random.RandomState(9999)
    eng_seqs = [all_data[o:o+100] for o in [eval_rng.randint(0, len(all_data)-100) for _ in range(6)]]
    eng_extra = {'bp': bp, 'bigram': bigram, 'seqs': eng_seqs}

    eng_cfg = TASKS['english']
    arms_eng = []
    for seed in [42, 777]:
        arms_eng.append((f'single_s{seed}', 'single', 'english', eng_cfg, seed, eng_extra))
        arms_eng.append((f'burst7_s{seed}', 'burst7', 'english', eng_cfg, seed, eng_extra))
        arms_eng.append((f'godlike_s{seed}', 'godlike', 'english', eng_cfg, seed, eng_extra))

    with Pool(3) as pool:
        r3 = pool.map(run_arm, arms_eng[:3])
    with Pool(3) as pool:
        r4 = pool.map(run_arm, arms_eng[3:])
    eng_results = r3 + r4

    print(f'\n  ENGLISH RESULTS:', flush=True)
    for method in ['single', 'burst7', 'godlike']:
        scores = [r['final'] for r in eng_results if r['method'] == method]
        print(f'    {method:10s}: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}%')

    # Final
    sep = '=' * 60
    print(f'\n{sep}')
    print('  FINAL COMPARISON')
    print(sep)
    for task_name, results in [('cycle', cycle_results), ('english', eng_results)]:
        print(f'\n  {task_name}:')
        for method in ['single', 'burst7', 'godlike']:
            scores = [r['final'] for r in results if r['method'] == method]
            accs = [r['accepts'] for r in results if r['method'] == method]
            print(f'    {method:10s}: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}%  '
                  f'avg_acc={np.mean(accs):.0f}')
    print(sep)
