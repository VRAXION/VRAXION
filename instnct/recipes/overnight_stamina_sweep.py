"""
Overnight stamina sweep
========================
4 parallel arms testing stamina vs no-stamina on English bigram.
Full mutate() with all ops, theta starts at 1, 4% density.

Arms (run sequentially due to single-process, but each is independent):
  1. BASELINE: no stamina, full mutate
  2. STAMINA default: drain=3, regen_period=4, thresholds=(5,11)
  3. STAMINA aggressive: drain=5, regen_period=6, thresholds=(4,10)
  4. STAMINA gentle: drain=2, regen_period=3, thresholds=(6,12)
  5. STAMINA + LOOPS: inject 30 loop-3 edges + stamina default

Each arm: 5000 steps, report every 250.
Tracks: cosine, accuracy, edges, theta, 2-cycles, predictions diversity.
"""
import sys, os, numpy as np, random, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes

H = 1024; IO = 256; V = 256; SEQ_LEN = 150; TICKS = 8; INPUT_DUR = 2
SEED = 42; STEPS = 5000; N_EVAL = 8; REPORT = 250; THRESHOLD = 0.00005

def make_bp(seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, IO).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def inject_loops(mask, loop_len, budget, rng):
    added = 0; att = 0
    while added < budget and att < budget * 10:
        nodes = rng.choice(H, size=loop_len, replace=False).tolist()
        edges = []; ok = True
        for i in range(loop_len):
            r, c = nodes[i], nodes[(i + 1) % loop_len]
            if mask[r, c]: ok = False; break
            edges.append((r, c))
        if ok and added + len(edges) <= budget:
            for r, c in edges: mask[r, c] = True; added += 1
        att += 1
    return added

def eval_cos(net, seqs, bp, bigram, stamina_arr=None):
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    theta_f = net.theta.astype(np.float32)
    pol = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    tot = 0.0
    for seq in seqs:
        net.reset()
        ss = 0.0; n = 0
        stam = stamina_arr.copy() if stamina_arr is not None else None
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ net.input_projection
            st, ch = SelfWiringGraph.rollout_token(
                inj, mask=net.mask, theta=theta_f, decay=net.decay,
                ticks=TICKS, input_duration=INPUT_DUR,
                state=net.state, charge=net.charge,
                sparse_cache=sc, polarity=pol,
                refractory=net.refractory, stamina=stam)
            net.state[:] = st; net.charge[:] = ch
            out = ch @ net.output_projection
            on = out / (np.linalg.norm(out) + 1e-8)
            sims = on @ pn.T
            e = np.exp(sims - sims.max()); pred = e / e.sum()
            tgt = bigram[seq[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            ss += cos; n += 1
        tot += ss / n if n else 0.0
    return tot / len(seqs)

def eval_diversity(net, seqs, bp, stamina_arr=None):
    """Count unique predictions — measures if output is input-dependent."""
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    theta_f = net.theta.astype(np.float32)
    pol = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    preds = set()
    net.reset()
    seq = seqs[0]
    stam = stamina_arr.copy() if stamina_arr is not None else None
    for i in range(min(50, len(seq) - 1)):
        inj = bp[seq[i]] @ net.input_projection
        st, ch = SelfWiringGraph.rollout_token(
            inj, mask=net.mask, theta=theta_f, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pol,
            refractory=net.refractory, stamina=stam)
        net.state[:] = st; net.charge[:] = ch
        out = ch @ net.output_projection
        on = out / (np.linalg.norm(out) + 1e-8)
        preds.add(int(np.argmax(on @ pn.T)))
    return len(preds)

def run_arm(label, stamina_cfg, use_loops, all_data, bigram, bp, eval_seqs):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    sys.stdout.flush()

    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(IO, hidden_ratio=4, projection_scale=1.0, seed=SEED,
                          theta_init=1, density=4)

    if use_loops:
        inject_loops(net.mask, 3, 30, np.random.RandomState(SEED + 99))
        net.resync_alive()

    # Configure stamina
    if stamina_cfg is not None:
        SelfWiringGraph.STAMINA_DRAIN = stamina_cfg['drain']
        SelfWiringGraph.STAMINA_REGEN_PERIOD = stamina_cfg['regen_period']
        SelfWiringGraph.STAMINA_THRESHOLDS = stamina_cfg['thresholds']

    np_rng = np.random.RandomState(SEED)
    log = []
    t0 = time.time()

    for step in range(1, STEPS + 1):
        saved = net.save_state()

        # Create stamina for current alive edges
        n_edges = len(net.alive)
        stam = np.full(n_edges, SelfWiringGraph.DEFAULT_STAMINA, dtype=np.uint8) if stamina_cfg else None

        tr = [all_data[o:o + SEQ_LEN]
              for o in [np_rng.randint(0, len(all_data) - SEQ_LEN) for _ in range(2)]]
        old_score = eval_cos(net, tr, bp, bigram, stam)

        undo = net.mutate()

        # Rebuild stamina for possibly changed edge list
        n_edges_new = len(net.alive)
        stam_new = np.full(n_edges_new, SelfWiringGraph.DEFAULT_STAMINA, dtype=np.uint8) if stamina_cfg else None

        new_score = eval_cos(net, tr, bp, bigram, stam_new)
        if new_score - old_score > THRESHOLD:
            pass  # keep
        else:
            net.restore_state(saved)

        if step % REPORT == 0:
            n_edges = len(net.alive)
            stam_eval = np.full(n_edges, SelfWiringGraph.DEFAULT_STAMINA, dtype=np.uint8) if stamina_cfg else None
            cos = eval_cos(net, eval_seqs, bp, bigram, stam_eval)
            div = eval_diversity(net, eval_seqs, bp, stam_eval)
            edges = net.count_connections()
            theta_m = float(net.theta.mean())
            drive = int(net.mutation_drive)
            elapsed = time.time() - t0
            sps = step / elapsed

            es = set(zip(*[x.tolist() for x in np.where(net.mask)]))
            n2 = sum(1 for r, c in es if (c, r) in es) // 2

            entry = {'step': step, 'cos': round(cos, 6), 'edges': edges,
                     'theta': round(theta_m, 2), 'drive': drive, '2cyc': n2,
                     'diversity': div, 'sps': round(sps, 2)}
            log.append(entry)
            print(f'  [{step:5d}] cos={cos:.6f} e={edges} θ={theta_m:.2f} '
                  f'd={drive} 2c={n2} div={div} {sps:.1f}sps')
            sys.stdout.flush()

    # Save checkpoint
    ckpt_path = ROOT / 'recipes' / 'checkpoints' / f'overnight_{label.replace(" ", "_").replace("+","_")}.npz'
    net.save(str(ckpt_path))
    print(f'  Saved: {ckpt_path}')

    return {'label': label, 'log': log, 'final_cos': log[-1]['cos'] if log else 0}


if __name__ == '__main__':
    print('Loading data...')
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
    bp = make_bp()

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[o:o + SEQ_LEN]
                 for o in [eval_rng.randint(0, len(all_data) - SEQ_LEN) for _ in range(N_EVAL)]]
    print(f'Data: {len(all_data)//1000}K bytes')

    ARMS = [
        ('baseline',       None, False),
        ('stamina_default', {'drain': 3, 'regen_period': 4, 'thresholds': (5, 11)}, False),
        ('stamina_aggr',    {'drain': 5, 'regen_period': 6, 'thresholds': (4, 10)}, False),
        ('stamina_gentle',  {'drain': 2, 'regen_period': 3, 'thresholds': (6, 12)}, False),
        ('stamina+loops',   {'drain': 3, 'regen_period': 4, 'thresholds': (5, 11)}, True),
    ]

    all_results = []
    for label, cfg, loops in ARMS:
        r = run_arm(label, cfg, loops, all_data, bigram, bp, eval_seqs)
        all_results.append(r)

    # Summary
    print(f'\n{"="*60}')
    print(f'  OVERNIGHT SWEEP SUMMARY')
    print(f'{"="*60}')
    for r in all_results:
        log = r['log']
        if log:
            final = log[-1]
            print(f'  {r["label"]:20s}: cos={final["cos"]:.6f} e={final["edges"]} '
                  f'θ={final["theta"]:.1f} div={final["diversity"]} 2c={final["2cyc"]}')
    print(f'{"="*60}')

    # Save full results
    out_path = ROOT / 'recipes' / 'overnight_stamina_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nFull results: {out_path}')
