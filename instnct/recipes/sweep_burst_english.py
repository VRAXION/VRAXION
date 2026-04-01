"""
Burst mutation sweep on English bigram
=======================================
Does burst mutation work on the real task?
4 burst sizes × 3 seeds = 12 arms, 3 cores parallel.
Full mutate(), H=1024, theta=1, 4% density.
2000 steps each (enough to see trajectory).
"""
import sys, os, numpy as np, random, time, json
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes

H = 1024; IO = 256; SEQ_LEN = 150; TICKS = 8; INPUT_DUR = 2
STEPS = 2000; N_EVAL = 6; REPORT = 400; THRESHOLD = 0.00005

_all_data = None; _bigram = None; _bp = None; _eval_seqs = None

def init_worker():
    global _all_data, _bigram, _bp, _eval_seqs
    _all_data = load_fineweb_bytes()
    _bigram = np.load(ROOT / 'recipes' / 'data' / 'bigram_table.npy')
    rng = np.random.RandomState(12345)
    _bp = rng.randn(256, IO).astype(np.float32)
    _bp /= np.linalg.norm(_bp, axis=1, keepdims=True)
    eval_rng = np.random.RandomState(9999)
    _eval_seqs = [_all_data[o:o+SEQ_LEN]
                  for o in [eval_rng.randint(0, len(_all_data)-SEQ_LEN) for _ in range(N_EVAL)]]

def eval_cos(net, seqs):
    pn = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    tot = 0.0
    for seq in seqs:
        net.reset()
        ss = 0.0; n = 0
        for i in range(len(seq) - 1):
            inj = _bp[seq[i]] @ net.input_projection
            st, ch = SelfWiringGraph.rollout_token(
                inj, mask=net.mask, theta=tf, decay=net.decay,
                ticks=TICKS, input_duration=INPUT_DUR,
                state=net.state, charge=net.charge,
                sparse_cache=sc, polarity=pl, refractory=net.refractory)
            net.state[:] = st; net.charge[:] = ch
            out = ch @ net.output_projection
            on = out / (np.linalg.norm(out) + 1e-8)
            sims = on @ pn.T
            e = np.exp(sims - sims.max()); pred = e / e.sum()
            tgt = _bigram[seq[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            ss += cos; n += 1
        tot += ss / n if n else 0.0
    return tot / len(seqs)

def eval_diversity(net):
    pn = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    preds = set(); net.reset(); seq = _eval_seqs[0]
    for i in range(min(50, len(seq) - 1)):
        inj = _bp[seq[i]] @ net.input_projection
        st, ch = SelfWiringGraph.rollout_token(
            inj, mask=net.mask, theta=tf, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pl, refractory=net.refractory)
        net.state[:] = st; net.charge[:] = ch
        out = ch @ net.output_projection
        on = out / (np.linalg.norm(out) + 1e-8)
        preds.add(int(np.argmax(on @ pn.T)))
    return len(preds)

def run_arm(args):
    label, burst_size, seed = args
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(IO, hidden_ratio=4, projection_scale=1.0,
                          seed=seed, theta_init=1, density=4)
    np_rng = np.random.RandomState(seed + 200)
    log = []; accepts = 0; t0 = time.time()

    for step in range(1, STEPS + 1):
        saved = net.save_state()
        tr = [_all_data[o:o+SEQ_LEN]
              for o in [np_rng.randint(0, len(_all_data)-SEQ_LEN) for _ in range(2)]]
        old_score = eval_cos(net, tr)

        # Burst: apply N mutations
        for _ in range(burst_size):
            net.mutate()

        new_score = eval_cos(net, tr)
        if new_score - old_score > THRESHOLD:
            accepts += 1
        else:
            net.restore_state(saved)

        if step % REPORT == 0:
            cos = eval_cos(net, _eval_seqs)
            div = eval_diversity(net)
            edges = net.count_connections()
            tm = float(net.theta.mean())
            dr = int(net.mutation_drive)
            sps = step / (time.time() - t0)
            entry = {'step': step, 'cos': round(cos, 6), 'edges': edges,
                     'theta': round(tm, 2), 'drive': dr, 'div': div,
                     'accepts': accepts, 'sps': round(sps, 2)}
            log.append(entry)
            print(f'  [{step:4d}] {label}: cos={cos:.6f} θ={tm:.2f} '
                  f'e={edges} d={dr} div={div} acc={accepts} {sps:.1f}sps',
                  flush=True)

    # Save checkpoint
    ckpt = ROOT / 'recipes' / 'checkpoints' / f'burst_{label}.npz'
    net.save(str(ckpt))
    return {'label': label, 'burst': burst_size, 'seed': seed,
            'log': log, 'final_cos': log[-1]['cos'] if log else 0,
            'final_div': log[-1]['div'] if log else 0}

if __name__ == '__main__':
    print('BURST MUTATION ON ENGLISH BIGRAM', flush=True)
    print(f'H={H}, steps={STEPS}, 4 burst sizes × 3 seeds', flush=True)
    print()

    # 4 burst sizes × 3 seeds = 12 arms
    # Run in 4 batches of 3 (one per core)
    configs = []
    for burst in [1, 5, 7, 10]:
        for seed in [42, 777, 123]:
            configs.append((f'b{burst}_s{seed}', burst, seed))

    all_results = []
    for batch_start in range(0, len(configs), 3):
        batch = configs[batch_start:batch_start+3]
        print(f'--- Batch: {[c[0] for c in batch]} ---', flush=True)
        with Pool(3, initializer=init_worker) as pool:
            results = pool.map(run_arm, batch)
        all_results.extend(results)

    # Summary
    sep = '=' * 60
    print(f'\n{sep}', flush=True)
    print('  ENGLISH BIGRAM BURST RESULTS', flush=True)
    print(sep, flush=True)

    for burst in [1, 5, 7, 10]:
        scores = [r['final_cos'] for r in all_results if r['burst'] == burst]
        divs = [r['final_div'] for r in all_results if r['burst'] == burst]
        print(f'  burst={burst:2d}: cos={np.mean(scores):.6f} ± {np.std(scores):.6f} '
              f'div={np.mean(divs):.1f} '
              f'[{", ".join(f"{s:.4f}" for s in scores)}]')

    print(f'\n  Best individual:')
    best = max(all_results, key=lambda r: r['final_cos'])
    print(f'    {best["label"]}: cos={best["final_cos"]:.6f} div={best["final_div"]}')
    print(sep, flush=True)

    out_path = ROOT / 'recipes' / 'burst_english_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults: {out_path}')
