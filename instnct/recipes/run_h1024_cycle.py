"""
H=1024 BUILD+CRYSTAL cycle — scaling test
==========================================
Does the BUILD+CRYSTAL ratchet work at 4x scale?
H=1024, 1% init density, ticks=16 (more ticks for bigger loops), 12 workers.
"""
import sys, os, time, random
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from quaternary_mask import QuaternaryMask

H = 1024; N_WORKERS = 12; TICKS = 16; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 25
INIT_DENSITY = 0.01; BUILD_STEPS = 2000
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SEQ_LEN = 100; N_TRAIN_SEQS = 2; N_EVAL_SEQS = 5

SCHEDULE_BUILD = [
    'add', 'enhance', 'reverse', 'mirror',
    'loop3', 'loop5', 'loop8',
    'flip', 'theta', 'channel', 'channel', 'remove',
]

BP_IN = None
_bp_out = None; _all_data = None; _bigram = None; _pol = None

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n): t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            p[byte_idx, d] = np.sin(2 * np.pi * t * (d+1) / dim * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

def init_w(bpi, bpo, data, bg, pol):
    global BP_IN, _bp_out, _all_data, _bigram, _pol
    BP_IN = bpi; _bp_out = bpo; _all_data = data; _bigram = bg; _pol = pol

def _eval_bigram(qdata, theta, channel, pol_f, seqs):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    _dummy = np.zeros((H, H), dtype=bool)
    total = 0.0
    for tb in seqs:
        state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
        s = 0.0; n = 0
        for i in range(len(tb) - 1):
            inj = np.zeros(H, np.float32); inj[0:IN_DIM] = BP_IN[tb[i]]
            state, charge = SelfWiringGraph.rollout_token(
                inj, mask=_dummy, theta=theta,
                decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=pol_f, channel=channel)
            logits = np.dot(_bp_out, charge[H - OUT_DIM:])
            e = np.exp(logits - logits.max()); pred = e / e.sum()
            tgt = _bigram[tb[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            s += cos; n += 1
        total += s / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    qdata_bytes, theta, channel, pol_f, seed, pt = args
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    qm = QuaternaryMask(H, np.frombuffer(qdata_bytes, dtype=np.uint8).copy())
    nq = qm.copy(); nt = theta.copy(); nc = channel.copy(); npf = pol_f.copy()

    if pt == 'add':
        undo = []; nq.mutate_add(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt}
    elif pt == 'enhance':
        bm = qm.to_bool_mask()
        in_deg = bm.sum(axis=0).astype(np.float64) + 1.0
        top = np.argsort(in_deg)[::-1][:H // 4]
        c = int(top[rng.randint(0, len(top) - 1)]); r = rng.randint(0, H - 1)
        if r == c or nq.get_pair(r, c) != 0: return {'delta': -1e9, 'type': pt}
        nq.set_pair(r, c, 1)
    elif pt == 'reverse':
        undo = []; nq.mutate_flip(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt}
    elif pt == 'mirror':
        undo = []; nq.mutate_upgrade(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt}
        if undo:
            idx = undo[-1][1]
            ii, jj = nq._triu_i, nq._triu_j
            i_n, j_n = int(ii[idx]), int(jj[idx])
            if (pol_f[i_n] > 0) == (pol_f[j_n] > 0):
                npf = pol_f.copy(); npf[j_n] *= -1
    elif pt in ('loop3', 'loop5', 'loop8'):
        loop_len = int(pt[4:])
        nodes = [rng.randint(0, H - 1)]
        for _ in range(loop_len - 1):
            n = rng.randint(0, H - 1)
            if n in nodes: return {'delta': -1e9, 'type': pt}
            nodes.append(n)
        for k in range(loop_len):
            r, c = nodes[k], nodes[(k + 1) % loop_len]
            if nq.get_pair(r, c) != 0: return {'delta': -1e9, 'type': pt}
        for k in range(loop_len):
            r, c = nodes[k], nodes[(k + 1) % loop_len]
            nq.set_pair(r, c, 1)
    elif pt == 'remove':
        undo = []; nq.mutate_remove(rng, undo)
        if not undo: return {'delta': -1e9, 'type': pt}
    elif pt == 'flip':
        idx = rng.randint(0, H - 1)
        npf = pol_f.copy(); npf[idx] *= -1
    elif pt == 'theta':
        idx = rng.randint(0, H - 1); nt = theta.copy()
        nt[idx] = float(rng.randint(1, 15))
    elif pt == 'channel':
        idx = rng.randint(0, H - 1); nc = channel.copy()
        nc[idx] = np.uint8(rng.randint(1, 8))

    seqs = []
    for _ in range(N_TRAIN_SEQS):
        off = nrng.randint(0, len(_all_data) - SEQ_LEN)
        seqs.append(_all_data[off:off + SEQ_LEN])

    old = _eval_bigram(qm.data, theta, channel, pol_f, seqs)
    new = _eval_bigram(nq.data, nt, nc, npf, seqs)
    return {
        'delta': float(new - old), 'type': pt,
        'new_qdata': nq.data.tobytes() if new > old else None,
        'new_theta': nt if new > old else None,
        'new_channel': nc if new > old else None,
        'new_pol': npf if new > old else None,
    }

def eval_accuracy(qdata, theta, channel, pol_f, text_bytes, bp_in, bp_out):
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    _dummy = np.zeros((H, H), dtype=bool)
    state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
    cor = 0; tot = 0
    for i in range(len(text_bytes) - 1):
        inj = np.zeros(H, np.float32); inj[0:IN_DIM] = bp_in[text_bytes[i]]
        state, charge = SelfWiringGraph.rollout_token(
            inj, mask=_dummy, theta=theta,
            decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sc,
            polarity=pol_f, channel=channel)
        logits = np.dot(bp_out, charge[H - OUT_DIM:])
        if np.argmax(logits) == text_bytes[i + 1]: cor += 1
        tot += 1
    return cor / tot if tot else 0


if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_in = build_sdr(256, IN_DIM, SDR_K, 42)
    BP_IN = bp_in
    bp_out = build_freq_order(OUT_DIM, bigram)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN]
                 for off in [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN)
                             for _ in range(N_EVAL_SEQS)]]

    # Fresh start
    init_rng = np.random.RandomState(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    init_mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(init_mask, False)
    qm = QuaternaryMask.from_bool_mask(init_mask)
    qdata = qm.data.copy()
    theta = np.full(H, 1.0, np.float32)
    channel = init_rng.randint(1, 9, size=H).astype(np.uint8)
    pol_f = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)

    bidir0, tri0 = qm.loop_levels()
    print(f"H={H}, ticks={TICKS}, density={INIT_DENSITY*100:.1f}%")
    print(f"Init: {qm.count_edges()} edges, {qm.count_bidir()} bidir, {tri0.sum()} tri")
    print(f"Workers={N_WORKERS}, Steps={BUILD_STEPS}")
    print(f"Qmask memory: {qm.memory_bytes/1024:.0f} KB")
    print()

    # === BUILD ===
    print(f"=== BUILD ({BUILD_STEPS} steps) ===")
    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, ALL_DATA, bigram, pol_f))
    acc_by_type = defaultdict(int)
    t0 = time.time(); best_eval = 0

    try:
        for step in range(1, BUILD_STEPS + 1):
            pt = SCHEDULE_BUILD[(step - 1) % len(SCHEDULE_BUILD)]
            edges = QuaternaryMask(H, qdata).count_edges()
            if pt in ('remove', 'reverse', 'mirror') and edges < 50: pt = 'add'

            args = [(qdata.tobytes(), theta.copy(), channel.copy(), pol_f.copy(),
                     5000 + step * 50 + w, pt) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)
            best_r = max(results, key=lambda x: x['delta'])

            if best_r['delta'] > THRESHOLD:
                if best_r['new_qdata'] is not None:
                    qdata[:] = np.frombuffer(best_r['new_qdata'], dtype=np.uint8)
                if best_r['new_theta'] is not None: theta[:] = best_r['new_theta']
                if best_r['new_channel'] is not None: channel[:] = best_r['new_channel']
                if best_r['new_pol'] is not None: pol_f[:] = best_r['new_pol']
                acc_by_type[best_r['type']] += 1

            if step % EVAL_EVERY == 0:
                ea = np.mean([eval_accuracy(qdata, theta, channel, pol_f, s, bp_in, bp_out)
                              for s in eval_seqs])
                if ea > best_eval: best_eval = ea
                qm_now = QuaternaryMask(H, qdata)
                bidir_s, tri_s = qm_now.loop_levels()
                elapsed = time.time() - t0
                at = ' '.join(f"{k}={v}" for k, v in sorted(acc_by_type.items()))
                print(f"  [{step:5d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={qm_now.count_edges()} bidir={qm_now.count_bidir()} "
                      f"tri={tri_s.sum()} [{at}] {elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    build_time = time.time() - t0
    qm_pre = QuaternaryMask(H, qdata)
    eval_pre = np.mean([eval_accuracy(qdata, theta, channel, pol_f, s, bp_in, bp_out)
                        for s in eval_seqs])
    print(f"\n  BUILD DONE: edges={qm_pre.count_edges()}, eval={eval_pre*100:.1f}%, "
          f"best={best_eval*100:.1f}%, {build_time:.0f}s")
    print(f"  accepts: {dict(acc_by_type)}")

    # Save checkpoint
    ckpt_path = os.path.join(BASE_DIR, "data", "h1024_build_checkpoint.npz")
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    np.savez_compressed(ckpt_path, qdata=qdata, theta=theta, channel=channel, pol_f=pol_f)
    print(f"  Checkpoint: {ckpt_path}")

    # === CRYSTALLIZE ===
    print(f"\n=== CRYSTALLIZE ===")
    net = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    net.qmask = QuaternaryMask(H, qdata.copy())
    net.theta = theta.copy(); net._theta_f32 = theta.astype(np.float32)
    net.channel = channel.copy()
    net.polarity = (pol_f > 0); net._polarity_f32 = pol_f.copy()
    net.resync_alive()

    def crystal_eval():
        net._sync_sparse_idx()
        sc = net._sp_cache
        _dummy = np.zeros((H, H), dtype=bool)
        total = 0.0
        for tb in eval_seqs:
            state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
            s = 0.0; n = 0
            for i in range(len(tb) - 1):
                inj = np.zeros(H, np.float32); inj[0:IN_DIM] = bp_in[tb[i]]
                state, charge = SelfWiringGraph.rollout_token(
                    inj, mask=_dummy, theta=net._theta_f32,
                    decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
                    state=state, charge=charge, sparse_cache=sc,
                    polarity=net._polarity_f32, channel=net.channel)
                logits = np.dot(bp_out, charge[H - OUT_DIM:])
                if np.argmax(logits) == tb[i + 1]: s += 1
                n += 1
            total += s / n if n else 0
        return total / len(eval_seqs)

    t1 = time.time()
    removed = net.crystallize(crystal_eval, eps=1e-6, verbose=True)
    crystal_time = time.time() - t1

    qm_post = net.qmask
    bidir_post, tri_post = qm_post.loop_levels()
    eval_post = np.mean([eval_accuracy(qm_post.data, theta, channel, pol_f, s, bp_in, bp_out)
                         for s in eval_seqs])

    print(f"\n  CRYSTAL DONE:")
    print(f"    Removed: {removed} ({removed/qm_pre.count_edges()*100:.1f}%)")
    print(f"    Edges:  {qm_pre.count_edges()} -> {qm_post.count_edges()}")
    print(f"    Bidir:  {qm_pre.count_bidir()} -> {qm_post.count_bidir()}")
    print(f"    Tri:    -> {tri_post.sum()}")
    print(f"    Eval:   {eval_pre*100:.1f}% -> {eval_post*100:.1f}%")

    final_path = os.path.join(BASE_DIR, "data", "h1024_crystal_checkpoint.npz")
    np.savez_compressed(final_path, qdata=qm_post.data, theta=theta, channel=channel, pol_f=pol_f)

    print(f"\n{'='*60}")
    print(f"  H=1024 SCALING TEST RESULT")
    print(f"  BUILD:   {eval_pre*100:.1f}%  ({qm_pre.count_edges()} edges)")
    print(f"  CRYSTAL: {eval_post*100:.1f}%  ({qm_post.count_edges()} edges)")
    print(f"  Time:    build={build_time:.0f}s + crystal={crystal_time:.0f}s")
    print(f"  Compare: H=256 cycle 1 = 18.2%")
    print(f"{'='*60}")
