"""
INSTNCT Overnight Run — all winners baked in
==============================================
sign+mag mask (free weight), int8 injection table, int8 output_projection,
int8 retention=217, charge ReLU, theta=0, 8 ticks, dur=2,
bigram 2seq eval, 2a/1f/5d schedule, thresh=0.00005.
10000 steps — let it run overnight.
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_output_projection_f = None; _bigram = None; _inj_table = None

def init_w(b, d, sl, nt, wof, bg, it):
    global _bp, _all_data, _seq_len, _n_train, _output_projection_f, _bigram, _inj_table
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _output_projection_f, _bigram, _inj_table = wof, bg, it

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(msign, mmag, H, seqs):
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 217.0 / 256.0
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(8):
                if t < 2:
                    act = act + _inj_table[text_bytes[i]].astype(np.float32) / 128.0
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection_f
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos
            n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    msign_flat, mmag_flat, H, seed, ptype = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    msign = msign_flat.reshape(H, H); mmag = mmag_flat.reshape(H, H)
    new_s = msign.copy(); new_m = mmag.copy()

    if ptype == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mmag[r, c] > 0:
            return {'delta': -1e9, 'type': 'add'}
        new_s[r, c] = rng.random() < 0.5
        new_m[r, c] = rng.randint(1, 255)
    elif ptype == 'flip':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0: return {'delta': -1e9, 'type': 'flip'}
        idx = rng.randint(0, len(rs)-1)
        new_s[rs[idx], cs[idx]] = not msign[rs[idx], cs[idx]]
    elif ptype == 'mag_resample':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0: return {'delta': -1e9, 'type': 'mag_resample'}
        idx = rng.randint(0, len(rs)-1)
        new_m[rs[idx], cs[idx]] = rng.randint(1, 255)

    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, len(_all_data) - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old = _eval_bigram(msign, mmag, H, seqs)
    new = _eval_bigram(new_s, new_m, H, seqs)
    return {'delta': new - old, 'type': ptype,
            'new_s': new_s.flatten() if new > old else None,
            'new_m': new_m.flatten() if new > old else None}

def eval_accuracy(msign, mmag, H, output_projection_f, text_bytes, bp, inj_table):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp_vals = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(8):
            if t < 2: act = act + inj_table[text_bytes[i]].astype(np.float32) / 128.0
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge, 0.0); charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection_f
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18
    BUDGET = 10000
    THRESHOLD = 0.00005
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'add', 'add']

    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA = os.path.join(BASE_DIR, "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    # Int8 projections
    inj_table = np.clip(bp @ input_projection * 128, -128, 127).astype(np.int8)
    output_projection_int8 = np.clip(output_projection * 128, -128, 127).astype(np.int8)
    output_projection_f = output_projection_int8.astype(np.float32) / 128.0

    # Sign+mag mask, empty start
    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)

    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    LOG = os.path.join(BASE_DIR, "overnight_live.txt")

    print(f"\n{'='*60}")
    print(f"  INSTNCT OVERNIGHT RUN")
    print(f"  {H}n, {N_WORKERS}w, sign+mag, int8 inj+output_projection, ret=217")
    print(f"  schedule={SCHEDULE}, budget={BUDGET}")
    print(f"{'='*60}")
    sys.stdout.flush()

    with open(LOG, "w") as f:
        f.write(f"--- OVERNIGHT START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0}
    log_data = []
    t0 = time.time()

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, output_projection_f, bigram, inj_table))
    try:
        for step in range(1, BUDGET+1):
            ptype = SCHEDULE[(step-1) % len(SCHEDULE)]
            if ptype in ('flip', 'mag_resample') and (mmag > 0).sum() == 0:
                ptype = 'add'

            args = [(msign.flatten(), mmag.flatten(), H,
                     50000+step*50+w, ptype) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)

            best = max(results, key=lambda x: x['delta'])
            if best['delta'] > THRESHOLD and best.get('new_s') is not None:
                msign = best['new_s'].reshape(H, H)
                mmag = best['new_m'].reshape(H, H)
                accepts[best['type']] += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                ea = np.mean([eval_accuracy(msign, mmag, H, output_projection_f, s, bp, inj_table)
                              for s in eval_seqs])
                sps = step / elapsed
                vals = mmag[mmag > 0]
                quality = ea / max(edges, 1) * 100

                line = (f"[{step:5d}] eval={ea*100:.1f}% edges={edges} q={quality:.3f}%/e "
                        f"A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} "
                        f"mag=[{vals.min()},{vals.max()}] {elapsed:.0f}s ({sps:.2f} step/s)")
                print(f"  {line}")
                with open(LOG, "a") as f:
                    f.write(line + "\n")

                log_data.append({
                    'step': step, 'eval': round(ea * 100, 2), 'edges': edges,
                    'A': accepts['add'], 'F': accepts['flip'], 'M': accepts['mag_resample'],
                    'sps': round(sps, 2), 'time': int(elapsed)
                })
                sys.stdout.flush()

            if step % 1000 == 0:
                ckpt = os.path.join(CKPT_DIR, f"overnight_step{step}.npz")
                np.savez_compressed(ckpt,
                    msign=msign, mmag=mmag,
                    inj_table=inj_table, output_projection_int8=output_projection_int8)
                print(f"  SAVED: {ckpt}")
                with open(os.path.join(BASE_DIR, "overnight_log.json"), 'w') as f:
                    json.dump(log_data, f, separators=(',', ':'))
                sys.stdout.flush()

    finally:
        pool.terminate(); pool.join()
        final_ckpt = os.path.join(CKPT_DIR, "overnight_final.npz")
        np.savez_compressed(final_ckpt,
            msign=msign, mmag=mmag,
            inj_table=inj_table, output_projection_int8=output_projection_int8)
        print(f"  SAVED FINAL: {final_ckpt}")

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())
    ea = np.mean([eval_accuracy(msign, mmag, H, output_projection_f, s, bp, inj_table)
                  for s in eval_seqs])
    print(f"\nFINAL: eval={ea*100:.1f}% edges={edges} "
          f"A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']} "
          f"{elapsed:.0f}s ({BUDGET/elapsed:.2f} step/s)")
