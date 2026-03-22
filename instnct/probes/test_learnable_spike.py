"""
INSTNCT Learnable Spike Threshold
===================================
Global spike threshold starts at 5.0, mutated during training.
5000 steps to see where it converges.
LL eval, w=2 superposition, threshold=0.00005.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 120; _n_train = 2
_output_projection_f = None; _inj_table = None; _spike_thr = 5.0

def init_w(b, d, sl, nt, wof, it, sthr):
    global _bp, _all_data, _seq_len, _n_train, _output_projection_f, _inj_table, _spike_thr
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _output_projection_f, _inj_table, _spike_thr = wof, it, sthr

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_loglik(msign, mmag, H, seqs):
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
            inj = _inj_table[text_bytes[i]].astype(np.float32) / 128.0
            if i > 0:
                inj = inj + _inj_table[text_bytes[i-1]].astype(np.float32) / 128.0 * 0.5
            for t in range(8):
                if t < 2:
                    act = act + inj
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                fired = charge > _spike_thr
                charge[fired] = 0.0
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection_f
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            probs = e / e.sum()
            seq_score += np.log(probs[text_bytes[i+1]] + 1e-10)
            n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    msign_flat, mmag_flat, H, seed, ptype, proposed_spike = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    msign = msign_flat.reshape(H, H); mmag = mmag_flat.reshape(H, H)
    new_s = msign.copy(); new_m = mmag.copy()

    # Save/restore spike threshold for this worker
    global _spike_thr
    old_spike = _spike_thr
    if ptype == 'spike':
        _spike_thr = proposed_spike

    if ptype == 'add':
        alive_rs, alive_cs = np.where(mmag > 0)
        if len(alive_rs) > 0 and rng.random() < 0.5:
            if rng.random() < 0.5:
                r = alive_rs[rng.randint(0, len(alive_rs)-1)]
                c = rng.randint(0, H-1)
            else:
                r = rng.randint(0, H-1)
                c = alive_cs[rng.randint(0, len(alive_cs)-1)]
        else:
            r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mmag[r, c] > 0:
            _spike_thr = old_spike
            return {'delta': -1e9, 'type': 'add', 'spike': old_spike}
        new_s[r, c] = rng.random() < 0.5
        new_m[r, c] = rng.randint(1, 255)
    elif ptype == 'flip':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0:
            _spike_thr = old_spike
            return {'delta': -1e9, 'type': 'flip', 'spike': old_spike}
        idx = rng.randint(0, len(rs)-1)
        new_s[rs[idx], cs[idx]] = not msign[rs[idx], cs[idx]]
    elif ptype == 'mag_resample':
        rs, cs = np.where(mmag > 0)
        if len(rs) == 0:
            _spike_thr = old_spike
            return {'delta': -1e9, 'type': 'mag_resample', 'spike': old_spike}
        idx = rng.randint(0, len(rs)-1)
        new_m[rs[idx], cs[idx]] = rng.randint(1, 255)

    seqs = []
    for _ in range(_n_train):
        off = np_rng.randint(0, len(_all_data) - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old = _eval_loglik(msign, mmag, H, seqs)
    new = _eval_loglik(new_s, new_m, H, seqs)
    result_spike = _spike_thr
    _spike_thr = old_spike  # restore
    improved = new > old
    return {'delta': new - old, 'type': ptype,
            'new_s': new_s.flatten() if improved else None,
            'new_m': new_m.flatten() if improved else None,
            'spike': result_spike if improved else old_spike}


FACTS = [
    ("the sky is", "blue"), ("the sun is", "yellow and bright"),
    ("grass color is", "green"), ("fire is very", "hot and red"),
    ("snow is always", "white and cold"), ("coal color is", "black"),
    ("gold looks very", "shiny"), ("ice feels very", "cold"),
    ("milk color is", "white"), ("a cat says", "meow"),
    ("a dog says", "bark bark"), ("a cow says", "moo"),
    ("a bee says", "buzz"), ("the sea is", "salt water"),
    ("rain makes things", "wet"), ("iron feels very", "hard"),
    ("silk feels very", "soft"), ("a red flower is", "a rose"),
    ("leaves are always", "green"), ("the moon is", "round and bright"),
    ("stars look very", "bright at night"), ("a fish can", "swim fast"),
    ("a bird can", "fly high"), ("a frog can", "jump far"),
    ("a bear is", "big and strong"), ("an ant is", "tiny and small"),
    ("paris is in", "france"), ("tokyo is in", "japan"),
    ("london is in", "england"), ("rome is in", "italy"),
]

def eval_facts(msign, mmag, H, wof, bp, it, sthr):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mmag > 0)
    if len(rs) == 0: return 0.0
    s = msign[rs, cs].astype(np.float32) * 2 - 1
    sp = s * mmag[rs, cs].astype(np.float32) / 128.0
    ret = 217.0 / 256.0; tc = 0; ta = 0
    for key, val in FACTS:
        text = (key + '=' + val + '\n').encode('ascii')
        tb = np.frombuffer(text, dtype=np.uint8)
        st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
        eq = len(key)
        for i in range(len(tb)-1):
            act = st.copy()
            inj = it[tb[i]].astype(np.float32) / 128.0
            if i > 0: inj = inj + it[tb[i-1]].astype(np.float32) / 128.0 * 0.5
            for t in range(8):
                if t < 2: act = act + inj
                raw = np.zeros(H, dtype=np.float32)
                if len(rs): np.add.at(raw, cs, act[rs] * sp)
                ch += raw; ch *= ret
                fired = ch > sthr; ch[fired] = 0.0
                act = np.maximum(ch, 0.0); ch = np.maximum(ch, 0.0)
            st = act.copy()
            if i >= eq:
                out = ch @ wof
                out_n = out / (np.linalg.norm(out) + 1e-8)
                sims = out_n @ pat_norm.T
                pb = int(np.argmax(sims))
                actual = int(tb[i+1])
                if actual != ord('\n'):
                    ta += 1
                    if pb == actual: tc += 1
    return tc/ta if ta else 0


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    N_WORKERS = 18; BUDGET = 5000
    # spike mutation in schedule
    SCHEDULE = ['add', 'add', 'flip', 'mag_resample', 'spike', 'add']

    lines = [f"{k}={v}\n" for k, v in FACTS]
    random.seed(42)
    corpus_text = ''
    while len(corpus_text) < 80_000:
        random.shuffle(lines)
        corpus_text += ''.join(lines)
    ALL_DATA = np.frombuffer(corpus_text.encode('ascii'), dtype=np.uint8).copy()

    SelfWiringGraph.NV_RATIO = 4; bp = make_bp(IO)
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    inp = ref.input_projection / ref.INJ_SCALE * 1.0
    outp = ref.output_projection / ref.INJ_SCALE * 1.0
    inj_table = np.clip(bp @ inp * 128, -128, 127).astype(np.int8)
    woi = np.clip(outp * 128, -128, 127).astype(np.int8)
    wof = woi.astype(np.float32) / 128.0

    SPIKE_THR = 0.5  # LOW start — forces network to learn with spiking
    spike_history = [SPIKE_THR]

    print(f"LEARNABLE SPIKE THRESHOLD (5000 steps)")
    print(f"Start spike_thr={SPIKE_THR}, LL eval, w=2, edge threshold=0.00005")
    print(f"Schedule: {SCHEDULE}")
    print(f"{'='*60}")
    sys.stdout.flush()

    SEQ_LEN = 120
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(20)]]

    msign = np.zeros((H, H), dtype=np.bool_)
    mmag = np.zeros((H, H), dtype=np.uint8)
    accepts = {'add': 0, 'flip': 0, 'mag_resample': 0, 'spike': 0}
    t0 = time.time()

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp, ALL_DATA, SEQ_LEN, 2, wof, inj_table, SPIKE_THR))
    try:
        for step in range(1, BUDGET+1):
            ptype = SCHEDULE[(step-1) % len(SCHEDULE)]
            if ptype in ('flip', 'mag_resample') and (mmag > 0).sum() == 0:
                ptype = 'add'

            # For spike mutation: each worker tries a different threshold
            if ptype == 'spike':
                proposed_spikes = [max(0.5, SPIKE_THR + random.uniform(-2.0, 2.0))
                                   for _ in range(N_WORKERS)]
                args = [(msign.flatten(), mmag.flatten(), H,
                         10000+step*50+w, 'spike', proposed_spikes[w])
                        for w in range(N_WORKERS)]
            else:
                args = [(msign.flatten(), mmag.flatten(), H,
                         10000+step*50+w, ptype, SPIKE_THR)
                        for w in range(N_WORKERS)]

            results = pool.map(worker_eval, args)
            best = max(results, key=lambda x: x['delta'])

            if best['delta'] > 0.00005:
                if best.get('new_s') is not None:
                    msign = best['new_s'].reshape(H, H)
                    mmag = best['new_m'].reshape(H, H)
                if best['type'] == 'spike':
                    SPIKE_THR = best['spike']
                    # Update workers with new threshold
                    pool.terminate(); pool.join()
                    pool = Pool(N_WORKERS, initializer=init_w,
                                initargs=(bp, ALL_DATA, SEQ_LEN, 2, wof, inj_table, SPIKE_THR))
                accepts[best['type']] += 1

            spike_history.append(SPIKE_THR)

            if step % 500 == 0:
                elapsed = time.time() - t0
                edges = int((mmag > 0).sum())
                overall = eval_facts(msign, mmag, H, wof, bp, inj_table, SPIKE_THR)
                recent_spikes = spike_history[-100:]
                print(f"  [{step:4d}] answer={overall*100:.1f}% edges={edges} spike_thr={SPIKE_THR:.2f} "
                      f"A={accepts['add']}|F={accepts['flip']}|M={accepts['mag_resample']}|S={accepts['spike']} "
                      f"{elapsed:.0f}s")
                sys.stdout.flush()
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    edges = int((mmag > 0).sum())
    overall = eval_facts(msign, mmag, H, wof, bp, inj_table, SPIKE_THR)

    print(f"\n{'='*60}")
    print(f"  LEARNABLE SPIKE RESULTS ({BUDGET} steps)")
    print(f"{'='*60}")
    print(f"  Answer: {overall*100:.2f}%")
    print(f"  Edges: {edges}")
    print(f"  Final spike_thr: {SPIKE_THR:.2f} (started at 5.0)")
    print(f"  Spike accepts: {accepts['spike']}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"\n  Spike threshold trajectory:")
    for i in range(0, len(spike_history), max(1, len(spike_history)//20)):
        bar = '#' * int(spike_history[i] * 3)
        print(f"    step {i:5d}: {spike_history[i]:6.2f} {bar}")
    print(f"{'='*60}")
    sys.stdout.flush()
