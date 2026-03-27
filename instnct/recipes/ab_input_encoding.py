"""
Input Encoding Sweep: Which byte→neuron mapping learns best?
=============================================================
All modes use tentacle I/O (first N=input, last 64=output, hidden in middle).
Output is always 64-dim random projection (proven to work).
Only the INPUT encoding varies.

A: RANDOM_64   — 64-dim random unit vectors (baseline, proven 4.4%)
B: BINARY_8    — 8 raw bits (proven bad: 0.2%)
C: FOURIER_64  — 32 sin/cos harmonics = 64 neurons
D: SDR_64      — Sparse Distributed: each byte activates exactly 13/64 neurons
E: MULTISCALE  — 8 bits + 16 hi-nibble + 16 lo-nibble + 8 parity = 48 neurons

Adaptive plateau detection, detailed live output.
"""
import sys, os, time, random, json
from collections import deque
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# === CONFIG ===
H = 256
N_WORKERS = 18
SEQ_LEN = 100
N_TRAIN_SEQS = 2
N_EVAL_SEQS = 5
THRESHOLD = 0.00005
TICKS = 8
INPUT_DURATION = 2
THETA_INIT = 5.0
DECAY_INIT_LO = 0.08
DECAY_INIT_HI = 0.24
INIT_DENSITY = 0.05
EVAL_EVERY = 20
SCHEDULE = ['add', 'remove', 'flip', 'flip', 'decay', 'decay', 'decay', 'decay']
MAX_STEPS = 5000
PLATEAU_WINDOW = 8
PLATEAU_THRESH_PCT = 0.5
PLATEAU_STRIKES_MAX = 3
PLATEAU_MIN_STEPS = 300
OUTPUT_DIM = 64  # output always 64

# ============================================================
# ENCODING BUILDERS — each returns (bp_table[256, D], D)
# ============================================================

def build_random_64(seed=12345):
    """A: Random unit vectors, 64-dim."""
    rng = np.random.RandomState(seed)
    p = rng.randn(256, 64).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p, 64

def build_binary_8():
    """B: Raw 8-bit binary."""
    p = np.zeros((256, 8), dtype=np.float32)
    for i in range(256):
        for b in range(8):
            p[i, b] = float((i >> b) & 1)
    return p, 8

def build_fourier_64():
    """C: 32 sin/cos harmonics = 64 neurons. Smooth, structured."""
    D = 64
    p = np.zeros((256, D), dtype=np.float32)
    for v in range(256):
        for k in range(D // 2):
            freq = 2.0 * np.pi * (k + 1) / 256.0
            p[v, 2*k] = np.sin(freq * v)
            p[v, 2*k+1] = np.cos(freq * v)
    # Normalize rows
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p /= np.maximum(norms, 1e-8)
    return p, 64

def build_sdr_64(seed=42):
    """D: Sparse Distributed Representation. Each byte activates exactly 13/64 neurons."""
    K = 13  # ~20% sparsity
    D = 64
    rng = np.random.RandomState(seed)
    p = np.zeros((256, D), dtype=np.float32)
    for v in range(256):
        active = rng.choice(D, size=K, replace=False)
        p[v, active] = 1.0
    return p, 64

def build_multiscale():
    """E: 8 bits + 16 hi-nibble onehot + 16 lo-nibble onehot + 8 parity features = 48."""
    D = 48
    p = np.zeros((256, D), dtype=np.float32)
    for v in range(256):
        # Bits 0-7: raw binary
        for b in range(8):
            p[v, b] = float((v >> b) & 1)
        # Bits 8-23: high nibble one-hot (16)
        hi = (v >> 4) & 0xF
        p[v, 8 + hi] = 1.0
        # Bits 24-39: low nibble one-hot (16)
        lo = v & 0xF
        p[v, 24 + lo] = 1.0
        # Bits 40-47: parity features
        p[v, 40] = float(bin(v).count('1') % 2)     # overall parity
        p[v, 41] = float(bin(v).count('1') / 8.0)   # popcount normalized
        p[v, 42] = float(v / 255.0)                  # linear position
        p[v, 43] = float((v ^ (v >> 1)) / 255.0)    # gray code normalized
        p[v, 44] = np.sin(2 * np.pi * v / 256.0)    # low-freq sine
        p[v, 45] = np.cos(2 * np.pi * v / 256.0)    # low-freq cosine
        p[v, 46] = np.sin(4 * np.pi * v / 256.0)    # mid-freq sine
        p[v, 47] = np.cos(4 * np.pi * v / 256.0)    # mid-freq cosine
    return p, 48

ENCODINGS = {
    'A': ('RANDOM_64',  build_random_64),
    'B': ('BINARY_8',   build_binary_8),
    'C': ('FOURIER_64', build_fourier_64),
    'D': ('SDR_64',     build_sdr_64),
    'E': ('MULTISCALE',  build_multiscale),
}

# ============================================================
# WORKER GLOBALS
# ============================================================
_bp_in = None; _bp_out = None; _all_data = None; _bigram = None
_polarity = None; _freq_g = None; _phase_g = None; _rho_g = None
_in_dim = None; _out_dim = None

def init_w(bp_in, bp_out, data, bigram, pol, fr, ph, rh, in_dim, out_dim):
    global _bp_in, _bp_out, _all_data, _bigram
    global _polarity, _freq_g, _phase_g, _rho_g, _in_dim, _out_dim
    _bp_in = bp_in; _bp_out = bp_out; _all_data = data; _bigram = bigram
    _polarity = pol; _freq_g = fr; _phase_g = ph; _rho_g = rh
    _in_dim = in_dim; _out_dim = out_dim

def _inject(byte_val):
    injected = np.zeros(H, dtype=np.float32)
    injected[0:_in_dim] = _bp_in[byte_val]
    return injected

def _readout(charge):
    out_vec = charge[H - _out_dim:]
    logits = np.dot(_bp_out, out_vec)  # (256, out_dim) . (out_dim,) = (256,)
    return logits

def _eval_bigram(mask, theta, decay, seqs):
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes) - 1):
            injected = _inject(text_bytes[i])
            state, charge = SelfWiringGraph.rollout_token(
                injected, mask=mask, theta=theta, decay=decay,
                ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge,
                sparse_cache=sparse_cache, polarity=_polarity,
                freq=_freq_g, phase=_phase_g, rho=_rho_g,
            )
            logits = _readout(charge)
            e = np.exp(logits - logits.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos
            n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    mask_flat, theta, decay, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta; new_decay = decay

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c]:
            return {'delta': -1e9, 'type': 'add'}
        new_mask = mask.copy(); new_mask[r, c] = True
    elif proposal_type == 'remove':
        alive = list(zip(*np.where(mask)))
        if not alive: return {'delta': -1e9, 'type': 'remove'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask = mask.copy(); new_mask[r, c] = False
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask)))
        if not alive: return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        nc = rng.randint(0, H-1)
        if nc == r or nc == c or mask[r, nc]:
            return {'delta': -1e9, 'type': 'flip'}
        new_mask = mask.copy(); new_mask[r, c] = False; new_mask[r, nc] = True
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

    seqs = []
    for _ in range(2):
        off = np_rng.randint(0, len(_all_data) - 100)
        seqs.append(_all_data[off:off+100])

    old = _eval_bigram(mask, theta, decay, seqs)
    new = _eval_bigram(new_mask, new_theta, new_decay, seqs)
    return {'delta': float(new - old), 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new > old else None,
            'new_decay': new_decay if proposal_type == 'decay' else None}

def eval_accuracy(mask, theta, decay, pol, freq, phase, rho, text_bytes, bp_in, bp_out, in_dim, out_dim):
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        inj = np.zeros(H, np.float32)
        inj[0:in_dim] = bp_in[text_bytes[i]]
        state, charge = SelfWiringGraph.rollout_token(
            inj, mask=mask, theta=theta, decay=decay,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sparse_cache,
            polarity=pol, freq=freq, phase=phase, rho=rho)
        logits = np.dot(bp_out, charge[H - out_dim:])
        if np.argmax(logits) == text_bytes[i+1]: correct += 1
        total += 1
    return correct / total if total else 0

# BFS connectivity
def ensure_connectivity(mask, rng, io_in, io_out):
    from collections import deque
    for _ in range(500):
        visited = set(range(io_in))
        queue = deque(range(io_in))
        targets = set(range(H - io_out, H))
        found = False
        while queue:
            node = queue.popleft()
            for n in np.where(mask[node])[0]:
                if n in targets: found = True; break
                if n not in visited:
                    visited.add(n); queue.append(n)
            if found: break
        if found: return
        # Add bridge
        s = rng.randint(io_in, H - io_out)
        t = rng.randint(io_in, H - io_out)
        if s != t: mask[s, t] = True

def run_mode(mode_key, ALL_DATA, bigram, eval_seqs):
    label, builder = ENCODINGS[mode_key]
    bp_in, in_dim = builder()
    # Output: always 64-dim random
    rng_out = np.random.RandomState(12345)
    bp_out = rng_out.randn(256, OUTPUT_DIM).astype(np.float32)
    bp_out /= np.linalg.norm(bp_out, axis=1, keepdims=True)
    out_dim = OUTPUT_DIM
    hidden = H - in_dim - out_dim

    print(f"\n{'='*65}")
    print(f"  {mode_key}: {label}")
    print(f"  in={in_dim}, out={out_dim}, hidden={hidden}, H={H}")
    print(f"{'='*65}")

    if hidden < 16:
        print(f"  !! SKIP: only {hidden} hidden neurons, not enough")
        return {'mode': mode_key, 'label': label, 'best': 0, 'steps': 0, 'time': 0,
                'edges': 0, 'accepts': 0, 'hidden': hidden}

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(in_dim, 16), hidden=H, projection_scale=1.0)
    pol = ref.polarity.astype(np.float32)

    init_rng = np.random.RandomState(42)
    mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(mask, False)
    ensure_connectivity(mask, init_rng, in_dim, out_dim)

    theta = np.full(H, THETA_INIT, np.float32)
    decay = np.random.RandomState(99).uniform(DECAY_INIT_LO, DECAY_INIT_HI, H).astype(np.float32)

    print(f"  Init: {int(mask.sum())} edges ({mask.sum()/(H*H)*100:.1f}%)")

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, ALL_DATA, bigram,
                          pol, ref.freq, ref.phase, ref.rho, in_dim, out_dim))

    add_acc=0; rem_acc=0; flip_acc=0; decay_acc=0; accepts=0
    eval_history = []; plateau_strikes = 0
    best_eval = 0; stall_steps = 0
    t0 = time.time()

    try:
        for step in range(1, MAX_STEPS+1):
            ptype = SCHEDULE[(step-1) % len(SCHEDULE)]
            if ptype in ('flip','remove','decay') and mask.sum() == 0: ptype = 'add'

            args = [(mask.flatten(), theta.copy(), decay.copy(),
                     1000+step*50+w, ptype) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            accepted = False
            if best_r['delta'] > THRESHOLD:
                if best_r['type'] in ('add','remove','flip') and best_r['new_mask_flat'] is not None:
                    mask[:] = best_r['new_mask_flat'].reshape(H, H)
                    if best_r['type'] == 'add': add_acc += 1
                    elif best_r['type'] == 'remove': rem_acc += 1
                    else: flip_acc += 1
                    accepted = True
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    decay[:] = best_r['new_decay']; decay_acc += 1; accepted = True
                if accepted: accepts += 1

            if step % EVAL_EVERY == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy(mask, theta, decay, pol,
                    ref.freq, ref.phase, ref.rho, s, bp_in, bp_out, in_dim, out_dim)
                    for s in eval_seqs])
                if ea > best_eval: best_eval = ea; stall_steps = 0
                else: stall_steps += EVAL_EVERY
                eval_history.append({'step': step, 'eval': round(ea*100, 2)})

                print(f"  [{mode_key}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={int(mask.sum())} [A={add_acc} R={rem_acc} F={flip_acc} D={decay_acc}] "
                      f"{elapsed:.0f}s ({step/elapsed:.1f}sps)")
                sys.stdout.flush()

                if len(eval_history) >= PLATEAU_WINDOW and step >= PLATEAU_MIN_STEPS:
                    window = [e['eval'] for e in eval_history[-PLATEAU_WINDOW:]]
                    if max(window) - min(window) < PLATEAU_THRESH_PCT:
                        plateau_strikes += 1
                        if plateau_strikes >= PLATEAU_STRIKES_MAX:
                            print(f"  ** PLATEAU at step {step}"); break
                    else: plateau_strikes = 0
                if stall_steps >= 400 and step >= PLATEAU_MIN_STEPS:
                    print(f"  ** STALL at step {step}"); break
    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    print(f"  [{mode_key}] DONE: best={best_eval*100:.1f}% hidden={hidden} {elapsed:.0f}s")
    return {'mode': mode_key, 'label': label, 'best': best_eval, 'hidden': hidden,
            'steps': eval_history[-1]['step'] if eval_history else 0,
            'edges': int(mask.sum()), 'accepts': accepts, 'time': elapsed}


if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram_path = os.path.join(BASE_DIR, "data", "bigram_table.npy")
    if os.path.exists(bigram_path):
        bigram = np.load(bigram_path)
    else:
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        counts = np.zeros((256, 256), dtype=np.float64)
        for i in range(len(ALL_DATA)-1):
            counts[ALL_DATA[i], ALL_DATA[i+1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True); row_sums[row_sums==0] = 1
        bigram = (counts/row_sums).astype(np.float32)
        np.save(bigram_path, bigram)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN]
                 for off in [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(N_EVAL_SEQS)]]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', default='A,C,D,E',
                        help='Comma-separated modes to run (default: A,C,D,E — skip B which is known bad)')
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(',')]

    results = []
    for m in modes:
        if m not in ENCODINGS:
            print(f"Unknown mode: {m}"); continue
        r = run_mode(m, ALL_DATA, bigram, eval_seqs)
        results.append(r)

    print(f"\n{'='*65}")
    print(f"  INPUT ENCODING SWEEP RESULTS")
    print(f"{'='*65}")
    print(f"  {'Mode':<4} {'Label':<16} {'Hidden':>6} {'Best':>6} {'Steps':>6} {'Time':>6}")
    print(f"  {'-'*4} {'-'*16} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        print(f"  {r['mode']:<4} {r['label']:<16} {r['hidden']:6d} "
              f"{r['best']*100:5.1f}% {r['steps']:6d} {r['time']:5.0f}s")
    best = max(results, key=lambda x: x['best'])
    print(f"\n  WINNER: {best['mode']} ({best['label']}) — {best['best']*100:.1f}%")
    print(f"{'='*65}")
