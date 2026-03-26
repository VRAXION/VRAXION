"""
INSTNCT — English Training Candidate
====================================
1024 neurons, 18 workers, bigram distribution eval.
Validated recipe candidate from the 2026-03-21 sweep session.

Config:
  - Bigram cosine eval (2 seq per worker, 3x faster than classic)
  - Threshold: 0.00005 (from adaptive sweep convergence)
  - Scale: 1.0 (projection_scale, sweep confirmed optimal)
  - Theta: 0 init (C19 additive formula: effective_theta = max(0, theta + rho*sin(t*freq+phase)))
  - Ticks: 8 (sweep: 8 > 6 > 4)
  - Injection: 2 ticks (sweep: 2 > 4 > 1 > 8, +3.26% vs tick-0-only)
  - Decay init: random [0.08, 0.24] per-neuron (23.72% peak vs 21.96% fix)
  - Schedule: triangle-derived 2 add / 1 flip / 5 decay (8-step fixed approximation)
  - Empty start (no checkpoint needed)
  - Forward pass matches graph.py: binary spikes, polarity, C19 soft-wave, MAX_CHARGE clamp
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_polarity = None   # per-neuron polarity float32 (+1 excitatory, -1 inhibitory)
_freq_g   = None   # per-neuron oscillation frequency (Musical Gating)
_phase_g  = None   # per-neuron oscillation phase (Musical Gating)
_rho_g    = None   # per-neuron modulation depth (Musical Gating)

def init_w(b, d, sl, nt, wi, wo, bg, pol, fr, ph, rh):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    global _polarity, _freq_g, _phase_g, _rho_g
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _polarity, _freq_g, _phase_g, _rho_g = pol, fr, ph, rh

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, theta, decay, seqs):
    """Bigram cosine eval — forward pass matches graph.py rollout_token():
    binary spikes, polarity-signed edges, C19 soft-wave threshold, MAX_CHARGE clamp."""
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs] * _polarity[rs]   # signed: inhibitory sources send -1
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(8):
                if t < 2:
                    act = act + _bp[text_bytes[i]] @ _input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw
                np.clip(charge, 0.0, 15.0, out=charge)   # MAX_CHARGE clamp
                charge *= ret
                # C19 soft-wave: additive threshold modulation per tick
                wave = np.sin(np.float32(t) * _freq_g + _phase_g)
                effective_theta = np.maximum(0.0, theta + _rho_g * wave)
                # Binary spike + polarity output (matches rollout_token)
                fired = charge >= effective_theta
                act = fired.astype(np.float32) * _polarity
            state = act.copy()
            out = charge @ _output_projection
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
    mask_flat, theta, decay, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta; new_decay = decay

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        new_mask = mask.copy(); new_mask[r, c] = 1.0
    elif proposal_type == 'flip':
        # Binary mask: flip = rewire (move edge to random target)
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        nc = rng.randint(0, H-1)
        if nc == r or nc == c or mask[r, nc] != 0:
            return {'delta': -1e9, 'type': 'flip'}
        new_mask = mask.copy(); new_mask[r, c] = 0.0; new_mask[r, nc] = 1.0
    elif proposal_type == 'theta':
        idx = rng.randint(0, H-1)
        new_theta = theta.copy()
        new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-0.05, 0.05)))
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, theta, decay, seqs)
    new_score = _eval_bigram(new_mask, H, new_theta, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_decay': new_decay if proposal_type == 'decay' else None}

def eval_accuracy(mask, H, input_projection, output_projection, theta, decay,
                  polarity, freq, phase, rho, text_bytes, bp):
    """Classic accuracy for reporting — same forward pass as _eval_bigram."""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs] * polarity[rs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(8):
            if t < 2: act = act + bp[text_bytes[i]] @ input_projection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw
            np.clip(charge, 0.0, 15.0, out=charge)
            charge *= ret
            wave = np.sin(np.float32(t) * freq + phase)
            effective_theta = np.maximum(0.0, theta + rho * wave)
            fired = charge >= effective_theta
            act = fired.astype(np.float32) * polarity
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0

if __name__ == "__main__":
    # === CONFIG ===
    IO = 256
    NV = 4              # 256*4 = 1024 neurons
    N_WORKERS = 18
    BUDGET = 1000       # medium run
    SEQ_LEN = 200
    N_TRAIN_SEQS = 2    # bigram eval needs only 2 seqs (3x faster)
    N_EVAL_SEQS = 10    # classic accuracy for reporting
    THRESHOLD = 0.00005 # from adaptive sweep convergence
    PROJECTION_SCALE = 1.0  # sweep confirmed optimal
    THETA_INIT = 0.0    # redundant with charge ReLU (sweep: theta=0 = theta=3)
    DECAY_INIT_LO = 0.08   # random init range (sweep: [0.08,0.24] > fix 0.15)
    DECAY_INIT_HI = 0.24

    # Triangle converged: add=22%/flip=16%/decay=62% → approx 2a/1f/5d
    SCHEDULE = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']  # ~22/12/63%

    H = IO * NV  # 1024

    bp = make_bp(IO)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load training data
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    # Load or generate bigram table
    bigram_path = os.path.join(BASE_DIR, "data", "bigram_table.npy")
    if os.path.exists(bigram_path):
        bigram = np.load(bigram_path)
    else:
        print("bigram_table.npy not found — generating from training data...")
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        counts = np.zeros((256, 256), dtype=np.float64)
        for i in range(len(ALL_DATA) - 1):
            counts[ALL_DATA[i], ALL_DATA[i + 1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        bigram = (counts / row_sums).astype(np.float32)
        np.save(bigram_path, bigram)
        print(f"Generated and saved bigram_table.npy")
    print(f"Bigram table: {bigram.shape}")

    # Fixed eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(N_EVAL_SEQS)]]

    # Build network with deterministic projections
    random.seed(42); np.random.seed(42)
    net = SelfWiringGraph(IO, hidden_ratio=NV)
    # projection_scale=1.0 (sweep confirmed: beats default 3.0)
    proj_rng = np.random.RandomState(42)
    # Reconstruct input_projection/output_projection at scale=1.0 using same seed as SelfWiringGraph
    # (SelfWiringGraph uses its own proj_rng, so we replicate)
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO, hidden_ratio=NV, projection_scale=PROJECTION_SCALE)
    input_projection = ref.input_projection  # undo 3.0, apply 1.0
    output_projection = ref.output_projection

    # Empty start
    net.mask[:] = 0; net.alive = []; net.alive_set = set(); net._sync_sparse_idx()
    net.theta[:] = THETA_INIT
    decay_rng = np.random.RandomState(99)
    net.decay[:] = decay_rng.uniform(DECAY_INIT_LO, DECAY_INIT_HI, H).astype(np.float32)
    net.state *= 0; net.charge *= 0

    # Canonical biological parameters from graph.py (fixed — not mutated in this recipe)
    polarity_f32 = ref.polarity.astype(np.float32)  # ±1 per neuron (10% inhibitory)

    print(f"\n{'='*60}")
    print(f"  INSTNCT — English Recipe Candidate")
    print(f"  {H}n, {N_WORKERS}w, bigram {N_TRAIN_SEQS}seq, thresh={THRESHOLD}")
    print(f"  scale={PROJECTION_SCALE}, theta={THETA_INIT}, decay=[{DECAY_INIT_LO},{DECAY_INIT_HI}]")
    print(f"  schedule={SCHEDULE}")
    print(f"  budget={BUDGET} steps")
    print(f"{'='*60}")
    sys.stdout.flush()

    LOG = os.path.join(BASE_DIR, "english_1024n_live.txt")
    JSON_LOG = os.path.join(BASE_DIR, "training_live_data.json")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    with open(LOG, "w") as f:
        f.write(f"--- START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"INSTNCT candidate: {H}n, {N_WORKERS}w, bigram {N_TRAIN_SEQS}seq, "
                f"thresh={THRESHOLD}, scale={PROJECTION_SCALE}\n")

    add_acc = 0; flip_acc = 0; theta_acc = 0; decay_acc = 0
    accepts = 0
    log_data = []
    t0 = time.time()

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp, ALL_DATA, SEQ_LEN, N_TRAIN_SEQS, input_projection, output_projection, bigram,
                          polarity_f32, ref.freq, ref.phase, ref.rho))
    try:
        for step in range(1, BUDGET+1):
            ptype = SCHEDULE[(step - 1) % len(SCHEDULE)]
            if ptype in ('flip', 'theta', 'decay') and net.count_connections() == 0:
                ptype = 'add'

            mask_flat = net.mask.flatten()
            args = [(mask_flat, net.theta.copy(), net.decay.copy(), H,
                     1000+step*50+w, ptype) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > THRESHOLD:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    net.mask[:] = best_r['new_mask_flat'].reshape(H, H)
                    net.resync_alive()
                    if best_r['type'] == 'add': add_acc += 1
                    else: flip_acc += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    net.theta[:] = best_r['new_theta']
                    theta_acc += 1
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    net.decay[:] = best_r['new_decay']
                    decay_acc += 1
                accepts += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy(net.mask, H, input_projection, output_projection,
                              net.theta, net.decay, polarity_f32, ref.freq, ref.phase, ref.rho, s, bp)
                              for s in eval_seqs])
                edges = net.count_connections()
                sps = step / elapsed

                # Neuron distribution stats
                inh_count = int(np.sum(polarity_f32 < 0))
                exc_count = H - inh_count
                rho_nz = int(np.sum(ref.rho > 0.01))
                rho_max = float(ref.rho.max())
                freq_m = float(ref.freq.mean())
                freq_std = float(ref.freq.std())
                theta_min = float(net.theta.min())
                theta_max = float(net.theta.max())
                decay_min = float(net.decay.min())
                decay_max = float(net.decay.max())

                line = (f"[{step:5d}] eval={ea*100:.1f}% edges={edges} "
                        f"[A={add_acc}|F={flip_acc}|T={theta_acc}|D={decay_acc}] "
                        f"theta={net.theta.mean():.4f}[{theta_min:.3f},{theta_max:.3f}] "
                        f"decay={net.decay.mean():.4f}[{decay_min:.3f},{decay_max:.3f}] "
                        f"rho_nz={rho_nz} rho_max={rho_max:.4f} "
                        f"freq={freq_m:.4f}±{freq_std:.4f} "
                        f"inh={inh_count}/exc={exc_count} "
                        f"{elapsed:.0f}s ({sps:.2f} step/s)")
                print(f"  {line}")
                with open(LOG, "a") as f:
                    f.write(line + "\n")

                log_data.append({
                    'step': step, 'eval': round(ea * 100, 2), 'edges': edges,
                    'A': add_acc, 'F': flip_acc, 'T': theta_acc, 'D': decay_acc,
                    'theta_m': round(float(net.theta.mean()), 4),
                    'theta_min': round(theta_min, 4), 'theta_max': round(theta_max, 4),
                    'decay_m': round(float(net.decay.mean()), 4),
                    'decay_min': round(decay_min, 4), 'decay_max': round(decay_max, 4),
                    'rho_nz': rho_nz, 'rho_max': round(rho_max, 4),
                    'freq_m': round(freq_m, 4), 'freq_std': round(freq_std, 4),
                    'inh': inh_count, 'exc': exc_count,
                    'sps': round(sps, 2), 'time': int(elapsed)
                })
                with open(JSON_LOG, 'w') as f:
                    json.dump(log_data, f, separators=(',', ':'))
                sys.stdout.flush()

            if step % 500 == 0:
                ckpt = os.path.join(CKPT_DIR, f"instnct_step{step}.npz")
                net.save(ckpt)
                print(f"  SAVED: {ckpt}")
                sys.stdout.flush()

    finally:
        pool.terminate(); pool.join()
        final_ckpt = os.path.join(CKPT_DIR, "instnct_canonical_latest.npz")
        net.save(final_ckpt)
        print(f"  SAVED FINAL: {final_ckpt}")

    elapsed = time.time() - t0
    final_ea = np.mean([eval_accuracy(net.mask, H, input_projection, output_projection,
                        net.theta, net.decay, polarity_f32, ref.freq, ref.phase, ref.rho, s, bp)
                        for s in eval_seqs])
    print(f"\nFINAL: eval={final_ea*100:.1f}% edges={net.count_connections()} "
          f"accepts={accepts} {elapsed:.0f}s ({BUDGET/elapsed:.2f} step/s)")

