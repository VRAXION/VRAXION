"""
INSTNCT — A/B: Plain ReLU vs C19 Musical Gating
================================================
Compares two activation regimes head-to-head over a longer run.

Variant A — Plain ReLU (canonical recipe)
  act = max(charge - theta, 0)

Variant B — C19 Musical Gating (learnable rho + freq)
  wave      = sin(tick * freq[i] + phase[i])     per-neuron, per-tick
  threshold = theta[i] + rho[i] * wave[i]
  act[i]    = max(charge[i] - threshold[i], 0)

  Learnable via dedicated mutation operators:
    rho  — oscillation amplitude, range [0.0, 1.0], ±0.1 step
    freq — oscillation frequency, range [0.5, 2.0], resampled
  phase is frozen at random init (not mutated).

Both variants start from identical mask / theta / decay state,
use the same training data, and the same evaluation sequences.
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IO               = 256
NV               = 4               # H = 256×4 = 1024 neurons
N_WORKERS        = 18
BUDGET           = 3000            # steps per variant
SEQ_LEN          = 200
N_TRAIN_SEQS     = 2               # bigram eval: 2 seqs per worker
N_EVAL_SEQS      = 10              # classic accuracy for reporting
THRESHOLD        = 0.00005
PROJECTION_SCALE = 1.0
THETA_INIT       = 0.0
DECAY_INIT_LO    = 0.08
DECAY_INIT_HI    = 0.24
RHO_INIT_B       = 0.0             # Variant B: start silent, C19 only activates if rho mutates up

REPORT_EVERY     = 100             # print / log interval

# Variant A: triangle-derived canonical schedule
SCHEDULE_A = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']
# Variant B: same structure, 2 decay slots → rho + freq
SCHEDULE_B = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'rho', 'freq']

H = IO * NV  # 1024
# ─────────────────────────────────────────────────────────────────────────────

# ─── WORKER GLOBALS ──────────────────────────────────────────────────────────
_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_phase_g = None   # Variant B only — frozen throughout training


def init_w_a(b, d, sl, nt, wi, wo, bg):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg


def init_w_b(b, d, sl, nt, wi, wo, bg, ph):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram, _phase_g
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _phase_g = ph
# ─────────────────────────────────────────────────────────────────────────────


def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


# ─── VARIANT A — Plain ReLU ──────────────────────────────────────────────────

def _eval_bigram_a(mask, H, theta, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes) - 1):
            act = state.copy()
            for t in range(8):
                if t < 2:
                    act = act + _bp[text_bytes[i]] @ _input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (
                np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos; n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)


def worker_eval_a(args):
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
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        nc = rng.randint(0, H-1)
        if nc == r or nc == c or mask[r, nc] != 0:
            return {'delta': -1e9, 'type': 'flip'}
        new_mask = mask.copy(); new_mask[r, c] = 0.0; new_mask[r, nc] = 1.0
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram_a(mask, H, theta, decay, seqs)
    new_score = _eval_bigram_a(new_mask, H, new_theta, new_decay, seqs)

    return {
        'delta': new_score - old_score,
        'type': proposal_type,
        'new_mask_flat': (new_mask.flatten() if new_score > old_score else None)
                         if proposal_type in ('add', 'flip') else None,
        'new_decay': new_decay if (proposal_type == 'decay' and new_score > old_score) else None,
    }


def eval_accuracy_a(mask, H, input_projection, output_projection, theta, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        for t in range(8):
            if t < 2: act = act + bp[text_bytes[i]] @ input_projection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct / total if total else 0


# ─── VARIANT B — C19 Musical Gating ─────────────────────────────────────────

def _eval_bigram_b(mask, H, theta, decay, rho, freq, seqs):
    """C19 Musical Gating eval: additive per-neuron threshold oscillation."""
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes) - 1):
            act = state.copy()
            for t in range(8):
                if t < 2:
                    act = act + _bp[text_bytes[i]] @ _input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                wave = np.sin(np.float32(t) * freq + _phase_g)
                threshold = theta + rho * wave
                act = np.maximum(charge - threshold, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (
                np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos; n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)


def worker_eval_b(args):
    mask_flat, theta, decay, rho, freq, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_decay = decay; new_rho = rho; new_freq = freq

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        new_mask = mask.copy(); new_mask[r, c] = 1.0
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        nc = rng.randint(0, H-1)
        if nc == r or nc == c or mask[r, nc] != 0:
            return {'delta': -1e9, 'type': 'flip'}
        new_mask = mask.copy(); new_mask[r, c] = 0.0; new_mask[r, nc] = 1.0
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))
    elif proposal_type == 'rho':
        idx = rng.randint(0, H-1)
        new_rho = rho.copy()
        new_rho[idx] = max(0.0, min(1.0, rho[idx] + rng.uniform(-0.1, 0.1)))
    elif proposal_type == 'freq':
        idx = rng.randint(0, H-1)
        new_freq = freq.copy()
        new_freq[idx] = np.float32(np_rng.uniform(0.5, 2.0))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram_b(mask, H, theta, decay, rho, freq, seqs)
    new_score = _eval_bigram_b(new_mask, H, theta, new_decay, new_rho, new_freq, seqs)

    return {
        'delta': new_score - old_score,
        'type': proposal_type,
        'new_mask_flat': (new_mask.flatten() if new_score > old_score else None)
                         if proposal_type in ('add', 'flip') else None,
        'new_decay': new_decay if (proposal_type == 'decay' and new_score > old_score) else None,
        'new_rho':   new_rho   if (proposal_type == 'rho'   and new_score > old_score) else None,
        'new_freq':  new_freq  if (proposal_type == 'freq'  and new_score > old_score) else None,
    }


def eval_accuracy_b(mask, H, input_projection, output_projection,
                    theta, decay, rho, freq, phase, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        for t in range(8):
            if t < 2: act = act + bp[text_bytes[i]] @ input_projection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            wave = np.sin(np.float32(t) * freq + phase)
            threshold = theta + rho * wave
            act = np.maximum(charge - threshold, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct / total if total else 0


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bp = make_bp(IO)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    from lib.data import load_fineweb_bytes
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    bigram_path = os.path.join(BASE_DIR, "data", "bigram_table.npy")
    if os.path.exists(bigram_path):
        bigram = np.load(bigram_path)
    else:
        print("bigram_table.npy not found — generating from training data...")
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        counts = np.zeros((256, 256), dtype=np.float64)
        for i in range(len(ALL_DATA) - 1):
            counts[ALL_DATA[i], ALL_DATA[i+1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        bigram = (counts / row_sums).astype(np.float32)
        np.save(bigram_path, bigram)
        print("Generated and saved bigram_table.npy")
    print(f"Bigram table: {bigram.shape}")

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-SEQ_LEN) for _ in range(N_EVAL_SEQS)]]

    # Shared projections (same seed → identical for both variants)
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO, hidden_ratio=NV, projection_scale=PROJECTION_SCALE)
    input_projection = ref.input_projection
    output_projection = ref.output_projection

    # Shared initial state (empty start, same decay draw for both)
    init_mask  = np.zeros((H, H), dtype=np.float32)
    init_theta = np.zeros(H, dtype=np.float32)
    decay_rng  = np.random.RandomState(99)
    init_decay = decay_rng.uniform(DECAY_INIT_LO, DECAY_INIT_HI, H).astype(np.float32)

    # Variant B extras — deterministic separate seed so A is unchanged
    freq_rng   = np.random.RandomState(77)
    init_freq  = (freq_rng.rand(H) * 1.5 + 0.5).astype(np.float32)   # [0.5, 2.0]
    init_phase = (freq_rng.rand(H) * 2.0 * np.pi).astype(np.float32)  # frozen
    init_rho_b = np.full(H, RHO_INIT_B, dtype=np.float32)

    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  INSTNCT A/B — ReLU vs C19 Musical Gating")
    print(f"  {H}n, {N_WORKERS}w, {BUDGET} steps each")
    print(f"  A: SCHEDULE={SCHEDULE_A}")
    print(f"  B: SCHEDULE={SCHEDULE_B}, rho_init={RHO_INIT_B}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # ── VARIANT A — already completed, results from previous run ────────────
    ea_a      = 0.1859   # 18.59% at 3000 steps
    edges_a   = 206
    elapsed_a = 1213.0
    log_a     = []
    print(f"\n  A (cached): eval={ea_a*100:.2f}%  edges={edges_a}  {elapsed_a:.0f}s")

    # ── VARIANT B — C19 Musical Gating ───────────────────────────────────────
    print(f"\n--- VARIANT B: C19 Musical Gating ---")
    mask_b  = init_mask.copy()
    theta_b = init_theta.copy()
    decay_b = init_decay.copy()
    rho_b   = init_rho_b.copy()
    freq_b  = init_freq.copy()
    phase_b = init_phase.copy()   # frozen
    add_acc = flip_acc = decay_acc = rho_acc = freq_acc = accepts_b = 0
    log_b = []
    t0 = time.time()

    pool_b = Pool(N_WORKERS, initializer=init_w_b,
                  initargs=(bp, ALL_DATA, SEQ_LEN, N_TRAIN_SEQS,
                            input_projection, output_projection, bigram, phase_b))
    try:
        for step in range(1, BUDGET+1):
            ptype = SCHEDULE_B[(step-1) % len(SCHEDULE_B)]
            if ptype in ('flip', 'decay', 'rho', 'freq') and int((mask_b != 0).sum()) == 0:
                ptype = 'add'

            mask_flat = mask_b.flatten()
            args = [(mask_flat, theta_b.copy(), decay_b.copy(),
                     rho_b.copy(), freq_b.copy(),
                     H, 1000+step*50+w, ptype) for w in range(N_WORKERS)]
            results_w = pool_b.map(worker_eval_b, args)

            best = max(results_w, key=lambda x: x['delta'])
            if best['delta'] > THRESHOLD:
                if best['type'] in ('add', 'flip') and best['new_mask_flat'] is not None:
                    mask_b[:] = best['new_mask_flat'].reshape(H, H)
                    if best['type'] == 'add': add_acc += 1
                    else: flip_acc += 1
                elif best['type'] == 'decay' and best['new_decay'] is not None:
                    decay_b[:] = best['new_decay']; decay_acc += 1
                elif best['type'] == 'rho' and best['new_rho'] is not None:
                    rho_b[:] = best['new_rho']; rho_acc += 1
                elif best['type'] == 'freq' and best['new_freq'] is not None:
                    freq_b[:] = best['new_freq']; freq_acc += 1
                accepts_b += 1

            if step % REPORT_EVERY == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy_b(mask_b, H, input_projection, output_projection,
                              theta_b, decay_b, rho_b, freq_b, phase_b, s, bp)
                              for s in eval_seqs])
                edges = int((mask_b != 0).sum())
                sps = step / elapsed
                line = (f"[B {step:5d}] eval={ea*100:.1f}% edges={edges} "
                        f"[Ad={add_acc}|Fl={flip_acc}|Dc={decay_acc}|Rh={rho_acc}|Fr={freq_acc}] "
                        f"rho={rho_b.mean():.3f} freq={freq_b.mean():.3f} {elapsed:.0f}s ({sps:.2f}/s)")
                print(f"  {line}")
                log_b.append({'step': step, 'eval': round(ea*100, 2), 'edges': edges,
                              'Ad': add_acc, 'Fl': flip_acc, 'Dc': decay_acc,
                              'Rh': rho_acc, 'Fr': freq_acc,
                              'rho_m': round(float(rho_b.mean()), 4),
                              'freq_m': round(float(freq_b.mean()), 4),
                              'sps': round(sps, 2), 'time': int(elapsed)})
                sys.stdout.flush()
    finally:
        pool_b.terminate(); pool_b.join()

    elapsed_b = time.time() - t0
    ea_b = np.mean([eval_accuracy_b(mask_b, H, input_projection, output_projection,
                    theta_b, decay_b, rho_b, freq_b, phase_b, s, bp) for s in eval_seqs])
    edges_b = int((mask_b != 0).sum())
    print(f"\n  B FINAL: eval={ea_b*100:.2f}%  edges={edges_b}  "
          f"accepts={accepts_b}  {elapsed_b:.0f}s")
    print(f"  B convergence: rho={rho_b.mean():.4f} (init={RHO_INIT_B})  "
          f"freq={freq_b.mean():.4f} (init={init_freq.mean():.4f})")

    # ── COMPARISON ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  A/B RESULT — {BUDGET} steps each, {H} neurons, {N_WORKERS} workers")
    print(f"{'='*60}")
    print(f"  A (ReLU):  eval={ea_a*100:.2f}%  edges={edges_a}  {elapsed_a:.0f}s")
    print(f"  B (C19):   eval={ea_b*100:.2f}%  edges={edges_b}  {elapsed_b:.0f}s")
    delta = ea_b - ea_a
    print(f"  Delta B-A: {delta*100:+.2f}%")
    if abs(delta) < 0.005:
        verdict = "NO SIGNIFICANT DIFFERENCE"
    elif delta > 0:
        verdict = f"C19 WINS (+{delta*100:.2f}%)"
    else:
        verdict = f"RELU WINS ({delta*100:.2f}%)"
    print(f"  Verdict: {verdict}")
    print(f"{'='*60}")

    json_path = os.path.join(BASE_DIR, "ab_relu_vs_c19_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'budget': BUDGET, 'H': H, 'workers': N_WORKERS,
                'rho_init_b': RHO_INIT_B,
                'schedule_a': SCHEDULE_A, 'schedule_b': SCHEDULE_B,
            },
            'final_a': round(ea_a*100, 2), 'edges_a': edges_a,
            'final_b': round(ea_b*100, 2), 'edges_b': edges_b,
            'delta': round(delta*100, 3),
            'rho_final': round(float(rho_b.mean()), 4),
            'freq_final': round(float(freq_b.mean()), 4),
            'verdict': verdict,
            'log_a': log_a,
            'log_b': log_b,
        }, f, separators=(',', ':'), indent=2)
    print(f"  Results saved: {json_path}")
