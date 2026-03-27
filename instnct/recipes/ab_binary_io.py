"""
A/B Test: 8-bit Binary I/O vs 256-dim Random Projection I/O
============================================================
Mode A (256-DIM):  Current design — 256 random byte patterns, IO=256
Mode B (8-BIT):    8 binary neurons per I/O — byte as 8 bits

Both use tentacle I/O (Mode C style: random prefill + BFS connectivity).
Adaptive: runs until plateau, prints live, stops intelligently.
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
H = 256           # total neurons (both modes)
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
# Plateau: aggressive — stop fast if stuck
PLATEAU_WINDOW = 8
PLATEAU_THRESH_PCT = 0.5
PLATEAU_STRIKES_MAX = 3
PLATEAU_MIN_STEPS = 300

# Will be set per-mode
_IO = None
_H = None
_bp = None; _all_data = None; _seq_len = 100; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_polarity = None; _freq_g = None; _phase_g = None; _rho_g = None
_mode = None


def init_w(bp, d, sl, nt, bigram, pol, fr, ph, rh, mode, io_dim, h):
    global _bp, _all_data, _seq_len, _n_train, _bigram
    global _polarity, _freq_g, _phase_g, _rho_g, _mode, _IO, _H
    _bp, _all_data, _seq_len, _n_train = bp, d, sl, nt
    _bigram = bigram
    _polarity, _freq_g, _phase_g, _rho_g = pol, fr, ph, rh
    _mode = mode
    _IO = io_dim
    _H = h


def byte_to_bits(b):
    """Convert byte value (0-255) to 8-element float32 array of 0/1."""
    return np.array([(b >> i) & 1 for i in range(8)], dtype=np.float32)


def bits_to_byte(bits):
    """Convert 8-element array to byte value. Threshold at 0.5."""
    val = 0
    for i in range(min(8, len(bits))):
        if bits[i] > 0.5:
            val |= (1 << i)
    return val


def make_bp_256(seed=12345):
    """256 random byte patterns (current design)."""
    rng = np.random.RandomState(seed)
    p = rng.randn(256, 256).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


def make_bp_8():
    """8-bit binary byte patterns."""
    p = np.zeros((256, 8), dtype=np.float32)
    for i in range(256):
        p[i] = byte_to_bits(i)
    return p


def _inject(byte_val):
    if _mode == 'A':
        injected = np.zeros(_H, dtype=np.float32)
        injected[0:_IO] = _bp[byte_val]
        return injected
    else:
        # 8-bit: first 8 neurons get binary representation
        injected = np.zeros(_H, dtype=np.float32)
        injected[0:8] = byte_to_bits(byte_val)
        return injected


def _readout_logits(charge):
    """Always returns 256-dim logits (one per possible byte value)."""
    if _mode == 'A':
        # Last IO neurons → dot with each byte pattern → 256 logits
        out_vec = charge[_H - _IO:]
        logits = _bp @ out_vec  # (256, IO) @ (IO,) = (256,)... wait no
        # bp is (256, IO), out_vec is (IO,), so bp @ out_vec doesn't work
        # We need out_vec @ bp.T = (IO,) @ (IO, 256) = (256,)
        logits = out_vec @ _bp.T  # (IO,) @ (IO,256) hmm bp is (256,IO)
        # Actually: _bp is (256, IO). out_vec is (IO,). _bp @ out_vec would be wrong.
        # logits[b] = dot(bp[b], out_vec) for each byte b
        logits = _bp @ out_vec  # nope: (256,IO) @ (IO,) doesn't broadcast
        # Let's just do it explicitly
        logits = np.dot(_bp, out_vec)  # (256,IO) . (IO,) = (256,) YES this works
        return logits
    else:
        # Last 8 neurons → dot with each byte's bit pattern → 256 logits
        out_bits = charge[_H - 8:]
        # Precomputed: all 256 byte bit patterns
        logits = np.dot(_bp, out_bits)  # (256, 8) . (8,) = (256,)
        return logits


def _eval_bigram(mask, theta, decay, seqs):
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(_H, dtype=np.float32)
        charge = np.zeros(_H, dtype=np.float32)
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
            # Get 256-dim logits (both modes produce this)
            logits = _readout_logits(charge)
            # Softmax
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
    mask = mask_flat.reshape(_H, _H)
    new_mask = mask; new_theta = theta; new_decay = decay

    if proposal_type == 'add':
        r = rng.randint(0, _H - 1); c = rng.randint(0, _H - 1)
        if r == c or mask[r, c]:
            return {'delta': -1e9, 'type': 'add'}
        new_mask = mask.copy(); new_mask[r, c] = True
    elif proposal_type == 'remove':
        alive = list(zip(*np.where(mask)))
        if not alive:
            return {'delta': -1e9, 'type': 'remove'}
        r, c = alive[rng.randint(0, len(alive) - 1)]
        new_mask = mask.copy(); new_mask[r, c] = False
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive) - 1)]
        nc = rng.randint(0, _H - 1)
        if nc == r or nc == c or mask[r, nc]:
            return {'delta': -1e9, 'type': 'flip'}
        new_mask = mask.copy(); new_mask[r, c] = False; new_mask[r, nc] = True
    elif proposal_type == 'decay':
        idx = rng.randint(0, _H - 1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off + _seq_len])

    old_score = _eval_bigram(mask, theta, decay, seqs)
    new_score = _eval_bigram(new_mask, new_theta, new_decay, seqs)

    return {'delta': float(new_score - old_score), 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_decay': new_decay if proposal_type == 'decay' else None}


def eval_accuracy(mask, theta, decay, polarity, freq, phase, rho, text_bytes, bp, mode, io_dim):
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes) - 1):
        if mode == 'A':
            injected = np.zeros(H, dtype=np.float32)
            injected[0:io_dim] = bp[text_bytes[i]]
        else:
            injected = np.zeros(H, dtype=np.float32)
            injected[0:8] = byte_to_bits(text_bytes[i])
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=mask, theta=theta, decay=decay,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge,
            sparse_cache=sparse_cache, polarity=polarity,
            freq=freq, phase=phase, rho=rho,
        )
        if mode == 'A':
            out_vec = charge[H - io_dim:]
            logits = np.dot(bp, out_vec)  # (256, IO) . (IO,) = (256,)
        else:
            out_bits = charge[H - 8:]
            logits = np.dot(bp, out_bits)  # (256, 8) . (8,) = (256,)
        if np.argmax(logits) == text_bytes[i + 1]:
            correct += 1
        total += 1
    return correct / total if total else 0


def bfs_reachable(mask, sources, targets):
    visited = set(sources)
    queue = deque(sources)
    target_set = set(targets)
    while queue:
        node = queue.popleft()
        for n in np.where(mask[node])[0]:
            if n in target_set:
                return True
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return False


def ensure_connectivity(mask, rng, io_in, io_out, hidden):
    bridges = 0
    for _ in range(500):
        if bfs_reachable(mask, list(range(io_in)), list(range(H - io_out, H))):
            return bridges
        s = rng.choice(hidden) if rng.random() < 0.5 else rng.randint(0, io_in)
        t = rng.choice(hidden) if rng.random() < 0.5 else rng.randint(H - io_out, H)
        if s != t and not mask[s, t]:
            mask[s, t] = True
            bridges += 1
    return bridges


def run_mode(mode, bp, ALL_DATA, bigram, eval_seqs):
    if mode == 'A':
        io_dim = 256
        io_in = 256
        io_out = 256
        hidden_count = H - io_in - io_out  # 256-256-256 < 0 !!
        # Problem: H=256 can't fit 256+256 I/O neurons!
        # Use IO=64 for mode A (like previous sweep) to keep it fair on same H
        io_dim = 64
        io_in = 64
        io_out = 64
        hidden_count = H - io_in - io_out  # 128
        label = "256-DIM (IO=64)"
        # Rebuild bp for 64-dim
        rng = np.random.RandomState(12345)
        bp = rng.randn(256, 64).astype(np.float32)
        bp /= np.linalg.norm(bp, axis=1, keepdims=True)
    else:
        io_dim = 8
        io_in = 8
        io_out = 8
        hidden_count = H - io_in - io_out  # 240!
        label = "8-BIT BINARY (IO=8)"

    print(f"\n{'='*60}")
    print(f"  Mode {mode}: {label}")
    print(f"  H={H}, IO_in={io_in}, IO_out={io_out}, hidden={hidden_count}")
    print(f"{'='*60}")

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(io_dim, 16), hidden=H, projection_scale=1.0)
    polarity_f32 = ref.polarity.astype(np.float32)

    # Init mask: random 5% + BFS connectivity
    init_rng = np.random.RandomState(42)
    mask = (init_rng.rand(H, H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(mask, False)
    hidden_neurons = list(range(io_in, H - io_out))
    bridges = ensure_connectivity(mask, init_rng, io_in, io_out, hidden_neurons)

    theta = np.full(H, THETA_INIT, dtype=np.float32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(DECAY_INIT_LO, DECAY_INIT_HI, H).astype(np.float32)

    init_edges = int(mask.sum())
    print(f"  Init: {init_edges} edges ({init_edges/(H*H)*100:.1f}%), {bridges} bridges")
    print(f"  Hidden neurons: {hidden_count} (B has {240-128}={240-128} MORE than A)")
    print()

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp, ALL_DATA, SEQ_LEN, N_TRAIN_SEQS, bigram,
                          polarity_f32, ref.freq, ref.phase, ref.rho,
                          mode, io_dim, H))

    add_acc = 0; rem_acc = 0; flip_acc = 0; decay_acc = 0
    accepts = 0
    eval_history = []
    plateau_strikes = 0
    recent_deltas = deque(maxlen=50)
    t0 = time.time()
    best_eval = 0
    stall_steps = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            ptype = SCHEDULE[(step - 1) % len(SCHEDULE)]
            edges = int(mask.sum())
            if ptype in ('flip', 'remove', 'decay') and edges == 0:
                ptype = 'add'

            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay.copy(),
                     1000 + step * 50 + w, ptype) for w in range(N_WORKERS)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            accepted = False
            if best_r['delta'] > THRESHOLD:
                if best_r['type'] in ('add', 'remove', 'flip') and best_r['new_mask_flat'] is not None:
                    mask[:] = best_r['new_mask_flat'].reshape(H, H)
                    if best_r['type'] == 'add': add_acc += 1
                    elif best_r['type'] == 'remove': rem_acc += 1
                    else: flip_acc += 1
                    accepted = True
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    decay[:] = best_r['new_decay']
                    decay_acc += 1
                    accepted = True
                if accepted:
                    accepts += 1
            recent_deltas.append(best_r['delta'] if best_r['delta'] > -1e8 else 0.0)

            if step % EVAL_EVERY == 0:
                elapsed = time.time() - t0
                ea = np.mean([eval_accuracy(
                    mask, theta, decay, polarity_f32,
                    ref.freq, ref.phase, ref.rho, s, bp, mode, io_dim
                ) for s in eval_seqs])
                edges = int(mask.sum())
                sps = step / elapsed

                if ea > best_eval:
                    best_eval = ea
                    stall_steps = 0
                else:
                    stall_steps += EVAL_EVERY

                eval_history.append({'step': step, 'eval': round(ea * 100, 2)})

                recent_rate = sum(1 for d in recent_deltas if d > THRESHOLD) / max(len(recent_deltas), 1)
                print(f"  [{mode}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={edges} acc={accepts} "
                      f"[A={add_acc} R={rem_acc} F={flip_acc} D={decay_acc}] "
                      f"delta_rate={recent_rate*100:.0f}% "
                      f"{elapsed:.0f}s ({sps:.1f}sps)")
                sys.stdout.flush()

                # Plateau check
                if len(eval_history) >= PLATEAU_WINDOW and step >= PLATEAU_MIN_STEPS:
                    window = [e['eval'] for e in eval_history[-PLATEAU_WINDOW:]]
                    spread = max(window) - min(window)
                    if spread < PLATEAU_THRESH_PCT:
                        plateau_strikes += 1
                        if plateau_strikes >= PLATEAU_STRIKES_MAX:
                            print(f"  ** PLATEAU STOP at step {step} "
                                  f"(spread={spread:.2f}% for {PLATEAU_STRIKES_MAX} windows)")
                            break
                    else:
                        plateau_strikes = 0

                # Absolute stall: no improvement for 400 steps
                if stall_steps >= 400 and step >= PLATEAU_MIN_STEPS:
                    print(f"  ** STALL STOP at step {step} (no improvement for {stall_steps} steps)")
                    break

    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    print(f"\n  [{mode}] DONE: best={best_eval*100:.1f}% edges={int(mask.sum())} "
          f"accepts={accepts} {elapsed:.0f}s")
    return {'mode': mode, 'label': label, 'best': best_eval,
            'edges': int(mask.sum()), 'accepts': accepts,
            'steps': eval_history[-1]['step'] if eval_history else 0,
            'time': elapsed, 'history': eval_history}


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
        for i in range(len(ALL_DATA) - 1):
            counts[ALL_DATA[i], ALL_DATA[i + 1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        bigram = (counts / row_sums).astype(np.float32)
        np.save(bigram_path, bigram)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN) for _ in range(N_EVAL_SEQS)]]

    bp_256 = make_bp_256()
    bp_8 = make_bp_8()

    results = []
    for m, bp in [('A', bp_256), ('B', bp_8)]:
        r = run_mode(m, bp, ALL_DATA, bigram, eval_seqs)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"  A/B: 256-DIM vs 8-BIT BINARY I/O")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['mode']} ({r['label']:25s}): best={r['best']*100:.1f}%  "
              f"edges={r['edges']}  steps={r['steps']}  {r['time']:.0f}s")
    diff = results[1]['best'] - results[0]['best']
    w = results[1] if diff > 0 else results[0]
    print(f"\n  WINNER: {w['mode']} ({w['label']}) by {abs(diff)*100:.1f}%")
    print(f"  Hidden neuron advantage for B: {240-128}=112 extra neurons")
    print(f"{'='*60}")
