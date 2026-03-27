"""
A/B/C Test: Holographic Projection vs Tentacle I/O Architectures
================================================================
Mode A (HOLOGRAPHIC):      Fixed random V×H projection broadcasts input to ALL neurons.
                           Output = charge @ output_projection. Current design.
Mode B (TENTACLES_IO):     First V neurons = input, last V = output. Structured I/O
                           seeding (3-5 edges per I/O neuron + hidden chains).
Mode C (TENTACLES_RANDOM): Same tentacle I/O, but fully random prefill (~5% density).
                           BFS guarantees paths exist input→output. No special I/O seeding.

Fair comparison: same H, same eval data, same schedule, same forward pass.
Plateau detection: stops early if eval flatlines for extended period.

Usage:
  python ab_projection_vs_tentacles.py                    # run all three
  python ab_projection_vs_tentacles.py --mode A           # only holographic
  python ab_projection_vs_tentacles.py --mode B           # only tentacles structured
  python ab_projection_vs_tentacles.py --mode C           # only tentacles random
  python ab_projection_vs_tentacles.py --budget 5000      # longer run
"""
import sys, os, time, random, json, argparse
from collections import deque
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# === CONFIG ===
IO = 64
H = 256
N_WORKERS = 18
BUDGET = 3000
SEQ_LEN = 100
N_TRAIN_SEQS = 2
N_EVAL_SEQS = 5
THRESHOLD = 0.00005
TICKS = 8
INPUT_DURATION = 2
THETA_INIT = 5.0
DECAY_INIT_LO = 0.08
DECAY_INIT_HI = 0.24
INIT_DENSITY_AB = 0.03   # 3% for modes A, B
INIT_DENSITY_C = 0.05    # 5% for mode C (full random, needs more material)
EVAL_EVERY = 25          # eval frequency (steps)
SCHEDULE = ['add', 'remove', 'flip', 'flip', 'decay', 'decay', 'decay', 'decay']

# Plateau detection
PLATEAU_WINDOW = 10       # eval points in window (10 × 25 = 250 steps)
PLATEAU_THRESH = 0.005    # max-min < 0.5% in window = plateau
PLATEAU_STRIKES = 3       # 3 consecutive plateau windows → stop
PLATEAU_MIN_STEPS = 500   # don't stop before this

MODE_LABELS = {
    'A': 'HOLOGRAPHIC',
    'B': 'TENTACLES_IO',
    'C': 'TENTACLES_RANDOM',
}

# --- Worker globals ---
_bp = None; _all_data = None; _seq_len = 100; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None
_polarity = None; _freq_g = None; _phase_g = None; _rho_g = None
_mode = None


def init_w(b, d, sl, nt, wi, wo, bg, pol, fr, ph, rh, mode):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    global _polarity, _freq_g, _phase_g, _rho_g, _mode
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg
    _polarity, _freq_g, _phase_g, _rho_g = pol, fr, ph, rh
    _mode = mode


def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


def _inject(byte_val):
    if _mode == 'A':
        return _bp[byte_val] @ _input_projection
    else:
        injected = np.zeros(H, dtype=np.float32)
        injected[0:IO] = _bp[byte_val]
        return injected


def _readout(charge):
    if _mode == 'A':
        return charge @ _output_projection
    else:
        return charge[H - IO:]


def _eval_bigram(mask, theta, decay, seqs):
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
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
            out = _readout(charge)
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
    mask_flat, theta, decay, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta; new_decay = decay

    if proposal_type == 'add':
        r = rng.randint(0, H - 1); c = rng.randint(0, H - 1)
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
        nc = rng.randint(0, H - 1)
        if nc == r or nc == c or mask[r, nc]:
            return {'delta': -1e9, 'type': 'flip'}
        new_mask = mask.copy(); new_mask[r, c] = False; new_mask[r, nc] = True
    elif proposal_type == 'theta':
        idx = rng.randint(0, H - 1)
        new_theta = theta.copy()
        new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-0.05, 0.05)))
    elif proposal_type == 'decay':
        idx = rng.randint(0, H - 1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off + _seq_len])

    old_score = _eval_bigram(mask, theta, decay, seqs)
    new_score = _eval_bigram(new_mask, new_theta, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_decay': new_decay if proposal_type == 'decay' else None}


def eval_accuracy_detailed(mask, theta, decay, polarity, freq, phase, rho,
                           text_bytes, bp, ip, op, mode):
    """Eval accuracy + detailed internal stats (charge, spikes, zone activity)."""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    charge_samples = []
    spike_counts = []

    for i in range(len(text_bytes) - 1):
        if mode == 'A':
            injected = bp[text_bytes[i]] @ ip
        else:
            injected = np.zeros(H, dtype=np.float32)
            injected[0:IO] = bp[text_bytes[i]]
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=mask, theta=theta, decay=decay,
            ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge,
            sparse_cache=sparse_cache, polarity=polarity,
            freq=freq, phase=phase, rho=rho,
        )
        # Collect stats
        charge_samples.append(charge.copy())
        spike_counts.append(int(np.sum(state != 0)))

        if mode == 'A':
            out = charge @ op
        else:
            out = charge[H - IO:]
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i + 1]:
            correct += 1
        total += 1

    acc = correct / total if total else 0
    charges = np.array(charge_samples)
    return {
        'accuracy': acc,
        'charge_mean': float(charges.mean()),
        'charge_std': float(charges.std()),
        'charge_max': float(charges.max()),
        'spike_mean': float(np.mean(spike_counts)),
        'spike_std': float(np.std(spike_counts)),
        'firing_rate': float(np.mean(spike_counts)) / H,
    }


def edge_zone_stats(mask):
    """Count edges by zone: in>hid, hid>hid, hid>out, in>out, etc."""
    in_zone = slice(0, IO)
    hid_zone = slice(IO, H - IO)
    out_zone = slice(H - IO, H)
    return {
        'in_to_hid':  int(mask[in_zone, hid_zone].sum()),
        'in_to_out':  int(mask[in_zone, out_zone].sum()),
        'in_to_in':   int(mask[in_zone, in_zone].sum()),
        'hid_to_hid': int(mask[hid_zone, hid_zone].sum()),
        'hid_to_out': int(mask[hid_zone, out_zone].sum()),
        'hid_to_in':  int(mask[hid_zone, in_zone].sum()),
        'out_to_hid': int(mask[out_zone, hid_zone].sum()),
        'out_to_out': int(mask[out_zone, out_zone].sum()),
        'out_to_in':  int(mask[out_zone, in_zone].sum()),
    }


def io_activity(mask):
    """How many I/O neurons have at least 1 connection."""
    in_out_deg = mask[0:IO, :].sum(axis=1)     # output degree of input neurons
    out_in_deg = mask[:, H-IO:H].sum(axis=0)   # input degree of output neurons
    return {
        'in_active': int((in_out_deg > 0).sum()),
        'in_total': IO,
        'out_active': int((out_in_deg > 0).sum()),
        'out_total': IO,
    }


def bfs_reachable(mask, sources, targets):
    """BFS from source set, return True if any target is reachable."""
    visited = set(sources)
    queue = deque(sources)
    target_set = set(targets)
    while queue:
        node = queue.popleft()
        neighbors = np.where(mask[node])[0]
        for n in neighbors:
            if n in target_set:
                return True
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return False


def ensure_connectivity(mask, rng):
    """Add random bridge edges until input zone can reach output zone via BFS."""
    input_nodes = list(range(IO))
    output_nodes = list(range(H - IO, H))
    hidden_nodes = list(range(IO, H - IO))
    bridges_added = 0
    max_attempts = 500

    for attempt in range(max_attempts):
        if bfs_reachable(mask, input_nodes, output_nodes):
            return bridges_added
        # Add a random bridge: pick a random node, connect to random other
        # Prefer connecting across zones
        if rng.random() < 0.4:
            # input → hidden
            s = rng.choice(input_nodes)
            t = rng.choice(hidden_nodes)
        elif rng.random() < 0.7:
            # hidden → hidden (cross-cluster bridge)
            s = rng.choice(hidden_nodes)
            t = rng.choice(hidden_nodes)
        else:
            # hidden → output
            s = rng.choice(hidden_nodes)
            t = rng.choice(output_nodes)
        if s != t and not mask[s, t]:
            mask[s, t] = True
            bridges_added += 1

    return bridges_added


def check_plateau(eval_history, min_steps):
    """Check if we should stop due to plateau."""
    if len(eval_history) < PLATEAU_WINDOW:
        return False, 0
    if eval_history[-1]['step'] < min_steps:
        return False, 0

    window = [e['eval'] for e in eval_history[-PLATEAU_WINDOW:]]
    spread = max(window) - min(window)
    is_plateau = spread < PLATEAU_THRESH * 100  # eval is in %, thresh is fraction

    return is_plateau, spread


def run_one(mode, bp, ALL_DATA, bigram, eval_seqs):
    global _mode
    _mode = mode
    label = MODE_LABELS[mode]

    print(f"\n{'='*70}")
    print(f"  Mode {mode}: {label}")
    print(f"  H={H}, V={IO}, hidden={H-2*IO if mode != 'A' else H}, "
          f"budget={BUDGET}, workers={N_WORKERS}")
    print(f"  schedule={SCHEDULE}")
    print(f"{'='*70}")

    # Build network (deterministic seed)
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO, hidden=H, projection_scale=1.0)
    polarity_f32 = ref.polarity.astype(np.float32)

    input_projection = ref.input_projection if mode == 'A' else None
    output_projection = ref.output_projection if mode == 'A' else None

    # --- MASK INIT ---
    init_rng = np.random.RandomState(42)

    if mode == 'C':
        # Mode C: full random 5%, then BFS connectivity repair
        mask = (init_rng.rand(H, H) < INIT_DENSITY_C).astype(bool)
        np.fill_diagonal(mask, False)
        repair_rng = np.random.RandomState(78)
        bridges = ensure_connectivity(mask, repair_rng)
        print(f"  [C] Random 5% init + {bridges} bridge edges for connectivity")
    else:
        # Mode A & B: 3% random base
        mask = (init_rng.rand(H, H) < INIT_DENSITY_AB).astype(bool)
        np.fill_diagonal(mask, False)

    if mode == 'B':
        # Structured I/O seeding
        seed_rng = np.random.RandomState(77)
        hidden_start = IO
        hidden_end = H - IO
        for i in range(IO):
            n_out = seed_rng.randint(3, 6)
            targets = seed_rng.randint(hidden_start, hidden_end, size=n_out)
            for t in targets:
                mask[i, t] = True
        for o in range(H - IO, H):
            n_in = seed_rng.randint(3, 6)
            sources = seed_rng.randint(hidden_start, hidden_end, size=n_in)
            for s in sources:
                mask[s, o] = True
        for _ in range(IO):
            chain_len = seed_rng.randint(2, 5)
            nodes = seed_rng.randint(hidden_start, hidden_end, size=chain_len)
            for j in range(len(nodes) - 1):
                if nodes[j] != nodes[j + 1]:
                    mask[nodes[j], nodes[j + 1]] = True

    theta = np.full(H, THETA_INIT, dtype=np.float32)
    decay_rng = np.random.RandomState(99)
    decay = decay_rng.uniform(DECAY_INIT_LO, DECAY_INIT_HI, H).astype(np.float32)

    init_edges = int(mask.sum())
    init_density = init_edges / (H * H)
    zones = edge_zone_stats(mask)
    io_act = io_activity(mask)

    print(f"  Init: {init_edges} edges ({init_density*100:.2f}%)")
    print(f"  Zones: in>hid={zones['in_to_hid']} hid>hid={zones['hid_to_hid']} "
          f"hid>out={zones['hid_to_out']} in>out={zones['in_to_out']}")
    print(f"  I/O active: in={io_act['in_active']}/{IO} out={io_act['out_active']}/{IO}")
    print()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG = os.path.join(BASE_DIR, f"ab_{mode}_live.txt")
    JSON_LOG = os.path.join(ROOT, f"training_live_data.json")

    with open(LOG, "w") as f:
        f.write(f"--- {label} START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"H={H} V={IO} budget={BUDGET} init_edges={init_edges}\n")

    add_acc = 0; rem_acc = 0; flip_acc = 0; theta_acc = 0; decay_acc = 0
    accepts = 0
    log_data = []
    eval_history = []
    plateau_strikes = 0
    prev_edges = init_edges
    recent_accepts = deque(maxlen=50)  # rolling window of accept/reject
    recent_deltas = deque(maxlen=50)
    stopped_early = False
    t0 = time.time()

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp, ALL_DATA, SEQ_LEN, N_TRAIN_SEQS,
                          input_projection, output_projection, bigram,
                          polarity_f32, ref.freq, ref.phase, ref.rho, mode))
    try:
        for step in range(1, BUDGET + 1):
            ptype = SCHEDULE[(step - 1) % len(SCHEDULE)]
            edges = int(mask.sum())
            if ptype in ('flip', 'remove', 'theta', 'decay') and edges == 0:
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
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta[:] = best_r['new_theta']
                    theta_acc += 1
                    accepted = True
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    decay[:] = best_r['new_decay']
                    decay_acc += 1
                    accepted = True
                if accepted:
                    accepts += 1

            recent_accepts.append(1 if accepted else 0)
            recent_deltas.append(float(best_r['delta']) if best_r['delta'] > -1e8 else 0.0)

            # --- EVAL ---
            if step % EVAL_EVERY == 0:
                elapsed = time.time() - t0
                edges = int(mask.sum())
                density = edges / (H * H)
                sps = step / elapsed

                # Detailed eval on first eval sequence
                det = eval_accuracy_detailed(
                    mask, theta, decay, polarity_f32,
                    ref.freq, ref.phase, ref.rho, eval_seqs[0], bp,
                    input_projection, output_projection, mode
                )
                # Quick accuracy on all eval sequences
                all_acc = [eval_accuracy_detailed(
                    mask, theta, decay, polarity_f32,
                    ref.freq, ref.phase, ref.rho, s, bp,
                    input_projection, output_projection, mode
                )['accuracy'] for s in eval_seqs]
                ea = np.mean(all_acc)

                zones = edge_zone_stats(mask)
                io_act = io_activity(mask)
                edge_delta = edges - prev_edges
                prev_edges = edges

                recent_rate = sum(recent_accepts) / max(len(recent_accepts), 1)
                recent_best = max(recent_deltas) if recent_deltas else 0

                # Print detailed stats
                print(f"  [{mode}][{step:5d}] eval={ea*100:.1f}% | "
                      f"edges={edges} (d={edge_delta:+d}) | density={density*100:.2f}%")
                print(f"           accepts: A={add_acc} R={rem_acc} F={flip_acc} "
                      f"T={theta_acc} D={decay_acc} (total={accepts}, rate={accepts/step*100:.1f}%)")
                print(f"           theta:  mean={theta.mean():.3f} std={theta.std():.3f} "
                      f"min={theta.min():.3f} max={theta.max():.3f}")
                print(f"           decay:  mean={decay.mean():.4f} std={decay.std():.4f} "
                      f"min={decay.min():.4f} max={decay.max():.4f}")
                print(f"           charge: mean={det['charge_mean']:.2f} std={det['charge_std']:.2f} "
                      f"max={det['charge_max']:.2f}")
                print(f"           spikes: mean={det['spike_mean']:.1f}/step "
                      f"std={det['spike_std']:.1f} (firing_rate={det['firing_rate']*100:.1f}%)")
                if mode != 'A':
                    print(f"           I/O:    in_active={io_act['in_active']}/{IO} "
                          f"out_active={io_act['out_active']}/{IO}")
                    print(f"           zones:  in>hid={zones['in_to_hid']} "
                          f"hid>hid={zones['hid_to_hid']} hid>out={zones['hid_to_out']} "
                          f"in>out={zones['in_to_out']}")
                print(f"           recent: accept_rate={recent_rate*100:.0f}% "
                      f"best_delta={recent_best:.6f}")
                print(f"           speed:  {elapsed:.0f}s elapsed ({sps:.2f} sps)")
                print()

                with open(LOG, "a") as f:
                    f.write(f"[{step:5d}] eval={ea*100:.1f}% edges={edges} "
                            f"A={add_acc} R={rem_acc} F={flip_acc} T={theta_acc} D={decay_acc} "
                            f"charge={det['charge_mean']:.2f} spikes={det['spike_mean']:.1f} "
                            f"firing={det['firing_rate']*100:.1f}%\n")

                entry = {
                    'step': step, 'eval': round(ea * 100, 2), 'edges': edges,
                    'density': round(density, 5),
                    'A': add_acc, 'R': rem_acc, 'F': flip_acc, 'T': theta_acc, 'D': decay_acc,
                    'accepts': accepts,
                    'theta_m': round(float(theta.mean()), 4),
                    'theta_std': round(float(theta.std()), 4),
                    'theta_min': round(float(theta.min()), 4),
                    'theta_max': round(float(theta.max()), 4),
                    'decay_m': round(float(decay.mean()), 4),
                    'decay_std': round(float(decay.std()), 4),
                    'decay_min': round(float(decay.min()), 4),
                    'decay_max': round(float(decay.max()), 4),
                    'charge_mean': round(det['charge_mean'], 3),
                    'charge_std': round(det['charge_std'], 3),
                    'charge_max': round(det['charge_max'], 3),
                    'spike_mean': round(det['spike_mean'], 2),
                    'spike_std': round(det['spike_std'], 2),
                    'firing_rate': round(det['firing_rate'], 4),
                    'recent_accept_rate': round(recent_rate, 3),
                    'recent_best_delta': round(recent_best, 6),
                    'sps': round(sps, 2), 'time': int(elapsed),
                    'mode': mode,
                }
                if mode != 'A':
                    entry.update({
                        'in_active': io_act['in_active'],
                        'out_active': io_act['out_active'],
                        'z_in_hid': zones['in_to_hid'],
                        'z_hid_hid': zones['hid_to_hid'],
                        'z_hid_out': zones['hid_to_out'],
                        'z_in_out': zones['in_to_out'],
                    })
                log_data.append(entry)
                eval_history.append(entry)

                with open(JSON_LOG, 'w') as f:
                    json.dump(log_data, f, separators=(',', ':'))
                sys.stdout.flush()

                # Plateau detection
                is_plateau, spread = check_plateau(eval_history, PLATEAU_MIN_STEPS)
                if is_plateau:
                    plateau_strikes += 1
                    print(f"  !! Plateau detected (spread={spread:.2f}%, "
                          f"strike {plateau_strikes}/{PLATEAU_STRIKES})")
                    if plateau_strikes >= PLATEAU_STRIKES:
                        print(f"  ** EARLY STOP at step {step} — plateau for "
                              f"{PLATEAU_STRIKES * PLATEAU_WINDOW * EVAL_EVERY} steps")
                        stopped_early = True
                        break
                else:
                    plateau_strikes = 0

    finally:
        pool.terminate(); pool.join()

    elapsed = time.time() - t0
    final_ea = np.mean([eval_accuracy_detailed(
        mask, theta, decay, polarity_f32,
        ref.freq, ref.phase, ref.rho, s, bp,
        input_projection, output_projection, mode
    )['accuracy'] for s in eval_seqs])

    peak_eval = max(e['eval'] for e in eval_history) if eval_history else 0
    final_step = eval_history[-1]['step'] if eval_history else 0

    print(f"\n  [{mode}] {'STOPPED' if stopped_early else 'FINISHED'} at step {final_step}")
    print(f"  [{mode}] FINAL: eval={final_ea*100:.1f}%  peak={peak_eval:.1f}%  "
          f"edges={int(mask.sum())}  accepts={accepts}  {elapsed:.0f}s")

    return {
        'mode': mode, 'label': label,
        'eval': final_ea, 'peak': peak_eval,
        'edges': int(mask.sum()), 'accepts': accepts,
        'time': elapsed, 'steps': final_step,
        'stopped_early': stopped_early,
        'log': log_data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B/C I/O architecture sweep")
    parser.add_argument('--mode', choices=['A', 'B', 'C', 'all'], default='all')
    parser.add_argument('--budget', type=int, default=BUDGET)
    args = parser.parse_args()
    BUDGET = args.budget

    bp = make_bp(IO)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

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
    print(f"Bigram table: {bigram.shape}")

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN] for off in
                 [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN) for _ in range(N_EVAL_SEQS)]]

    modes = ['A', 'B', 'C'] if args.mode == 'all' else [args.mode]
    results = []
    for m in modes:
        r = run_one(m, bp, ALL_DATA, bigram, eval_seqs)
        results.append(r)

    # === FINAL COMPARISON ===
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  A/B/C RESULTS COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Mode':<4} {'Label':<20} {'Final':>6} {'Peak':>6} {'Edges':>7} "
              f"{'Accept':>7} {'Steps':>6} {'Time':>6} {'Stop':>6}")
        print(f"  {'-'*4} {'-'*20} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
        for r in results:
            print(f"  {r['mode']:<4} {r['label']:<20} {r['eval']*100:5.1f}% "
                  f"{r['peak']:5.1f}% {r['edges']:7d} {r['accepts']:7d} "
                  f"{r['steps']:6d} {r['time']:5.0f}s "
                  f"{'YES' if r['stopped_early'] else 'no':>6}")

        best = max(results, key=lambda x: x['peak'])
        print(f"\n  WINNER (by peak): {best['mode']} ({best['label']}) — {best['peak']:.1f}%")
        print(f"{'='*70}")

    # Save combined results
    combined_log = os.path.join(ROOT, "training_live_data.json")
    all_logs = []
    for r in results:
        all_logs.extend(r['log'])
    with open(combined_log, 'w') as f:
        json.dump(all_logs, f, separators=(',', ':'))
    print(f"Combined log: {combined_log}")
