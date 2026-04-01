"""
A/B test: Superposition channel membership layer
=================================================
Hypothesis: neurons belonging to multiple channels with probabilistic
synchrony improves English bigram accuracy.

Mechanism:
  - Each neuron has a channel_mask (uint8 bitmask): which channels it belongs to
  - After each tick's spike decision: if ANY member of channel C fired,
    ALL other members fire with P = 1/membership_count
  - Neurons in 1 channel = 100% committed (relay)
  - Neurons in N channels = 1/N probability per activation (generalist/bridge)

Sweep: num_channels in [0 (baseline), 3 (ternary), 4 (quad), 8 (int8)]
  - 0: no synchrony — control, matches current C19 channel behavior
  - 3: ternary channel space, 3-bit mask, 7 meaningful membership patterns
  - 4: quad channel space, 4-bit mask, 15 meaningful membership patterns
  - 8: int8 channel space, 8-bit mask, 255 meaningful membership patterns

Mutation of channel_mask:
  - Flip one random bit in one random neuron's mask
  - Ensure at least 1 channel membership (mask != 0)
"""

import sys, os, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes, resolve_fineweb_path

# ── Config ────────────────────────────────────────────────────────────────────
IO          = 256
NV          = 4          # H = IO * NV = 1024 neurons
BUDGET      = 600        # steps per config
SEQ_LEN     = 150
N_EVAL_SEQS = 8
REPORT_EVERY = 100
PROJECTION_SCALE = 1.0
THETA_INIT  = 0.0
DECAY_LO    = 0.08
DECAY_HI    = 0.24
TICKS       = 8
INPUT_DUR   = 2
THRESHOLD   = 0.00005
SCHEDULE    = ['add','add','flip','decay','decay','decay','decay','decay']

CONFIGS = [
    {'num_channels': 0, 'label': 'baseline (no sync)'},
    {'num_channels': 3, 'label': 'ternary (3ch)'},
    {'num_channels': 4, 'label': 'quad (4ch)'},
    {'num_channels': 8, 'label': 'int8 (8ch)'},
]

SEED = 42
H = IO * NV  # 1024

# ── Projection pattern ────────────────────────────────────────────────────────
def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

# ── Forward pass with optional superposition synchrony ───────────────────────
def rollout_super(injected, *, mask, theta, decay, ticks, input_duration,
                  state, charge, sparse_cache, polarity, refractory,
                  channel_mask=None, num_channels=0, membership_counts=None):
    """
    Forward pass identical to SelfWiringGraph.rollout_token() plus an optional
    superposition synchrony step after each spike decision.
    """
    act = state.copy()
    cur_charge = charge.copy()
    ref = refractory.copy()
    theta_f = np.asarray(theta, dtype=np.float32)
    decay_f = np.asarray(decay, dtype=np.float32)
    injected_f = np.asarray(injected, dtype=np.float32)

    # Decay mode: scalar → int period subtract; array → per-neuron float
    is_scalar = decay_f.ndim == 0 or decay_f.shape == ()
    dp = max(1, int(round(1.0 / max(float(decay_f), 0.001)))) if is_scalar else 0

    rows, cols = sparse_cache

    for tick in range(ticks):
        # 1. DECAY
        if dp > 0:
            if tick % dp == 0:
                cur_charge = np.maximum(cur_charge - 1.0, 0.0)
        else:
            cur_charge = np.maximum(cur_charge - decay_f, 0.0)

        # 2. INPUT
        if tick < input_duration:
            act = act + injected_f

        # 3. PROPAGATE (sparse, multiply-free)
        raw = np.zeros(H, dtype=np.float32)
        if len(rows):
            np.add.at(raw, cols, act[rows])
        np.nan_to_num(raw, copy=False)
        cur_charge += raw

        # 4. CLAMP
        np.clip(cur_charge, 0.0, 15.0, out=cur_charge)

        # 5. SPIKE DECISION + REFRACTORY
        can_fire = (ref == 0)
        fired = (cur_charge >= theta_f) & can_fire
        ref[ref > 0] -= 1
        ref[fired] = 1

        # 5.5. SUPERPOSITION SYNCHRONY
        if num_channels > 0 and channel_mask is not None:
            for ch_bit in range(num_channels):
                ch_members = ((channel_mask >> ch_bit) & 1).astype(bool)
                if np.any(fired & ch_members):
                    # Channel activated — fire other members probabilistically
                    # P(fire) = 1 / membership_count (structured by count)
                    probs = 1.0 / membership_counts
                    candidates = ch_members & ~fired & can_fire
                    new_fires = candidates & (np.random.rand(H) < probs)
                    fired |= new_fires
                    ref[new_fires] = 1

        # 6. ACTIVATIONS + HARD RESET
        act = fired.astype(np.float32) * polarity
        cur_charge[fired] = 0.0

    return act, cur_charge, ref

# ── Evaluation ────────────────────────────────────────────────────────────────
def eval_accuracy(mask, theta, decay, channel_mask, num_channels,
                  membership_counts, polarity, seqs, bp, input_proj, output_proj,
                  refractory_init):
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    correct = 0; total = 0
    for seq in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        ref = refractory_init.copy()
        for i in range(len(seq) - 1):
            injected = bp[seq[i]] @ input_proj
            state, charge, ref = rollout_super(
                injected, mask=mask, theta=theta, decay=decay,
                ticks=TICKS, input_duration=INPUT_DUR,
                state=state, charge=charge, sparse_cache=sparse_cache,
                polarity=polarity, refractory=ref,
                channel_mask=channel_mask, num_channels=num_channels,
                membership_counts=membership_counts,
            )
            out = charge @ output_proj
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            if np.argmax(sims) == seq[i + 1]:
                correct += 1
            total += 1
    return correct / total if total else 0.0

def eval_score(mask, theta, decay, channel_mask, num_channels,
               membership_counts, polarity, seqs, bp, input_proj, output_proj,
               refractory_init, bigram):
    """Bigram cosine score for accept/reject."""
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0
    for seq in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        ref = refractory_init.copy()
        seq_score = 0.0; n = 0
        for i in range(len(seq) - 1):
            injected = bp[seq[i]] @ input_proj
            state, charge, ref = rollout_super(
                injected, mask=mask, theta=theta, decay=decay,
                ticks=TICKS, input_duration=INPUT_DUR,
                state=state, charge=charge, sparse_cache=sparse_cache,
                polarity=polarity, refractory=ref,
                channel_mask=channel_mask, num_channels=num_channels,
                membership_counts=membership_counts,
            )
            out = charge @ output_proj
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            tgt = bigram[seq[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            seq_score += cos; n += 1
        total += seq_score / n if n else 0.0
    return total / len(seqs)

# ── Channel mask helpers ──────────────────────────────────────────────────────
def init_channel_mask(num_channels, rng):
    """Random channel_mask: each neuron in exactly 1 channel initially."""
    if num_channels == 0:
        return None, None
    mask = np.zeros(H, dtype=np.uint8)
    for i in range(H):
        ch = rng.randint(0, num_channels)
        mask[i] = np.uint8(1 << ch)
    counts = membership_count(mask, num_channels)
    return mask, counts

def membership_count(channel_mask, num_channels):
    """popcount per neuron = number of channels each neuron belongs to."""
    counts = np.zeros(H, dtype=np.float32)
    for bit in range(num_channels):
        counts += ((channel_mask >> bit) & 1).astype(np.float32)
    return np.maximum(counts, 1.0)  # avoid div/0

def mutate_channel_mask(channel_mask, num_channels, rng):
    """Flip one random bit in one random neuron's mask. Ensure mask != 0."""
    new_mask = channel_mask.copy()
    idx = rng.randint(0, H)
    bit = rng.randint(0, num_channels)
    new_mask[idx] ^= np.uint8(1 << bit)
    if new_mask[idx] == 0:
        # Ensure at least 1 membership
        new_mask[idx] = np.uint8(1 << rng.randint(0, num_channels))
    return new_mask, membership_count(new_mask, num_channels)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_config(cfg, all_data, bigram, bp, input_proj, output_proj,
               polarity, eval_seqs):
    num_channels = cfg['num_channels']
    label = cfg['label']

    print(f"\n{'='*60}")
    print(f"  CONFIG: {label}")
    print(f"  num_channels={num_channels}, H={H}, budget={BUDGET}")
    print(f"{'='*60}")

    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)

    # Network — empty start
    mask = np.zeros((H, H), dtype=np.bool_)
    theta = np.full(H, THETA_INIT, dtype=np.float32)
    decay = np_rng.uniform(DECAY_LO, DECAY_HI, H).astype(np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    alive = []; alive_set = set()

    # Channel mask
    ch_rng = np.random.RandomState(SEED + 1)
    channel_mask, mem_counts = init_channel_mask(num_channels, ch_rng)

    def get_train_seqs(n=2):
        seqs = []
        for _ in range(n):
            off = np_rng.randint(0, len(all_data) - SEQ_LEN)
            seqs.append(all_data[off:off + SEQ_LEN])
        return seqs

    def get_sparse():
        rows, cols = np.where(mask)
        return rows.astype(np.intp), cols.astype(np.intp)

    accepts = 0
    t0 = time.time()
    results_log = []

    for step in range(1, BUDGET + 1):
        ptype = SCHEDULE[(step - 1) % len(SCHEDULE)]
        n_alive = int(np.sum(mask))
        if ptype in ('flip', 'decay') and n_alive == 0:
            ptype = 'add'

        # Propose mutation
        new_mask = mask; new_theta = theta; new_decay = decay
        new_ch_mask = channel_mask; new_mem = mem_counts

        if ptype == 'add':
            r = rng.randint(0, H - 1); c = rng.randint(0, H - 1)
            if r == c or mask[r, c]:
                continue
            new_mask = mask.copy(); new_mask[r, c] = True
        elif ptype == 'flip':
            alive_list = list(zip(*np.where(mask))) if n_alive > 0 else []
            if not alive_list: continue
            r, c = alive_list[rng.randint(0, len(alive_list) - 1)]
            nc = rng.randint(0, H - 1)
            if nc == r or nc == c or mask[r, nc]: continue
            new_mask = mask.copy(); new_mask[r, c] = False; new_mask[r, nc] = True
        elif ptype == 'theta':
            idx = rng.randint(0, H - 1)
            new_theta = theta.copy()
            new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-0.05, 0.05)))
        elif ptype == 'decay':
            idx = rng.randint(0, H - 1)
            new_decay = decay.copy()
            new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

        # Optionally also mutate channel_mask (1-in-5 chance)
        if num_channels > 0 and rng.randint(0, 4) == 0:
            new_ch_mask, new_mem = mutate_channel_mask(channel_mask, num_channels,
                                                       np.random.RandomState(rng.randint(0, 2**31)))

        train_seqs = get_train_seqs()
        sp_old = get_sparse()

        old_score = eval_score(mask, theta, decay, channel_mask, num_channels,
                               mem_counts, polarity, train_seqs, bp,
                               input_proj, output_proj, refractory, bigram)
        new_sp = SelfWiringGraph.build_sparse_cache(new_mask) if new_mask is not mask else sp_old
        new_score = eval_score(new_mask, new_theta, new_decay, new_ch_mask, num_channels,
                               new_mem, polarity, train_seqs, bp,
                               input_proj, output_proj, refractory, bigram)

        if new_score - old_score > THRESHOLD:
            mask = new_mask; theta = new_theta; decay = new_decay
            if num_channels > 0:
                channel_mask = new_ch_mask; mem_counts = new_mem
            accepts += 1

        if step % REPORT_EVERY == 0:
            elapsed = time.time() - t0
            acc = eval_accuracy(mask, theta, decay, channel_mask, num_channels,
                                mem_counts, polarity, eval_seqs, bp,
                                input_proj, output_proj, refractory, bigram)
            edges = int(np.sum(mask))

            if num_channels > 0:
                avg_mem = float(mem_counts.mean())
                multi = int(np.sum(mem_counts > 1))
                ch_info = f" avg_ch={avg_mem:.2f} multi={multi}"
            else:
                ch_info = ""

            line = (f"[{step:4d}] acc={acc*100:.1f}% edges={edges} "
                    f"accepts={accepts}{ch_info} {elapsed:.0f}s")
            print(f"  {line}")
            results_log.append({'step': step, 'acc': round(acc * 100, 2),
                                 'edges': edges, 'accepts': accepts})

    final_acc = eval_accuracy(mask, theta, decay, channel_mask, num_channels,
                              mem_counts, polarity, eval_seqs, bp,
                              input_proj, output_proj, refractory, bigram)
    elapsed = time.time() - t0
    print(f"  FINAL: {label} → {final_acc*100:.2f}% in {elapsed:.0f}s")
    return {'label': label, 'num_channels': num_channels,
            'final_acc': round(final_acc * 100, 2), 'log': results_log}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    all_data = load_fineweb_bytes()
    print(f"  {len(all_data)/1e6:.1f} MB")

    # Bigram table
    bigram_path = ROOT / 'recipes' / 'data' / 'bigram_table.npy'
    if bigram_path.exists():
        bigram = np.load(bigram_path)
    else:
        print("  Building bigram table...")
        os.makedirs(bigram_path.parent, exist_ok=True)
        counts = np.zeros((256, 256), dtype=np.float64)
        for i in range(len(all_data) - 1):
            counts[all_data[i], all_data[i + 1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        bigram = (counts / row_sums).astype(np.float32)
        np.save(bigram_path, bigram)
    print(f"  Bigram: {bigram.shape}")

    # Fixed projections + polarity
    ref = SelfWiringGraph(IO, hidden_ratio=NV, projection_scale=PROJECTION_SCALE, seed=SEED)
    bp = make_bp(IO)
    input_proj = ref.input_projection
    output_proj = ref.output_projection
    polarity = ref._polarity_f32

    # Fixed eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[off:off + SEQ_LEN]
                 for off in [eval_rng.randint(0, len(all_data) - SEQ_LEN)
                              for _ in range(N_EVAL_SEQS)]]

    # Run all configs
    all_results = []
    for cfg in CONFIGS:
        r = run_config(cfg, all_data, bigram, bp, input_proj, output_proj,
                       polarity, eval_seqs)
        all_results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        bar = '█' * int(r['final_acc'] / 2)
        print(f"  {r['label']:25s}  {r['final_acc']:5.2f}%  {bar}")
    print(f"{'='*60}")

    # Save results
    import json
    out_path = ROOT / 'recipes' / 'ab_superposition_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
