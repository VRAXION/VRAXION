"""
English 4096 neurons, GPU (PyTorch) dense forward
==================================================
Full byte-range (256 I/O), pattern encoding, real English text.
Dense matmul on GPU, batched sequences per proposal.

Key difference from CPU version:
- Forward pass runs on GPU as batched dense matmul
- Proposals evaluated sequentially but each batches all train seqs
- old_score computed ONCE per step (shared across all proposals)
- N_PROPOSALS >> 18 because GPU forward is fast
"""
import sys, os, time, random, json
import numpy as np
import torch

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Pattern encoding ---
def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

# --- GPU forward pass ---
def forward_seq_batch(W, input_projection, output_projection, theta, decay, bp_torch, text_seqs_batch, ticks=6):
    """
    Forward pass: batch of text sequences through the network.

    W: (H, H) dense weight matrix on GPU
    text_seqs_batch: list of 1D uint8 tensors on GPU, each (seq_len,)
    Returns: (total_score, n_correct, n_total)
    """
    H = W.shape[0]
    B = len(text_seqs_batch)
    seq_len = text_seqs_batch[0].shape[0]
    ret = 1.0 - decay  # (H,)

    # Stack all sequences: (B, seq_len) as long indices
    seqs = torch.stack(text_seqs_batch).long()  # (B, seq_len)

    # bp_norm for cosine similarity at output
    bp_norm = bp_torch / (bp_torch.norm(dim=1, keepdim=True) + 1e-8)

    # Process all B sequences in parallel
    state = torch.zeros(B, H, device=DEVICE)
    charge = torch.zeros(B, H, device=DEVICE)

    total_correct = 0
    total_prob = 0.0
    total_n = 0

    for i in range(seq_len - 1):
        act = state.clone()
        for t in range(ticks):
            if t == 0:
                # Inject: bp[byte] @ input_projection -> (B, H)
                inp = bp_torch[seqs[:, i]]  # (B, io_dim)
                act = act + inp @ input_projection  # (B, H)
            raw = act @ W.T  # (B, H) @ (H, H) -> (B, H)
            charge = charge + raw
            charge = charge * ret  # per-neuron decay
            act = torch.clamp(charge - theta, min=0.0)
            charge = torch.clamp(charge, -1.0, 1.0)
        state = act.clone()

        # Output: cosine similarity
        out = charge @ output_projection  # (B, V)
        out_n = out / (out.norm(dim=1, keepdim=True) + 1e-8)  # (B, V)
        sims = out_n @ bp_norm.T  # (B, 256)

        # Softmax + accuracy
        probs = torch.softmax(sims, dim=1)  # (B, 256)
        targets = seqs[:, i + 1]  # (B,)
        preds = sims.argmax(dim=1)  # (B,)
        total_correct += (preds == targets).sum().item()
        total_prob += probs[torch.arange(B, device=DEVICE), targets].sum().item()
        total_n += B

    acc = total_correct / total_n if total_n else 0
    avg_p = total_prob / total_n if total_n else 0
    score = 0.5 * acc + 0.5 * avg_p
    return score


def eval_accuracy_only(W, input_projection, output_projection, theta, decay, bp_torch, text_seqs_batch, ticks=6):
    """Just accuracy (no prob), for reporting."""
    H = W.shape[0]
    B = len(text_seqs_batch)
    seq_len = text_seqs_batch[0].shape[0]
    ret = 1.0 - decay
    bp_norm = bp_torch / (bp_torch.norm(dim=1, keepdim=True) + 1e-8)
    seqs = torch.stack(text_seqs_batch).long()
    state = torch.zeros(B, H, device=DEVICE)
    charge = torch.zeros(B, H, device=DEVICE)
    correct = 0; total = 0
    for i in range(seq_len - 1):
        act = state.clone()
        for t in range(ticks):
            if t == 0:
                act = act + bp_torch[seqs[:, i]] @ input_projection
            raw = act @ W.T
            charge = charge + raw
            charge = charge * ret
            act = torch.clamp(charge - theta, min=0.0)
            charge = torch.clamp(charge, -1.0, 1.0)
        state = act.clone()
        out = charge @ output_projection
        out_n = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        sims = out_n @ bp_norm.T
        correct += (sims.argmax(dim=1) == seqs[:, i + 1]).sum().item()
        total += B
    return correct / total if total else 0


if __name__ == "__main__":
    IO = 256
    H = IO * 4   # 1024 neurons (N=3072, benchmark sweet spot)
    N_PROPOSALS = 32  # proposals per step
    BUDGET = 20000
    SEQ_LEN = 200
    N_TRAIN_SEQS = 5
    N_EVAL_SEQS = 8
    TICKS = 6
    DRIVE = 0.6

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    bp = make_bp(IO)
    bp_torch = torch.from_numpy(bp).to(DEVICE)

    # Load training data
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    DATA_LEN = len(ALL_DATA)
    ALL_DATA_TORCH = torch.from_numpy(ALL_DATA.copy()).to(DEVICE)
    print(f"Loaded {DATA_LEN / 1e6:.1f} MB text")

    # Fixed eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_seqs = []
    for _ in range(N_EVAL_SEQS):
        off = eval_rng.randint(0, DATA_LEN - SEQ_LEN)
        eval_seqs.append(ALL_DATA_TORCH[off:off + SEQ_LEN])

    print(f"{H} neurons, I/O={IO}, {N_PROPOSALS} proposals/step, budget={BUDGET}")
    print(f"Train: {N_TRAIN_SEQS}x{SEQ_LEN} RANDOM per step | Eval: {N_EVAL_SEQS}x{SEQ_LEN} fixed")
    print(f"Dense W: {H*H*4/1e6:.1f} MB on GPU")
    sys.stdout.flush()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize network on GPU
    proj_rng = np.random.RandomState(42)
    input_projection_np = proj_rng.randn(IO, H).astype(np.float32)
    input_projection_np /= np.linalg.norm(input_projection_np, axis=0, keepdims=True)
    output_projection_np = proj_rng.randn(H, IO).astype(np.float32)
    output_projection_np /= np.linalg.norm(output_projection_np, axis=0, keepdims=True)
    INJ_SCALE = 3.0

    input_projection = torch.from_numpy(input_projection_np * INJ_SCALE).to(DEVICE)  # (IO, H)
    output_projection = torch.from_numpy(output_projection_np * INJ_SCALE).to(DEVICE)  # (H, IO)

    # Sparse mask on CPU (for bookkeeping), dense W on GPU (for compute)
    mask = np.zeros((H, H), dtype=np.float32)
    W = torch.zeros(H, H, device=DEVICE)

    theta = torch.full((H,), 0.1, device=DEVICE)
    decay = torch.full((H,), 0.15, device=DEVICE)

    alive_set = set()
    n_edges = 0

    print(f"Starting from empty network, theta={theta.mean().item():.3f}, decay={decay.mean().item():.3f}")
    sys.stdout.flush()

    # Logging
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG = os.path.join(BASE_DIR, "english_4096n_live.txt")
    JSON_LOG = os.path.join(BASE_DIR, "training_live_data.json")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    with open(LOG, "a") as f:
        f.write(f"\n--- START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"{H}n, {N_PROPOSALS} proposals, {N_TRAIN_SEQS}x{SEQ_LEN}b, GPU dense, budget={BUDGET}\n")

    # Round-robin schedule
    SCHEDULE = ['add', 'add', 'theta', 'add', 'add', 'decay']
    add_accepts = 0; theta_accepts = 0; decay_accepts = 0
    total_accepts = 0
    log_data = []
    t0 = time.time()

    step_t0 = time.time()
    for step in range(1, BUDGET + 1):
        ptype = SCHEDULE[(step - 1) % len(SCHEDULE)]

        # Sample random training sequences
        train_seqs = []
        for _ in range(N_TRAIN_SEQS):
            off = random.randint(0, DATA_LEN - SEQ_LEN)
            train_seqs.append(ALL_DATA_TORCH[off:off + SEQ_LEN])

        # Compute old score ONCE (shared across all proposals)
        with torch.no_grad():
            old_score = forward_seq_batch(W, input_projection, output_projection, theta, decay, bp_torch, train_seqs, TICKS)

        # Try N_PROPOSALS mutations, pick best
        best_delta = -1e9
        best_proposal = None

        for p in range(N_PROPOSALS):
            seed = step * 1000 + p
            rng = random.Random(seed)

            if ptype == 'add':
                r = rng.randint(0, H - 1)
                c = rng.randint(0, H - 1)
                if r == c or (r, c) in alive_set:
                    continue
                val = DRIVE if rng.random() < 0.5 else -DRIVE
                # Apply
                W[r, c] = val
                with torch.no_grad():
                    new_score = forward_seq_batch(W, input_projection, output_projection, theta, decay, bp_torch, train_seqs, TICKS)
                delta = new_score - old_score
                # Undo
                W[r, c] = 0.0
                if delta > best_delta:
                    best_delta = delta
                    best_proposal = ('add', r, c, val)

            elif ptype == 'theta':
                idx = rng.randint(0, H - 1)
                old_val = theta[idx].item()
                new_val = rng.random()
                theta[idx] = new_val
                with torch.no_grad():
                    new_score = forward_seq_batch(W, input_projection, output_projection, theta, decay, bp_torch, train_seqs, TICKS)
                delta = new_score - old_score
                theta[idx] = old_val
                if delta > best_delta:
                    best_delta = delta
                    best_proposal = ('theta', idx, new_val)

            elif ptype == 'decay':
                idx = rng.randint(0, H - 1)
                old_val = decay[idx].item()
                new_val = rng.uniform(0.01, 0.5)
                decay[idx] = new_val
                with torch.no_grad():
                    new_score = forward_seq_batch(W, input_projection, output_projection, theta, decay, bp_torch, train_seqs, TICKS)
                delta = new_score - old_score
                decay[idx] = old_val
                if delta > best_delta:
                    best_delta = delta
                    best_proposal = ('decay', idx, new_val)

        # Accept best if positive
        if best_delta > 0 and best_proposal is not None:
            op = best_proposal[0]
            if op == 'add':
                _, r, c, val = best_proposal
                W[r, c] = val
                mask[r, c] = val
                alive_set.add((r, c))
                n_edges += 1
                add_accepts += 1
            elif op == 'theta':
                _, idx, new_val = best_proposal
                theta[idx] = new_val
                theta_accepts += 1
            elif op == 'decay':
                _, idx, new_val = best_proposal
                decay[idx] = new_val
                decay_accepts += 1
            total_accepts += 1

        # Step timing (first 10 steps + every 100th)
        if step <= 10 or step % 100 == 0:
            step_elapsed = time.time() - step_t0
            print(f"  step {step}: {ptype} {'OK' if best_delta > 0 else '--'} dt={best_delta:.4f} {step_elapsed:.1f}s/step")
            sys.stdout.flush()
        step_t0 = time.time()

        # Logging every 50 steps
        if step % 50 == 0:
            elapsed = time.time() - t0
            with torch.no_grad():
                ea = eval_accuracy_only(W, input_projection, output_projection, theta, decay, bp_torch, eval_seqs, TICKS)
            th_mean = theta.mean().item()
            th_std = theta.std().item()
            dc_mean = decay.mean().item()
            dc_std = decay.std().item()

            line = (f"[{step:5d}] eval={ea*100:.1f}% "
                    f"edges={n_edges} [A={add_accepts}|T={theta_accepts}|D={decay_accepts}] "
                    f"theta={th_mean:.3f}+/-{th_std:.3f} decay={dc_mean:.3f}+/-{dc_std:.3f} {elapsed:.0f}s")
            print(f"  {line}")
            with open(LOG, "a") as f:
                f.write(line + "\n")

            # JSON for live dashboard
            log_data.append({
                'step': step, 'eval': round(ea * 100, 1),
                'edges': n_edges,
                'A': add_accepts, 'T': theta_accepts, 'D': decay_accepts,
                'theta_m': round(th_mean, 4), 'theta_s': round(th_std, 4),
                'decay_m': round(dc_mean, 4), 'decay_s': round(dc_std, 4),
                'time': int(elapsed)
            })
            with open(JSON_LOG, 'w') as f:
                json.dump(log_data, f, separators=(',', ':'))

            sys.stdout.flush()

        # Checkpoint every 500 steps
        if step % 500 == 0:
            ckpt = os.path.join(CKPT_DIR, f"english_4096n_step{step}.npz")
            rows, cols = np.where(mask != 0)
            vals = mask[rows, cols]
            np.savez_compressed(ckpt,
                V=IO, rows=rows.astype(np.int32), cols=cols.astype(np.int32),
                vals=vals, theta=theta.cpu().numpy(), decay=decay.cpu().numpy())
            print(f"  SAVED: {ckpt}")
            sys.stdout.flush()

    # Final save
    elapsed = time.time() - t0
    final_ckpt = os.path.join(CKPT_DIR, "english_4096n_final.npz")
    rows, cols = np.where(mask != 0)
    vals = mask[rows, cols]
    np.savez_compressed(final_ckpt,
        V=IO, rows=rows.astype(np.int32), cols=cols.astype(np.int32),
        vals=vals, theta=theta.cpu().numpy(), decay=decay.cpu().numpy())

    with torch.no_grad():
        final_ea = eval_accuracy_only(W, input_projection, output_projection, theta, decay, bp_torch, eval_seqs, TICKS)
    print(f"\nFINAL: eval={final_ea*100:.1f}% edges={n_edges} "
          f"accepts={total_accepts} {elapsed:.0f}s")
