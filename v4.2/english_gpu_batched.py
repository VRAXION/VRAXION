"""
English GPU training — FULLY BATCHED
=====================================
Key insight: batch ALL proposals × ALL sequences into one giant forward pass.
For add proposals, base W is the same — only 1 element differs per proposal.
So: compute base forward once, then add the DELTA from 1 extra edge.

Architecture:
- old_score: 1 forward pass with batch=N_TRAIN_SEQS
- For add proposals: perturbed forward = base + rank-1 update (1 edge)
  Instead of running full forward per proposal, we batch ALL proposals'
  sequences into one forward call with the base W, then correct.

Actually simplest correct approach:
- Stack all proposals into batch dimension: (N_PROPOSALS * N_TRAIN_SEQS, H)
- Each proposal-group shares the same W except 1 element
- But W is different per proposal... so we need per-proposal W.
- At N=3072, N_PROPOSALS=32: 32 copies of W = 32 * 3072^2 * 4 = 1.2GB. Fits!

CLEANEST APPROACH: Broadcast trick.
- Base forward with original W: batch = N_TRAIN_SEQS
- For each proposal, the delta is just the effect of 1 extra edge.
  The edge (r,c,val) adds val * act[c] to charge[r] every tick.
  We can't easily compute this delta without running the full forward...

SIMPLEST FAST APPROACH:
- Use CUDA graph or torch.compile to eliminate Python loop overhead
- Run proposals sequentially but with compiled inner loop
"""
import sys, os, time, random, json
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


@torch.no_grad()
def forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, seqs, ticks=6):
    """Batched forward: seqs is (B, seq_len) long tensor. Returns scalar score."""
    B, L = seqs.shape
    H = W.shape[0]
    ret = 1.0 - decay
    state = torch.zeros(B, H, device=DEVICE)
    charge = torch.zeros(B, H, device=DEVICE)
    correct = 0
    prob_sum = 0.0
    n = 0
    for i in range(L - 1):
        act = state
        for t in range(ticks):
            if t == 0:
                act = act + bp_torch[seqs[:, i]] @ input_projection
            raw = act @ W.T
            charge = charge + raw
            charge = charge * ret
            act = torch.clamp(charge - theta, min=0.0)
            charge = torch.clamp(charge, -1.0, 1.0)
        state = act
        out = charge @ output_projection
        out_n = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        sims = out_n @ bp_norm.T
        targets = seqs[:, i + 1]
        correct += (sims.argmax(dim=1) == targets).sum().item()
        probs = torch.softmax(sims, dim=1)
        prob_sum += probs[torch.arange(B, device=DEVICE), targets].sum().item()
        n += B
    acc = correct / n if n else 0
    avg_p = prob_sum / n if n else 0
    return 0.5 * acc + 0.5 * avg_p


@torch.no_grad()
def eval_acc(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, seqs, ticks=6):
    """Just accuracy for reporting."""
    B, L = seqs.shape
    H = W.shape[0]
    ret = 1.0 - decay
    state = torch.zeros(B, H, device=DEVICE)
    charge = torch.zeros(B, H, device=DEVICE)
    correct = 0; total = 0
    for i in range(L - 1):
        act = state
        for t in range(ticks):
            if t == 0:
                act = act + bp_torch[seqs[:, i]] @ input_projection
            raw = act @ W.T
            charge = charge + raw
            charge = charge * ret
            act = torch.clamp(charge - theta, min=0.0)
            charge = torch.clamp(charge, -1.0, 1.0)
        state = act
        out = charge @ output_projection
        out_n = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        sims = out_n @ bp_norm.T
        correct += (sims.argmax(dim=1) == seqs[:, i + 1]).sum().item()
        total += B
    return correct / total if total else 0


if __name__ == "__main__":
    IO = 256
    H = IO * 4   # 1024 neurons (N=3072)
    N_PROPOSALS = 32
    BUDGET = 20000
    SEQ_LEN = 64    # shorter sequences = faster iteration (was 200)
    N_TRAIN_SEQS = 32  # more seqs to compensate shorter length
    N_EVAL_SEQS = 32
    TICKS = 4        # fewer ticks = faster (benchmark showed ticks=4 >= ticks=6)
    DRIVE = 0.6

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    bp = make_bp(IO)
    bp_torch = torch.from_numpy(bp).to(DEVICE)
    bp_norm = bp_torch / (bp_torch.norm(dim=1, keepdim=True) + 1e-8)

    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    DATA_LEN = len(ALL_DATA)
    ALL_DATA_T = torch.from_numpy(ALL_DATA.copy()).long().to(DEVICE)
    print(f"Loaded {DATA_LEN / 1e6:.1f} MB text")

    # Fixed eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_offs = [eval_rng.randint(0, DATA_LEN - SEQ_LEN) for _ in range(N_EVAL_SEQS)]
    eval_seqs = torch.stack([ALL_DATA_T[o:o + SEQ_LEN] for o in eval_offs])  # (N_EVAL, SEQ_LEN)

    print(f"{H} neurons (N={H}), I/O={IO}, {N_PROPOSALS} proposals/step")
    print(f"Train: {N_TRAIN_SEQS}x{SEQ_LEN}b | Eval: {N_EVAL_SEQS}x{SEQ_LEN}b | Ticks: {TICKS}")
    print(f"Dense W: {H*H*4/1e6:.1f} MB")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Init
    proj_rng = np.random.RandomState(42)
    input_projection_np = proj_rng.randn(IO, H).astype(np.float32)
    input_projection_np /= np.linalg.norm(input_projection_np, axis=0, keepdims=True)
    output_projection_np = proj_rng.randn(H, IO).astype(np.float32)
    output_projection_np /= np.linalg.norm(output_projection_np, axis=0, keepdims=True)
    INJ_SCALE = 3.0

    input_projection = torch.from_numpy(input_projection_np * INJ_SCALE).to(DEVICE)
    output_projection = torch.from_numpy(output_projection_np * INJ_SCALE).to(DEVICE)
    W = torch.zeros(H, H, device=DEVICE)
    theta = torch.full((H,), 0.1, device=DEVICE)
    decay = torch.full((H,), 0.15, device=DEVICE)

    alive_set = set()
    n_edges = 0

    # Logging
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG = os.path.join(BASE_DIR, "english_gpu_live.txt")
    JSON_LOG = os.path.join(BASE_DIR, "training_live_data.json")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    with open(LOG, "w") as f:
        f.write(f"--- START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"{H}n, {N_PROPOSALS}p, {N_TRAIN_SEQS}x{SEQ_LEN}b, ticks={TICKS}, GPU\n")

    SCHEDULE = ['add', 'add', 'theta', 'add', 'add', 'decay']
    add_accepts = 0; theta_accepts = 0; decay_accepts = 0
    total_accepts = 0; log_data = []
    t0 = time.time()

    print(f"Starting training...")
    sys.stdout.flush()

    for step in range(1, BUDGET + 1):
        step_t = time.time()
        ptype = SCHEDULE[(step - 1) % len(SCHEDULE)]

        # Sample random training seqs: (N_TRAIN_SEQS, SEQ_LEN)
        offs = [random.randint(0, DATA_LEN - SEQ_LEN) for _ in range(N_TRAIN_SEQS)]
        train_seqs = torch.stack([ALL_DATA_T[o:o + SEQ_LEN] for o in offs])

        # Old score (once)
        old_score = forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, train_seqs, TICKS)

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
                W[r, c] = val
                new_score = forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, train_seqs, TICKS)
                W[r, c] = 0.0
                delta = new_score - old_score
                if delta > best_delta:
                    best_delta = delta
                    best_proposal = ('add', r, c, val)

            elif ptype == 'theta':
                idx = rng.randint(0, H - 1)
                old_val = theta[idx].item()
                new_val = rng.random()
                theta[idx] = new_val
                new_score = forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, train_seqs, TICKS)
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
                new_score = forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, train_seqs, TICKS)
                delta = new_score - old_score
                decay[idx] = old_val
                if delta > best_delta:
                    best_delta = delta
                    best_proposal = ('decay', idx, new_val)

        # Accept best
        if best_delta > 0 and best_proposal is not None:
            op = best_proposal[0]
            if op == 'add':
                _, r, c, val = best_proposal
                W[r, c] = val
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

        step_dt = time.time() - step_t

        # Progress: first 5 steps + every 10th
        if step <= 5 or step % 10 == 0:
            print(f"  step {step}: {ptype} {'OK' if best_delta > 0 else '--'} "
                  f"dt={best_delta:.4f} {step_dt:.1f}s/step edges={n_edges}")
            sys.stdout.flush()

        # Full log every 50 steps
        if step % 50 == 0:
            elapsed = time.time() - t0
            ea = eval_acc(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, eval_seqs, TICKS)
            th_m = theta.mean().item(); th_s = theta.std().item()
            dc_m = decay.mean().item(); dc_s = decay.std().item()
            sps = step / elapsed

            line = (f"[{step:5d}] eval={ea*100:.1f}% edges={n_edges} "
                    f"[A={add_accepts}|T={theta_accepts}|D={decay_accepts}] "
                    f"theta={th_m:.3f}+/-{th_s:.3f} decay={dc_m:.3f}+/-{dc_s:.3f} "
                    f"{elapsed:.0f}s ({sps:.2f} step/s)")
            print(f"  {line}")
            with open(LOG, "a") as f:
                f.write(line + "\n")

            log_data.append({
                'step': step, 'eval': round(ea * 100, 1), 'edges': n_edges,
                'A': add_accepts, 'T': theta_accepts, 'D': decay_accepts,
                'theta_m': round(th_m, 4), 'theta_s': round(th_s, 4),
                'decay_m': round(dc_m, 4), 'decay_s': round(dc_s, 4),
                'time': int(elapsed)
            })
            with open(JSON_LOG, 'w') as f:
                json.dump(log_data, f, separators=(',', ':'))
            sys.stdout.flush()

        if step % 500 == 0:
            ckpt = os.path.join(CKPT_DIR, f"english_gpu_{H}n_step{step}.npz")
            mask_np = W.cpu().numpy()
            rows, cols = np.where(mask_np != 0)
            vals = mask_np[rows, cols]
            np.savez_compressed(ckpt, V=IO,
                rows=rows.astype(np.int32), cols=cols.astype(np.int32), vals=vals,
                theta=theta.cpu().numpy(), decay=decay.cpu().numpy())
            print(f"  SAVED: {ckpt}")
            sys.stdout.flush()

    elapsed = time.time() - t0
    ea = eval_acc(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, eval_seqs, TICKS)
    print(f"\nFINAL: eval={ea*100:.1f}% edges={n_edges} accepts={total_accepts} {elapsed:.0f}s")
