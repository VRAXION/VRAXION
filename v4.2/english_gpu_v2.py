"""
English GPU training — Single Proposal per Step
================================================
1 proposal per step, 2 forward passes (old vs new), fully batched seqs on GPU.
Round-robin [A,A,T,A,A,D] cycles through mutation types.

Speed target: ~10 step/sec (vs CPU 1.5 step/sec)
Compensation: 6-7x more steps = 6-7x more tries per unit time.
"""
import sys, os, time, random, json
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


@torch.no_grad()
def forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, seqs, ticks):
    """Batched forward. seqs: (B, L) long tensor. Returns scalar score."""
    B, L = seqs.shape
    H = W.shape[0]
    ret = 1.0 - decay
    state = torch.zeros(B, H, device=DEVICE)
    charge = torch.zeros(B, H, device=DEVICE)
    correct = 0; prob_sum = 0.0; n = 0
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
def eval_acc(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, seqs, ticks):
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
    H = IO * 4       # 1024 neurons
    BUDGET = 50000    # more steps since each step is fast
    SEQ_LEN = 64
    N_TRAIN_SEQS = 64  # big batch = stable eval per step
    N_EVAL_SEQS = 64
    TICKS = 4
    DRIVE = 0.6
    LOG_EVERY = 100   # log + eval every N steps

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    bp = make_bp(IO)
    bp_torch = torch.from_numpy(bp).to(DEVICE)
    bp_norm = bp_torch / (bp_torch.norm(dim=1, keepdim=True) + 1e-8)

    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    DATA_LEN = len(ALL_DATA)
    ALL_DATA_T = torch.from_numpy(ALL_DATA.copy()).long().to(DEVICE)
    print(f"Loaded {DATA_LEN / 1e6:.1f} MB text")

    # Fixed eval
    eval_rng = np.random.RandomState(9999)
    eval_offs = [eval_rng.randint(0, DATA_LEN - SEQ_LEN) for _ in range(N_EVAL_SEQS)]
    eval_seqs = torch.stack([ALL_DATA_T[o:o + SEQ_LEN] for o in eval_offs])

    print(f"{H} neurons, I/O={IO}, 1 proposal/step, budget={BUDGET}")
    print(f"Train: {N_TRAIN_SEQS}x{SEQ_LEN}b | Eval: {N_EVAL_SEQS}x{SEQ_LEN}b | Ticks: {TICKS}")
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

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(BASE_DIR, "english_gpu_live.txt")
    JSON_LOG = os.path.join(BASE_DIR, "training_live_data.json")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    with open(LOG_FILE, "w") as f:
        f.write(f"--- START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"{H}n, 1 prop/step, {N_TRAIN_SEQS}x{SEQ_LEN}b, ticks={TICKS}, GPU\n")

    SCHEDULE = ['add', 'add', 'theta', 'add', 'add', 'decay']
    add_accepts = 0; theta_accepts = 0; decay_accepts = 0
    total_accepts = 0; log_data = []
    t0 = time.time()

    # Warmup: 1 forward pass to compile CUDA kernels
    dummy_seqs = torch.randint(0, 256, (N_TRAIN_SEQS, SEQ_LEN), device=DEVICE)
    forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, dummy_seqs, TICKS)
    torch.cuda.synchronize()
    print("CUDA warmup done. Training...")
    sys.stdout.flush()

    for step in range(1, BUDGET + 1):
        ptype = SCHEDULE[(step - 1) % len(SCHEDULE)]

        # Sample train seqs
        offs = torch.randint(0, DATA_LEN - SEQ_LEN, (N_TRAIN_SEQS,), device='cpu')
        train_seqs = torch.stack([ALL_DATA_T[o:o + SEQ_LEN] for o in offs])

        # Old score
        old_score = forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, train_seqs, TICKS)

        # Single proposal
        accepted = False
        if ptype == 'add':
            r = random.randint(0, H - 1)
            c = random.randint(0, H - 1)
            if r != c and (r, c) not in alive_set:
                val = DRIVE if random.random() < 0.5 else -DRIVE
                W[r, c] = val
                new_score = forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, train_seqs, TICKS)
                if new_score > old_score:
                    alive_set.add((r, c))
                    n_edges += 1
                    add_accepts += 1
                    accepted = True
                else:
                    W[r, c] = 0.0

        elif ptype == 'theta':
            idx = random.randint(0, H - 1)
            old_val = theta[idx].item()
            theta[idx] = random.random()
            new_score = forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, train_seqs, TICKS)
            if new_score > old_score:
                theta_accepts += 1
                accepted = True
            else:
                theta[idx] = old_val

        elif ptype == 'decay':
            idx = random.randint(0, H - 1)
            old_val = decay[idx].item()
            decay[idx] = random.uniform(0.01, 0.5)
            new_score = forward_score(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, train_seqs, TICKS)
            if new_score > old_score:
                decay_accepts += 1
                accepted = True
            else:
                decay[idx] = old_val

        if accepted:
            total_accepts += 1

        # Progress: first 10 + every 10th
        if step <= 10 or step % 10 == 0:
            dt = time.time() - t0
            sps = step / dt
            print(f"  [{step:5d}] {ptype:5s} {'OK' if accepted else '--'} "
                  f"edges={n_edges} {sps:.1f} step/s")
            sys.stdout.flush()

        # Full eval + log
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            ea = eval_acc(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, eval_seqs, TICKS)
            th_m = theta.mean().item(); th_s = theta.std().item()
            dc_m = decay.mean().item(); dc_s = decay.std().item()
            sps = step / elapsed

            line = (f"[{step:5d}] eval={ea*100:.1f}% edges={n_edges} "
                    f"[A={add_accepts}|T={theta_accepts}|D={decay_accepts}] "
                    f"theta={th_m:.3f}+/-{th_s:.3f} decay={dc_m:.3f}+/-{dc_s:.3f} "
                    f"{elapsed:.0f}s ({sps:.1f} step/s)")
            print(f"  === {line}")
            with open(LOG_FILE, "a") as f:
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

        if step % 1000 == 0:
            ckpt = os.path.join(CKPT_DIR, f"english_gpu_{H}n_step{step}.npz")
            W_np = W.cpu().numpy()
            rows, cols = np.where(W_np != 0)
            vals = W_np[rows, cols]
            np.savez_compressed(ckpt, V=IO,
                rows=rows.astype(np.int32), cols=cols.astype(np.int32), vals=vals,
                theta=theta.cpu().numpy(), decay=decay.cpu().numpy())
            print(f"  SAVED: {ckpt}")
            sys.stdout.flush()

    elapsed = time.time() - t0
    ea = eval_acc(W, input_projection, output_projection, theta, decay, bp_torch, bp_norm, eval_seqs, TICKS)
    print(f"\nFINAL: eval={ea*100:.1f}% edges={n_edges} accepts={total_accepts} "
          f"{elapsed:.0f}s ({BUDGET/elapsed:.1f} step/s)")
