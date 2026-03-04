"""A/B test: baseline vs circuit fixes (ring_decay, gate_bias, expert_output_weights, ptr_gradient).

Runs 2 short training loops on synthetic echo data with same seed.
Compares loss, accuracy, and ring diagnostics.
"""

import sys, time, torch, torch.nn as nn, numpy as np
sys.path.insert(0, 'model')
from instnct import INSTNCT

# ── Config ──
STEPS = 300
SEQ = 64
BATCH = 16
LR = 1e-3
SEED = 1337
DEVICE = 'cpu'

# Smaller model for CPU speed
MODEL_BASE = dict(
    M=128, hidden_dim=128, slot_dim=32, N=2, R=1,
    embed_mode=True, embed_encoding='bitlift', output_encoding='lowrank_c19',
    pointer_mode='pilot', kernel_mode='vshape', write_mode='replace',
)


def make_echo_batch(batch, seq_len, block=8, repeat=4, rng=None):
    """Generate echo task: random block repeated N times.
    Returns (x, mask) where mask marks predictable positions."""
    if rng is None:
        rng = np.random.default_rng()
    data = np.zeros((batch, seq_len), dtype=np.uint8)
    mask = np.zeros((batch, seq_len), dtype=np.float32)
    for b in range(batch):
        pos = 0
        while pos < seq_len:
            blk = rng.integers(0, 256, size=block, dtype=np.uint8)
            for r in range(repeat):
                end = min(pos + block, seq_len)
                data[b, pos:end] = blk[:end - pos]
                if r > 0:  # repeats are predictable
                    mask[b, pos:end] = 1.0
                pos = end
    x = torch.from_numpy(data.astype(np.int64))
    m = torch.from_numpy(mask) > 0.5
    return x, m


def train_run(tag, extra_kwargs, model_override=None):
    """Train for STEPS and return metrics history."""
    torch.manual_seed(SEED)
    base = model_override if model_override is not None else MODEL_BASE
    model = INSTNCT(**base, **extra_kwargs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {tag}  ({n_params:,} params)")
    print(f"{'='*60}")

    history = {'loss': [], 'acc': [], 'ring_norm': []}
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    for step in range(1, STEPS + 1):
        x, mask = make_echo_batch(BATCH, SEQ, rng=rng)
        x = x.to(DEVICE)
        mask = mask.to(DEVICE)

        out, _ = model(x)
        logits = out[:, :-1].reshape(-1, 256)
        targets = x[:, 1:].reshape(-1)
        m = mask[:, 1:].reshape(-1)

        if m.any():
            loss = criterion(logits[m], targets[m])
        else:
            loss = criterion(logits, targets)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            if m.any():
                acc = (preds[m] == targets[m]).float().mean().item()
            else:
                acc = (preds == targets).float().mean().item()

        history['loss'].append(loss.item())
        history['acc'].append(acc)
        rn = model._diag.get('ring_norm', 0.0)
        history['ring_norm'].append(rn)

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            # extra diag
            alpha_info = ""
            a0 = model._diag.get('alpha_0_mean', None)
            if a0 is not None:
                a1 = model._diag.get('alpha_1_mean', 0)
                alpha_info = f"  α=[{a0:.2f},{a1:.2f}]"
            print(f"  step {step:4d}  loss={loss.item():.4f}  acc={acc:.3f}  "
                  f"ring={rn:.1f}{alpha_info}  [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({STEPS/elapsed:.1f} steps/s)")
    return history


def main():
    # ── Run A: Baseline (all fixes off) ──
    hist_base = train_run("BASELINE (no fixes)", {})

    # ── Run B: Gate bias + expert weights + ptr gradient (no decay) ──
    hist_3fix = train_run("3 FIXES (no decay)", {
        'gate_bias': True,
        'expert_output_weights': True,
        'ptr_gradient': True,
    })

    # ── Run C: All 4 fixes with additive write (decay makes sense here) ──
    # Override write_mode in MODEL_BASE for this run
    additive_base = {**MODEL_BASE, 'write_mode': 'additive'}
    hist_additive = train_run("ALL 4 + additive write", {
        'ring_decay': True,
        'gate_bias': True,
        'expert_output_weights': True,
        'ptr_gradient': True,
    }, model_override=additive_base)

    # ── Compare ──
    def avg_last(lst, n=50):
        return sum(lst[-n:]) / min(len(lst), n)

    configs = [
        ("Baseline", hist_base),
        ("3 Fixes", hist_3fix),
        ("All4+additive", hist_additive),
    ]

    print(f"\n{'='*70}")
    print(f"  RESULTS (last 50 steps average)")
    print(f"{'='*70}")
    header = f"  {'Metric':<20}" + "".join(f" {name:>14}" for name, _ in configs)
    print(header)
    print(f"  {'-'*66}")

    for metric, key, fmt, better in [
        ('Loss', 'loss', '.4f', 'lower'),
        ('Accuracy', 'acc', '.1%', 'higher'),
        ('Ring norm', 'ring_norm', '.1f', 'lower'),
    ]:
        vals = [avg_last(h[key]) for _, h in configs]
        row = f"  {metric:<20}" + "".join(f" {v:>14{fmt}}" for v in vals)
        print(row)

    # convergence speed
    def steps_to_thresh(accs, thresh):
        for i, a in enumerate(accs):
            if a >= thresh:
                return i + 1
        return len(accs)

    for thresh in [0.05, 0.10]:
        vals = [steps_to_thresh(h['acc'], thresh) for _, h in configs]
        strs = [str(v) if v < STEPS else f'>{STEPS}' for v in vals]
        label = f'Steps to {thresh:.0%} acc'
        row = f"  {label:<20}" + "".join(f" {s:>14}" for s in strs)
        print(row)

    print()


if __name__ == '__main__':
    main()
