"""
Ring norm growth test — does ring_signal_norm stabilize or explode?

Runs N=1 and N=2 (pilot) for 500 steps each, logs ring_signal_norm
every 10 steps. Random data (not WikiText) — we only care about norm
dynamics, not accuracy.
"""
import sys, time, torch, torch.nn as nn
sys.path.insert(0, r"S:\AI\_tmp\nightly_worktree\v4")
from model.instnct import INSTNCT

device = "cuda" if torch.cuda.is_available() else "cpu"
B, T = 64, 16
MAX_STEPS = 500
LOG_EVERY = 10

CONFIGS = [
    {'name': 'N=1 pilot', 'N': 1, 'pointer_mode': 'pilot'},
    {'name': 'N=2 pilot', 'N': 2, 'pointer_mode': 'pilot'},
    {'name': 'N=2 sequential', 'N': 2, 'pointer_mode': 'sequential'},
]

print(f"Device: {device}")
print(f"B={B}, T={T}, steps={MAX_STEPS}")


def run_config(cfg):
    name = cfg['name']
    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"{'#' * 70}")

    model = INSTNCT(
        M=128, hidden_dim=4096, slot_dim=128,
        N=cfg['N'], R=1,
        embed_mode=True, kernel_mode='vshape',
        pointer_mode=cfg['pointer_mode'],
        write_mode='replace',
        embed_encoding='bitlift',
        output_encoding='lowrank_c19',
    ).to(device)
    model.train()

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}, N={cfg['N']}, pointer={cfg['pointer_mode']}")
    print(f"{'step':>5} {'loss':>8} {'ring_0':>10} {'ring_1':>10} {'blend_0':>10} {'blend_1':>10} "
          f"{'hidden_0':>10} {'hidden_1':>10} {'alpha_0':>8} {'alpha_1':>8}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    t0 = time.perf_counter()

    for step in range(MAX_STEPS):
        x = torch.randint(0, 256, (B, T), device=device)
        target = torch.randint(0, 256, (B, T), device=device)

        is_log = (step % LOG_EVERY == 0) or (step == MAX_STEPS - 1)
        model._diag_enabled = is_log

        result = model(x)
        out = result[0] if isinstance(result, tuple) else result
        if out.dim() == 3:
            loss = criterion(out.reshape(-1, out.size(-1)), target.reshape(-1))
        else:
            loss = criterion(out, target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_log:
            d = model._diag
            N = cfg['N']

            def g(key, idx):
                for suffix in ['_mean', '']:
                    v = d.get(f'{key}_{idx}{suffix}')
                    if v is not None:
                        return v
                return d.get(f'{key}_{idx}', 0.0)

            r0 = g('ring_signal_norm', 0)
            r1 = g('ring_signal_norm', 1) if N > 1 else 0.0
            b0 = g('blended_norm', 0)
            b1 = g('blended_norm', 1) if N > 1 else 0.0
            h0 = g('hidden_norm', 0)
            h1 = g('hidden_norm', 1) if N > 1 else 0.0
            a0 = g('alpha', 0)
            a1 = g('alpha', 1) if N > 1 else 0.0

            na = '--' if N == 1 else ''
            print(f"{step:5d} {loss.item():8.4f} {r0:10.1f} {r1:10.1f}{na} "
                  f"{b0:10.1f} {b1:10.1f}{na} {h0:10.1f} {h1:10.1f}{na} "
                  f"{a0:8.3f} {a1:8.3f}{na}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Done: {elapsed:.1f}s ({MAX_STEPS / elapsed:.1f} step/s)")

    del model, optimizer
    torch.cuda.empty_cache()


print(f"\n{'=' * 70}")
print("RING NORM GROWTH TEST")
print(f"{'=' * 70}")

for cfg in CONFIGS:
    run_config(cfg)

print(f"\n{'=' * 70}")
print("ALL DONE")
print("=" * 70)
