"""
Compile validation on simulated sequential data at the nightly production length.

T=256 uses the auto policy's chunk-compile path rather than full-model compile:
  - eager baseline
  - auto/chunk compile with C=32

Checks that chunk compile no longer hangs on Dynamo trace and still learns.
"""
import gc
import sys
import time

import torch
import torch._dynamo
import torch.nn as nn

sys.path.insert(0, r"S:\AI\_tmp\nightly_worktree\v4")
from model.instnct import INSTNCT

device = 'cuda'
B, T = 64, 256
STEPS = 100
COMPILE_CHUNK_SIZE = 32

torch.set_float32_matmul_precision('high')

# Generate repeating pattern data (learnable, not pure random).
torch.manual_seed(42)
pattern = torch.randint(0, 256, (4096,))
corpus = pattern.repeat(100)


def run(label, use_compile):
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    m = INSTNCT(
        M=128,
        hidden_dim=4096,
        slot_dim=128,
        N=1,
        R=1,
        embed_mode=True,
        kernel_mode='vshape',
        pointer_mode='sequential',
        write_mode='replace',
        embed_encoding='bitlift',
        output_encoding='lowrank_c19',
    ).to(device)
    m.train()
    m._diag_enabled = False

    if use_compile:
        m._compile_mode = 'chunk'
        m._compile_chunks = True
        m.compile_chunk_size = COMPILE_CHUNK_SIZE
        m._disable_proxy_overlay_for_compile = (getattr(m, 'replace_impl', 'dense') == 'proxy_overlay')
        print(f'  {label} using auto/chunk compile (C={COMPILE_CHUNK_SIZE})', flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    state = None
    offset = 0

    print(f'  {label} warmup...', end='', flush=True)
    tw = time.perf_counter()
    for i in range(5):
        chunk = corpus[offset:offset + T + 1].to(device)
        x = chunk[:-1].unsqueeze(0).expand(B, -1)
        tgt = chunk[1:].unsqueeze(0).expand(B, -1)
        with torch.amp.autocast('cuda'):
            o, state = m(x, state=state)
            l = crit(o.reshape(-1, o.size(-1)), tgt.reshape(-1))
        opt.zero_grad()
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        offset = (offset + T) % (len(corpus) - T - 1)
        print(f' {i}', end='', flush=True)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - tw
    print(f' done ({warmup_s:.1f}s)', flush=True)

    losses = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(STEPS):
        chunk = corpus[offset:offset + T + 1].to(device)
        x = chunk[:-1].unsqueeze(0).expand(B, -1)
        tgt = chunk[1:].unsqueeze(0).expand(B, -1)
        with torch.amp.autocast('cuda'):
            o, state = m(x, state=state)
            l = crit(o.reshape(-1, o.size(-1)), tgt.reshape(-1))
        opt.zero_grad()
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        offset = (offset + T) % (len(corpus) - T - 1)
        losses.append(l.item())
        if step % 20 == 0:
            print(f'  {label} step {step:3d} loss={l.item():.4f}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / STEPS * 1000
    vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(
        f'  {label} DONE: {ms:.1f} ms/step, final_loss={losses[-1]:.4f}, '
        f'best={min(losses):.4f}, warmup={warmup_s:.1f}s, vram={vram:.0f} MB',
        flush=True,
    )
    return {'ms': ms, 'losses': losses, 'warmup': warmup_s, 'vram': vram}


print("=" * 60, flush=True)
print("COMPILE VALIDATION — Sequential data, T=256 (auto -> chunk)", flush=True)
print("=" * 60, flush=True)

eager = run('eager', False)
compiled = run('compile-auto', True)

print(f'\n{"=" * 60}', flush=True)
print(f'eager:        {eager["ms"]:.1f} ms/step  final_loss={eager["losses"][-1]:.4f}  warmup={eager["warmup"]:.1f}s', flush=True)
print(f'compile-auto: {compiled["ms"]:.1f} ms/step  final_loss={compiled["losses"][-1]:.4f}  warmup={compiled["warmup"]:.1f}s', flush=True)
print(f'speedup: {(eager["ms"] - compiled["ms"]) / eager["ms"] * 100:+.1f}%', flush=True)
print(f'loss delta (final): {abs(eager["losses"][-1] - compiled["losses"][-1]):.4f}', flush=True)
print(f'best delta: {abs(min(eager["losses"]) - min(compiled["losses"])):.4f}', flush=True)
print("=" * 60, flush=True)
