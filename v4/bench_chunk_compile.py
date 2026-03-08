"""
Chunk-compile benchmark: eager vs chunk-compiled at T=256

Tests the new chunk-level torch.compile approach where _process_chunk
is compiled with C=32 and called 8 times for T=256.

Configs:
  A: eager T=256 (+ AMP, baseline)
  B: chunk-compiled T=256 C=32 (+ AMP)
  C: eager T=48 (reference, small T)
"""
import sys, torch, torch._dynamo, time, torch.nn as nn, gc
sys.path.insert(0, r"S:\AI\_tmp\nightly_worktree\v4")
from model.instnct import INSTNCT

device = 'cuda'
WARMUP, MEASURE = 3, 10
torch.set_float32_matmul_precision('high')

print("=" * 60, flush=True)
print("CHUNK-COMPILE BENCHMARK", flush=True)
print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
print(f"warmup={WARMUP}, measure={MEASURE}", flush=True)
print("=" * 60, flush=True)


def make_model():
    m = INSTNCT(M=128, hidden_dim=4096, slot_dim=128, N=1, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='sequential',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19').to(device)
    m.train()
    m._diag_enabled = False
    return m


def bench(label, B, T, chunk_compile=False, chunk_size=32):
    gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42); torch.cuda.manual_seed(42)

    m = make_model()
    if chunk_compile:
        m._compile_chunks = True
        m.compile_chunk_size = chunk_size
        print(f'  {label}: chunk compile enabled (C={chunk_size})', flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    state = None

    # Warmup
    print(f'  {label} warmup ({WARMUP} steps)...', end='', flush=True)
    tw = time.perf_counter()
    for i in range(WARMUP):
        x = torch.randint(0, 256, (B, T), device=device)
        t = torch.randint(0, 256, (B, T), device=device)
        with torch.amp.autocast('cuda'):
            o, state = m(x, state=state)
            l = crit(o.reshape(-1, o.size(-1)), t.reshape(-1))
        opt.zero_grad()
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        elapsed_w = time.perf_counter() - tw
        print(f' {i}({elapsed_w:.0f}s)', end='', flush=True)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - tw
    print(f' done ({warmup_s:.1f}s)', flush=True)

    # Measure
    losses = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(MEASURE):
        x = torch.randint(0, 256, (B, T), device=device)
        t = torch.randint(0, 256, (B, T), device=device)
        with torch.amp.autocast('cuda'):
            o, state = m(x, state=state)
            l = crit(o.reshape(-1, o.size(-1)), t.reshape(-1))
        opt.zero_grad()
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        losses.append(l.item())
        if i % 3 == 0:
            print(f'  {label} step {i}/{MEASURE} loss={l.item():.4f}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms = elapsed / MEASURE * 1000
    vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    has_nan = any(v != v for v in losses)
    toks = B * T * MEASURE / elapsed

    print(f'  >> {label}: {ms:.1f} ms/step | {toks:.0f} tok/s | VRAM: {vram:.0f} MB | '
          f'warmup: {warmup_s:.1f}s | loss={losses[-1]:.4f} | NaN={has_nan}', flush=True)
    del m, opt, scaler
    return {'ms': ms, 'vram': vram, 'loss': losses[-1], 'nan': has_nan,
            'warmup': warmup_s, 'toks': toks}


# --- Run ---
print('\n--- A: eager T=256 (+ AMP) ---', flush=True)
rA = bench('eager T=256', B=64, T=256)

print('\n--- B: chunk-compile T=256 C=32 (+ AMP) ---', flush=True)
rB = bench('chunk-compile T=256', B=64, T=256, chunk_compile=True, chunk_size=32)

print('\n--- C: eager T=48 (reference) ---', flush=True)
rC = bench('eager T=48', B=64, T=48)

# --- Summary ---
print(f'\n{"=" * 60}', flush=True)
print('SUMMARY (all with AMP fp16 + TF32)', flush=True)
print(f'{"=" * 60}', flush=True)
for name, r in [('A: eager T=256', rA), ('B: chunk-compile T=256', rB), ('C: eager T=48', rC)]:
    print(f'  {name:<30s} {r["ms"]:7.1f} ms/step  {r["toks"]:8.0f} tok/s  '
          f'VRAM: {r["vram"]:5.0f} MB  warmup: {r["warmup"]:.1f}s', flush=True)

if rA['ms'] > 0:
    print(f'\n  chunk-compile speedup (T=256): {(rA["ms"]-rB["ms"])/rA["ms"]*100:+.1f}%', flush=True)
    print(f'  tok/s improvement: {rA["toks"]:.0f} -> {rB["toks"]:.0f}', flush=True)
print(f'  Loss match (eager vs compiled): {abs(rA["loss"]-rB["loss"]):.4f} (should be <0.1)', flush=True)
print(f'  NaN check: {"ALL PASS" if not any(r["nan"] for r in [rA, rB, rC]) else "FAIL"}', flush=True)
print(f'{"=" * 60}', flush=True)
