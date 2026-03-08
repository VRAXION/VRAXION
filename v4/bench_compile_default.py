"""
torch.compile mode='default' benchmark — T=48 & T=128

INSTNCT has a for-t-in-range(T) loop in forward.
Dynamo unrolls this, so T=256 trace takes forever / hangs.
T=128 is the largest that compiles in reasonable time.

Compares:
  A: eager T=128 (+ AMP)
  B: compile mode='default' T=128 (+ AMP)
  C: eager T=48 (reference)
  D: compile mode='default' T=48 (reference)
"""
import sys, torch, torch._dynamo, time, torch.nn as nn, gc
sys.path.insert(0, r"S:\AI\_tmp\nightly_worktree\v4")
from model.instnct import INSTNCT

device = 'cuda'
WARMUP, MEASURE = 3, 15
torch.set_float32_matmul_precision('high')

print("=" * 60, flush=True)
print("COMPILE mode='default' BENCHMARK", flush=True)
print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
print(f"warmup={WARMUP}, measure={MEASURE}", flush=True)
print("=" * 60, flush=True)


def bench(label, B, T, compile_mode=None):
    gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42); torch.cuda.manual_seed(42)

    m = INSTNCT(M=128, hidden_dim=4096, slot_dim=128, N=1, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='sequential',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19').to(device)
    m.train(); m._diag_enabled = False

    if compile_mode:
        print(f'  {label} compiling (mode={compile_mode})...', end='', flush=True)
        tc = time.perf_counter()
        m = torch.compile(m, mode=compile_mode)
        print(f' call done ({time.perf_counter()-tc:.1f}s)', flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    state = None

    # Warmup — prints per-step so we see progress
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
        if i % 5 == 0:
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
results = {}

print('\n--- A: eager T=48 ---', flush=True)
results['eager T=48'] = bench('eager T=48', B=64, T=48, compile_mode=None)

print('\n--- B: compile-default T=48 ---', flush=True)
results['compile T=48'] = bench('compile-default T=48', B=64, T=48, compile_mode='default')

print('\n--- C: eager T=128 ---', flush=True)
results['eager T=128'] = bench('eager T=128', B=64, T=128, compile_mode=None)

print('\n--- D: compile-default T=128 ---', flush=True)
results['compile T=128'] = bench('compile-default T=128', B=64, T=128, compile_mode='default')

# --- Summary ---
print(f'\n{"=" * 60}', flush=True)
print('SUMMARY (all with AMP fp16 + TF32)', flush=True)
print(f'{"=" * 60}', flush=True)
for name, r in results.items():
    print(f'  {name:<25s} {r["ms"]:7.1f} ms/step  {r["toks"]:8.0f} tok/s  '
          f'VRAM: {r["vram"]:5.0f} MB  warmup: {r["warmup"]:.1f}s', flush=True)

eT48 = results['eager T=48']['ms']
cT48 = results['compile T=48']['ms']
eT128 = results['eager T=128']['ms']
cT128 = results['compile T=128']['ms']

print(f'\n  compile speedup T=48:  {(eT48-cT48)/eT48*100:+.1f}%', flush=True)
print(f'  compile speedup T=128: {(eT128-cT128)/eT128*100:+.1f}%', flush=True)
print(f'  NaN check: {"ALL PASS" if not any(r["nan"] for r in results.values()) else "FAIL"}', flush=True)
print(f'{"=" * 60}', flush=True)
