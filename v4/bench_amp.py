"""
AMP (Automatic Mixed Precision) A/B Benchmark
Tests fp32 vs fp16 autocast on INSTNCT sequential pointer.

3 configs:
  A: fp32 baseline (TF32 enabled)
  B: AMP fp16 (autocast forward + backward, GradScaler)
  C: AMP bf16 (if supported — RTX 40xx has bf16)
"""
import sys, torch, time, torch.nn as nn, gc
sys.path.insert(0, r"S:\AI\_tmp\nightly_worktree\v4")
from model.instnct import INSTNCT

device = 'cuda'
B, T, WARMUP, MEASURE = 64, 48, 5, 30
torch.set_float32_matmul_precision('high')  # TF32 baseline

print("=" * 60, flush=True)
print("AMP A/B BENCHMARK — INSTNCT sequential pointer", flush=True)
print(f"B={B}, T={T}, warmup={WARMUP}, measure={MEASURE}", flush=True)
print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
print(f"bf16 support: {torch.cuda.is_bf16_supported()}", flush=True)
print("=" * 60, flush=True)


def bench(label, amp_dtype=None):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42); torch.cuda.manual_seed(42)

    m = INSTNCT(M=128, hidden_dim=4096, slot_dim=128, N=1, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='sequential',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19').to(device)
    m.train(); m._diag_enabled = False

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype is not None))
    state = None

    use_amp = amp_dtype is not None

    # Warmup
    print(f'  {label} warmup...', end='', flush=True)
    for i in range(WARMUP):
        x = torch.randint(0, 256, (B, T), device=device)
        t = torch.randint(0, 256, (B, T), device=device)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            o, state = m(x, state=state)
            l = crit(o.reshape(-1, o.size(-1)), t.reshape(-1))
        opt.zero_grad()
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        print(f' {i}', end='', flush=True)
    torch.cuda.synchronize()
    print(' done', flush=True)

    # Measure
    losses = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(MEASURE):
        x = torch.randint(0, 256, (B, T), device=device)
        t = torch.randint(0, 256, (B, T), device=device)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            o, state = m(x, state=state)
            l = crit(o.reshape(-1, o.size(-1)), t.reshape(-1))
        opt.zero_grad()
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        losses.append(l.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms = elapsed / MEASURE * 1000
    vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    has_nan = any(v != v for v in losses)

    print(f'  {label}: {ms:.1f} ms/step | VRAM: {vram:.0f} MB | '
          f'loss={losses[-1]:.4f} | NaN: {has_nan}', flush=True)
    del m, opt, scaler
    return {'ms': ms, 'vram': vram, 'loss': losses[-1], 'nan': has_nan}


# --- Run configs ---
print('\n--- A: fp32 baseline (TF32) ---', flush=True)
r_fp32 = bench('fp32', amp_dtype=None)

print('\n--- B: AMP fp16 ---', flush=True)
r_fp16 = bench('AMP fp16', amp_dtype=torch.float16)

if torch.cuda.is_bf16_supported():
    print('\n--- C: AMP bf16 ---', flush=True)
    r_bf16 = bench('AMP bf16', amp_dtype=torch.bfloat16)
else:
    r_bf16 = None
    print('\n--- C: bf16 NOT SUPPORTED, skipping ---', flush=True)

# --- Summary ---
print(f'\n{"=" * 60}', flush=True)
print('SUMMARY', flush=True)
print(f'{"=" * 60}', flush=True)
print(f'  fp32:     {r_fp32["ms"]:6.1f} ms/step  VRAM: {r_fp32["vram"]:5.0f} MB  loss={r_fp32["loss"]:.4f}  NaN={r_fp32["nan"]}', flush=True)
print(f'  AMP fp16: {r_fp16["ms"]:6.1f} ms/step  VRAM: {r_fp16["vram"]:5.0f} MB  loss={r_fp16["loss"]:.4f}  NaN={r_fp16["nan"]}', flush=True)
if r_bf16:
    print(f'  AMP bf16: {r_bf16["ms"]:6.1f} ms/step  VRAM: {r_bf16["vram"]:5.0f} MB  loss={r_bf16["loss"]:.4f}  NaN={r_bf16["nan"]}', flush=True)

print(f'\n  fp16 speedup: {(r_fp32["ms"]-r_fp16["ms"])/r_fp32["ms"]*100:+.1f}%', flush=True)
print(f'  fp16 VRAM save: {(r_fp32["vram"]-r_fp16["vram"])/r_fp32["vram"]*100:+.1f}%', flush=True)
if r_bf16:
    print(f'  bf16 speedup: {(r_fp32["ms"]-r_bf16["ms"])/r_fp32["ms"]*100:+.1f}%', flush=True)
    print(f'  bf16 VRAM save: {(r_fp32["vram"]-r_bf16["vram"])/r_fp32["vram"]*100:+.1f}%', flush=True)
print(f'{"=" * 60}', flush=True)
