"""
CUDA Graph capture fix benchmark.

The chunk-compile path gets a CUDAGraph warning:
  "Unable to hit fast path of CUDAGraphs because of pending, uninvoked backwards"

torch.compiler.cudagraph_mark_step_begin() tells PyTorch that the previous
step is complete and it's safe to capture a new CUDA graph.

Configs (all chunk-compile C=32, T=256, AMP):
  A: baseline (no mark_step)
  B: with cudagraph_mark_step_begin() before each forward
  C: with torch.cuda.synchronize() before each forward (reference)

Deterministic: fixed seed, same data.
"""
import sys, torch, torch._dynamo, time, torch.nn as nn, gc
sys.path.insert(0, r"S:\AI\_tmp\nightly_worktree\v4")
from model.instnct import INSTNCT

device = 'cuda'
B, T = 64, 256
WARMUP, MEASURE = 3, 15
torch.set_float32_matmul_precision('high')

print("=" * 60, flush=True)
print("CUDA GRAPH FIX BENCHMARK", flush=True)
print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
print(f"B={B}, T={T}, warmup={WARMUP}, measure={MEASURE}", flush=True)
print(f"torch version: {torch.__version__}", flush=True)

# Check if cudagraph_mark_step_begin exists
has_mark_step = hasattr(torch.compiler, 'cudagraph_mark_step_begin')
print(f"cudagraph_mark_step_begin available: {has_mark_step}", flush=True)
print("=" * 60, flush=True)


def make_model():
    m = INSTNCT(M=128, hidden_dim=4096, slot_dim=128, N=1, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='sequential',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19').to(device)
    m.train()
    m._diag_enabled = False
    m._compile_chunks = True
    m._compile_mode = 'chunk'
    m.compile_chunk_size = 32
    return m


def bench(label, pre_forward_fn=None):
    gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42); torch.cuda.manual_seed(42)

    m = make_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    state = None

    # Warmup
    print(f'  {label} warmup ({WARMUP} steps)...', end='', flush=True)
    tw = time.perf_counter()
    for i in range(WARMUP):
        if pre_forward_fn:
            pre_forward_fn()
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
        if pre_forward_fn:
            pre_forward_fn()
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
            print(f'  {label} step {i}/{MEASURE} loss={l.item():.4f} ms={((time.perf_counter()-t0)/(i+1))*1000:.0f}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms = elapsed / MEASURE * 1000
    vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    has_nan = any(v != v for v in losses)

    print(f'  >> {label}: {ms:.1f} ms/step | VRAM: {vram:.0f} MB | '
          f'warmup: {warmup_s:.1f}s | loss={losses[-1]:.4f} | NaN={has_nan}', flush=True)
    del m, opt, scaler
    return {'ms': ms, 'vram': vram, 'loss': losses[-1], 'nan': has_nan, 'warmup': warmup_s}


# --- Run ---
print('\n--- A: chunk-compile baseline (no fix) ---', flush=True)
rA = bench('baseline')

if has_mark_step:
    print('\n--- B: + cudagraph_mark_step_begin() ---', flush=True)
    rB = bench('mark_step', pre_forward_fn=torch.compiler.cudagraph_mark_step_begin)
else:
    print('\n--- B: SKIPPED (cudagraph_mark_step_begin not available) ---', flush=True)
    rB = None

print('\n--- C: + torch.cuda.synchronize() (reference) ---', flush=True)
rC = bench('cuda_sync', pre_forward_fn=torch.cuda.synchronize)

# --- Summary ---
print(f'\n{"=" * 60}', flush=True)
print('SUMMARY (chunk-compile C=32, T=256, AMP fp16 + TF32)', flush=True)
print(f'{"=" * 60}', flush=True)
print(f'  A: baseline         {rA["ms"]:7.1f} ms/step  VRAM: {rA["vram"]:5.0f} MB  loss={rA["loss"]:.4f}', flush=True)
if rB:
    print(f'  B: mark_step        {rB["ms"]:7.1f} ms/step  VRAM: {rB["vram"]:5.0f} MB  loss={rB["loss"]:.4f}', flush=True)
print(f'  C: cuda_sync        {rC["ms"]:7.1f} ms/step  VRAM: {rC["vram"]:5.0f} MB  loss={rC["loss"]:.4f}', flush=True)

if rB:
    print(f'\n  mark_step speedup: {(rA["ms"]-rB["ms"])/rA["ms"]*100:+.1f}%', flush=True)
print(f'  cuda_sync speedup: {(rA["ms"]-rC["ms"])/rA["ms"]*100:+.1f}%', flush=True)
print(f'  NaN: {"ALL PASS" if not any(r["nan"] for r in [rA, rC] + ([rB] if rB else [])) else "FAIL"}', flush=True)
print(f'{"=" * 60}', flush=True)
