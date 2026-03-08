"""
torch.compile A/B Benchmark — Sequential Pointer (simplified)

4 configs, 50 training steps each:
  A: eager,   diag=off
  B: compile, diag=off
  C: eager,   diag=on
  D: compile, diag=on
"""
import sys, time, gc, torch, torch.nn as nn, torch._dynamo
sys.path.insert(0, r"S:\AI\_tmp\nightly_worktree\v4")
from model.instnct import INSTNCT

device = "cuda"
B, T = 64, 48
WARMUP = 5
MEASURE = 50
SEED = 42

print("=" * 60)
print("COMPILE A/B BENCHMARK")
print("=" * 60)
print(f"B={B}, T={T}, warmup={WARMUP}, measure={MEASURE}")


def build_model():
    return INSTNCT(
        M=128, hidden_dim=4096, slot_dim=128, N=1, R=1,
        embed_mode=True, kernel_mode='vshape',
        pointer_mode='sequential', write_mode='replace',
        embed_encoding='bitlift', output_encoding='lowrank_c19',
    ).to(device)


def set_diag(model, val):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    m._diag_enabled = val


def run_config(name, use_compile, diag_on):
    print(f"\n--- {name} ---", flush=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    model = build_model()
    model.train()

    if use_compile:
        torch._dynamo.reset()
        t_comp = time.perf_counter()
        model = torch.compile(model, mode='reduce-overhead')
        print(f"  compile call: {time.perf_counter() - t_comp:.1f}s", flush=True)

    set_diag(model, diag_on)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    state = None

    # Warmup
    print(f"  warmup ({WARMUP} steps)...", end="", flush=True)
    t_warmup = time.perf_counter()
    for i in range(WARMUP):
        x = torch.randint(0, 256, (B, T), device=device)
        tgt = torch.randint(0, 256, (B, T), device=device)
        out, state = model(x, state=state)
        loss = criterion(out.reshape(-1, out.size(-1)), tgt.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        print(f" {i}", end="", flush=True)
    torch.cuda.synchronize()
    warmup_sec = time.perf_counter() - t_warmup
    print(f" done ({warmup_sec:.1f}s)", flush=True)

    # Measure
    print(f"  measuring ({MEASURE} steps)...", end="", flush=True)
    losses = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(MEASURE):
        x = torch.randint(0, 256, (B, T), device=device)
        tgt = torch.randint(0, 256, (B, T), device=device)
        out, state = model(x, state=state)
        loss = criterion(out.reshape(-1, out.size(-1)), tgt.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        losses.append(loss.item())
        if i % 10 == 0:
            print(f" {i}", end="", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(" done", flush=True)

    ms = elapsed / MEASURE * 1000
    sps = MEASURE / elapsed
    vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    has_nan = any(l != l for l in losses)

    print(f"  >> {ms:.1f} ms/step | {sps:.1f} step/s | VRAM: {vram:.0f} MB | "
          f"warmup: {warmup_sec:.1f}s | NaN: {has_nan} | loss: {losses[-1]:.4f}", flush=True)

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {'ms': ms, 'sps': sps, 'vram': vram, 'warmup': warmup_sec, 'nan': has_nan, 'loss': losses[-1]}


# ── RUN ALL ──
results = {}
for name, comp, diag in [
    ('A: eager diag=off', False, False),
    ('B: compile diag=off', True, False),
    ('C: eager diag=on', False, True),
    ('D: compile diag=on', True, True),
]:
    results[name] = run_config(name, comp, diag)

# ── SUMMARY ──
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
for name, r in results.items():
    print(f"  {name:<25} {r['ms']:6.1f} ms/step  {r['sps']:5.1f} step/s  VRAM: {r['vram']:5.0f} MB  warmup: {r['warmup']:.1f}s")

a = results['A: eager diag=off']
b = results['B: compile diag=off']
c = results['C: eager diag=on']
d = results['D: compile diag=on']
print(f"\n  Speedup compile (diag=off): {(a['ms']-b['ms'])/a['ms']*100:+.1f}%")
print(f"  Speedup compile (diag=on):  {(c['ms']-d['ms'])/c['ms']*100:+.1f}%")
print(f"  VRAM save compile (diag=off): {(a['vram']-b['vram'])/a['vram']*100:+.1f}%")
print(f"  Diag overhead eager: {(c['ms']-a['ms'])/a['ms']*100:+.1f}%")
print(f"\n  NaN check: {'ALL PASS' if all(not r['nan'] for r in results.values()) else 'FAIL'}")
print("=" * 60)
