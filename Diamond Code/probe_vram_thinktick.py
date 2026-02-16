"""
VRAM probe: measure activation cost of tt=1 vs tt=2 at batch=1.
Isolates the pure think-tick activation delta.
"""
import time
import sys
import os
import torch

STEP_TIMEOUT = 60

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swarm_model import SwarmByteRingModel

CONFIG = dict(
    num_memory_positions=62,
    embedding_dim=6180,
    num_beings=1,
    depth=12,
    num_bits=8,
    combiner_mode='mean',
    think_ticks=1,
    use_lcx=True,
    lcx_mode='hash',
    lcx_num_slots=618,
    lcx_key_dim=618,
    lcx_top_k=6,
    lcx_num_levels=3,
    lcx_level_slots=[618, 6180, 61800],
    attention_radius=2,
)

DEVICE = 'cuda'
BATCH = 1
SEQ_LEN = 62


def detach_lcx(model):
    if not hasattr(model, '_lcx_allocated_levels'):
        return
    for _lvl in range(model._lcx_num_levels):
        if _lvl not in model._lcx_allocated_levels:
            continue
        _k = getattr(model, f'lcx_keys_{_lvl}', None)
        _v = getattr(model, f'lcx_values_{_lvl}', None)
        if _v is not None and _v.requires_grad:
            setattr(model, f'lcx_values_{_lvl}', _v.detach())
        if _k is not None and _k.requires_grad:
            setattr(model, f'lcx_keys_{_lvl}', _k.detach())


def measure(model, optimizer, tt, x, y, label):
    model.think_ticks = tt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    mem_before = torch.cuda.memory_allocated() / 1024**2
    print(f'  [{label}] before={mem_before:.0f} MiB, starting fwd+bwd (tt={tt}, batch={BATCH})...', flush=True)
    t0 = time.time()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output, stats = model(x, return_stats=True, return_being_outputs=True)
        loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    if elapsed > STEP_TIMEOUT:
        print(f'  TIMEOUT: {elapsed:.0f}s', flush=True)
        sys.exit(1)

    peak = torch.cuda.max_memory_allocated() / 1024**2
    after = torch.cuda.memory_allocated() / 1024**2
    alloc = sorted(model._lcx_allocated_levels)

    detach_lcx(model)

    print(f'  [{label}] peak={peak:.0f} MiB  after={after:.0f} MiB  '
          f'time={elapsed:.1f}s  lcx_levels={alloc}', flush=True)
    return peak, after, elapsed


def main():
    print('=' * 60)
    print('VRAM PROBE: Think Tick Cost (batch=1)')
    print('=' * 60)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f'GPU: {torch.cuda.get_device_name()} — {total:.0f} MiB')
    print()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Creating model...', flush=True)
    model = SwarmByteRingModel(**CONFIG).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f'Params: {params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randn(BATCH, SEQ_LEN, CONFIG['num_bits']).to(DEVICE)
    y = torch.randn(BATCH, SEQ_LEN, CONFIG['num_bits']).to(DEVICE)

    # Warmup — initializes optimizer states, lazy-allocs L1
    print('\nWarmup (tt=1)...', flush=True)
    model.think_ticks = 1
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out, _ = model(x, return_stats=True, return_being_outputs=True)
        loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    detach_lcx(model)
    del out, loss
    torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated() / 1024**2
    print(f'Static memory (model+opt+LCX L0+L1): {base_mem:.0f} MiB')
    print(f'LCX levels allocated: {sorted(model._lcx_allocated_levels)}')
    print()

    # --- Measure tt=1 ---
    peak1, _, t1 = measure(model, optimizer, 1, x, y, 'tt=1')

    # --- Measure tt=2 (will lazy-alloc L2) ---
    peak2, _, t2 = measure(model, optimizer, 2, x, y, 'tt=2')

    # Get static mem after L2 allocation
    torch.cuda.empty_cache()
    static_with_l2 = torch.cuda.memory_allocated() / 1024**2

    # --- Measure tt=2 again (L2 already allocated, cleaner measurement) ---
    peak2b, _, t2b = measure(model, optimizer, 2, x, y, 'tt=2 (warm)')

    # --- Measure tt=3 if it fits ---
    peak3 = None
    try:
        peak3, _, t3 = measure(model, optimizer, 3, x, y, 'tt=3')
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print('  [tt=3] *** OOM ***', flush=True)

    # --- Summary ---
    print('\n' + '=' * 60)
    print('RESULTS (batch=1)')
    print('=' * 60)
    print(f'Static (model+opt+L0+L1):        {base_mem:.0f} MiB')
    print(f'Static (model+opt+L0+L1+L2):     {static_with_l2:.0f} MiB')
    print(f'L2 buffer cost:                   {static_with_l2 - base_mem:.0f} MiB')
    print()
    print(f'tt=1  peak: {peak1:.0f} MiB  (activations: {peak1 - base_mem:.0f} MiB)  time: {t1:.1f}s')
    print(f'tt=2  peak: {peak2b:.0f} MiB  (activations: {peak2b - static_with_l2:.0f} MiB)  time: {t2b:.1f}s')
    act_delta = (peak2b - static_with_l2) - (peak1 - base_mem)
    print(f'Activation delta per extra tick:  {act_delta:.0f} MiB')
    print()
    print(f'Total VRAM: {total:.0f} MiB')
    print(f'Free at tt=1 b=1: {total - peak1:.0f} MiB')
    print(f'Free at tt=2 b=1: {total - peak2b:.0f} MiB')
    if peak3:
        print(f'tt=3  peak: {peak3:.0f} MiB  time: {t3:.1f}s')
        print(f'Free at tt=3 b=1: {total - peak3:.0f} MiB')
    print()

    # Extrapolate: how much batch can we fit at tt=2?
    act_per_sample_tt1 = peak1 - base_mem
    act_per_sample_tt2 = peak2b - static_with_l2
    budget_tt1 = total - base_mem
    budget_tt2 = total - static_with_l2
    max_batch_tt1 = int(budget_tt1 / act_per_sample_tt1) if act_per_sample_tt1 > 0 else 0
    max_batch_tt2 = int(budget_tt2 / act_per_sample_tt2) if act_per_sample_tt2 > 0 else 0
    print(f'Estimated max batch at tt=1: ~{max_batch_tt1}')
    print(f'Estimated max batch at tt=2: ~{max_batch_tt2}')
    print(f'(These are upper bounds — fragmentation reduces them)')

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
