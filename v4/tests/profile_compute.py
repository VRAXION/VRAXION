"""INSTNCT v4 — Compute profiling: what % of wall time is spent where?

Profiles both forward and backward pass, breaking down:
1. Input encoding (bitlift / embedding)
2. Ring read (gather + weighted sum)
3. Ring write (scatter / HDD replace)
4. Hidden update (c19 activation + add)
5. Phase computation
6. Pointer movement (pilot seek)
7. Output projection (lowrank_c19)
8. Ring clone overhead
9. Diagnostics overhead
10. Full forward vs backward ratio

Usage: python v4/tests/profile_compute.py
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add model dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))
from instnct import (
    INSTNCT, _c19_activation, _rho_from_raw, _C_from_raw,
    func_ringstart_tns, func_softread_tns, func_softwrit_tns,
    func_hdd_write_tns,
)

DEVICE = 'cpu'  # CPU profiling — no CUDA async issues
WARMUP = 3
REPEATS = 10

# Match production config
B = 16       # batch
T = 64       # seq_len (shorter for profiling)
M = 1024     # ring slots
HIDDEN = 2048
SLOT = 128
N = 1        # experts
R = 1        # attention radius


def time_fn(fn, warmup=WARMUP, repeats=REPEATS):
    """Time a function, return mean ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def profile_components():
    """Profile individual components of the forward pass."""
    print("=" * 70)
    print("INSTNCT v4 — Component-Level Compute Profile")
    print(f"Config: B={B}, T={T}, M={M}, hidden={HIDDEN}, slot={SLOT}, N={N}, R={R}")
    print(f"Device: {DEVICE}, warmup={WARMUP}, repeats={REPEATS}")
    print("=" * 70)

    torch.manual_seed(42)
    results = {}

    # ── 1. Input encoding (bitlift: byte → 8 bits → Linear(8, hidden)) ──
    inp_linear = nn.Linear(8, HIDDEN)
    c19_rho = nn.Parameter(torch.full((HIDDEN,), 1.0))
    c19_C = nn.Parameter(torch.full((HIDDEN,), 1.0))
    x_byte = torch.randint(0, 256, (B,), dtype=torch.long)
    bit_shifts = torch.arange(7, -1, -1)

    def input_encode():
        bits = ((x_byte.unsqueeze(-1) >> bit_shifts) & 1).float()
        v = inp_linear(bits)
        return _c19_activation(v, rho=_rho_from_raw(c19_rho), C=_C_from_raw(c19_C))

    results['1_input_encode'] = time_fn(input_encode)

    # ── 2. Ring read (gather + weighted sum) ──
    ring = torch.randn(B, M, SLOT)
    offsets = torch.arange(-R, R + 1)
    center = torch.randint(0, M, (B,))
    indices = (center.unsqueeze(1) + offsets) % M
    W = 2 * R + 1
    weights = torch.ones(B, W) / W

    def ring_read():
        exp_idx = indices.unsqueeze(-1).expand(-1, -1, SLOT)
        neighbors = ring.gather(1, exp_idx)
        return (weights.unsqueeze(-1) * neighbors).sum(1)

    results['2_ring_read'] = time_fn(ring_read)

    # ── 3. Read projection (slot_dim → hidden_dim) ──
    read_proj = nn.Linear(SLOT, HIDDEN)
    read_vec = torch.randn(B, SLOT)

    def read_projection():
        return read_proj(read_vec)

    results['3_read_projection'] = time_fn(read_projection)

    # ── 4. C19 activation (the custom activation function) ──
    pre_act = torch.randn(B, HIDDEN)

    def c19_act():
        return _c19_activation(pre_act, rho=_rho_from_raw(c19_rho), C=_C_from_raw(c19_C))

    results['4_c19_activation'] = time_fn(c19_act)

    # ── 5. Phase computation ──
    phase_cos = nn.Parameter(torch.randn(HIDDEN) * 0.01)
    phase_sin = nn.Parameter(torch.randn(HIDDEN) * 0.01)
    ptr_val = torch.rand(B) * M
    import math

    def phase_compute():
        theta = (ptr_val / M) * (2 * math.pi)
        return (torch.cos(theta).unsqueeze(-1) * phase_cos
                + torch.sin(theta).unsqueeze(-1) * phase_sin)

    results['5_phase'] = time_fn(phase_compute)

    # ── 6. Hidden update (add 4 terms + c19) ──
    inp_vec = torch.randn(B, HIDDEN)
    ring_sig = torch.randn(B, HIDDEN)
    phase_vec = torch.randn(B, HIDDEN)
    hidden = torch.randn(B, HIDDEN)

    def hidden_update():
        return _c19_activation(
            inp_vec + ring_sig + phase_vec + hidden,
            rho=_rho_from_raw(c19_rho), C=_C_from_raw(c19_C),
        )

    results['6_hidden_update'] = time_fn(hidden_update)

    # ── 7. Ring write — HDD replace mode ──
    write_vec = torch.randn(B, SLOT)
    exp_idx = indices.unsqueeze(-1).expand(-1, -1, SLOT)
    write_strength = torch.sigmoid(torch.randn(B, 1))

    def ring_write():
        return func_hdd_write_tns(ring, write_vec, exp_idx, weights, write_strength)

    results['7_ring_write_hdd'] = time_fn(ring_write)

    # ── 8. Ring clone overhead ──
    def ring_clone():
        return ring.clone()

    results['8_ring_clone'] = time_fn(ring_clone)

    # ── 9. Pilot pointer movement ──
    slot_identity = nn.Parameter(torch.randn(M, 32) * 0.01)
    ptr_query_linear = nn.Linear(HIDDEN, 32)
    ptr_tau = nn.Parameter(torch.tensor(5.0))

    def pilot_pointer():
        current_slot = ptr_val.long().clamp(0, M - 1)
        slot_id = F.normalize(slot_identity[current_slot], dim=-1)
        query = F.normalize(ptr_query_linear(hidden), dim=-1)
        sim = (query * slot_id).sum(-1)
        tau = F.softplus(ptr_tau)
        jump = 256 * torch.sigmoid(-sim * tau)
        return (ptr_val + 1 + jump) % M

    results['9_pilot_pointer'] = time_fn(pilot_pointer)

    # ── 10. Output projection (lowrank_c19: H→64→c19→256) ──
    out_head = nn.Sequential(
        nn.Linear(HIDDEN, 64),
        nn.ReLU(),  # placeholder — actual uses c19
        nn.Linear(64, 256),
    )
    mean_hidden = torch.randn(B, HIDDEN)

    def output_proj():
        return out_head(mean_hidden)

    results['10_output_projection'] = time_fn(output_proj)

    # ── 11. Cosine gate (dotprod ring gate) ──
    gate_tau = nn.Parameter(torch.tensor(4.0))

    def cosine_gate():
        cos_sim = F.cosine_similarity(inp_vec, ring_sig, dim=-1).unsqueeze(-1)
        alpha = torch.sigmoid(gate_tau * cos_sim)
        return alpha * ring_sig

    results['11_cosine_gate'] = time_fn(cosine_gate)

    # ── 12. Write gate (sigmoid Linear) ──
    write_gate_linear = nn.Linear(HIDDEN, 1)

    def write_gate():
        return torch.sigmoid(write_gate_linear(hidden))

    results['12_write_gate'] = time_fn(write_gate)

    # ── Print results ──
    total = sum(results.values())
    print(f"\n{'Component':<30} {'Time (ms)':>10} {'% of total':>10}")
    print("-" * 52)
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    for name, ms in sorted_results:
        pct = 100 * ms / total
        bar = "#" * int(pct / 2)
        print(f"{name:<30} {ms:>9.3f}ms {pct:>9.1f}%  {bar}")
    print("-" * 52)
    print(f"{'TOTAL (1 timestep, 1 expert)':<30} {total:>9.3f}ms")

    # ── Estimated loop cost ──
    # Forward = T timesteps × N experts × per-expert-cost
    est_per_step = total * T * N
    print(f"\nEstimated forward pass: {est_per_step:.1f}ms ({est_per_step/1000:.2f}s) for T={T}, N={N}")
    print(f"Projected for T=256: {total * 256 * N:.1f}ms ({total * 256 * N / 1000:.2f}s)")

    return results


def profile_full_forward_backward():
    """Profile full model forward + backward."""
    print("\n" + "=" * 70)
    print("INSTNCT v4 — Full Forward/Backward Profile")
    print("=" * 70)

    # Use smaller config for full model profiling
    model = INSTNCT(
        M=M, hidden_dim=HIDDEN, slot_dim=SLOT, N=N, R=R,
        embed_mode=True, kernel_mode='vshape',
        pointer_mode='pilot', write_mode='replace',
        embed_encoding='bitlift', output_encoding='lowrank_c19',
    )
    model.train()

    x = torch.randint(0, 256, (B, T), dtype=torch.long)

    # Warmup
    for _ in range(2):
        out, _ = model(x)
        loss = F.cross_entropy(out.view(-1, 256), x.view(-1))
        loss.backward()
        model.zero_grad()

    # Forward
    fwd_times = []
    for _ in range(5):
        model.zero_grad()
        t0 = time.perf_counter()
        out, _ = model(x)
        fwd_times.append((time.perf_counter() - t0) * 1000)

    # Backward
    bwd_times = []
    for _ in range(5):
        model.zero_grad()
        out, _ = model(x)
        loss = F.cross_entropy(out.view(-1, 256), x.view(-1))
        t0 = time.perf_counter()
        loss.backward()
        bwd_times.append((time.perf_counter() - t0) * 1000)

    fwd_mean = sum(fwd_times) / len(fwd_times)
    bwd_mean = sum(bwd_times) / len(bwd_times)

    print(f"\nForward:  {fwd_mean:>8.1f}ms")
    print(f"Backward: {bwd_mean:>8.1f}ms")
    print(f"Total:    {fwd_mean + bwd_mean:>8.1f}ms")
    print(f"Ratio:    backward is {bwd_mean/fwd_mean:.1f}x forward")

    # Steps/sec
    total_ms = fwd_mean + bwd_mean
    print(f"\nThroughput: {1000/total_ms:.2f} steps/sec (B={B}, T={T})")
    print(f"Projected T=256: {1000/(total_ms * 256/T):.2f} steps/sec")

    return fwd_mean, bwd_mean


def profile_loop_overhead():
    """Profile the Python loop overhead (for t in range(T): for i in range(N))."""
    print("\n" + "=" * 70)
    print("INSTNCT v4 — Loop & Allocation Overhead")
    print("=" * 70)

    # Pure Python loop overhead
    def empty_loop():
        for t in range(T):
            for i in range(N):
                pass

    ms_empty = time_fn(empty_loop, warmup=100, repeats=100)
    print(f"\nEmpty Python loop (T={T} × N={N}): {ms_empty:.4f}ms")

    # Tensor allocation overhead
    def alloc_loop():
        for t in range(T):
            _ = torch.zeros(B, HIDDEN)

    ms_alloc = time_fn(alloc_loop)
    print(f"Tensor alloc loop (T={T} × zeros(B,H)): {ms_alloc:.3f}ms")

    # clone ring per write
    ring = torch.randn(B, M, SLOT)

    def clone_loop():
        r = ring
        for t in range(T):
            r = r.clone()

    ms_clone = time_fn(clone_loop)
    print(f"Ring clone loop (T={T}): {ms_clone:.1f}ms")
    print(f"Ring clone per step: {ms_clone/T:.3f}ms")

    # Ring memory
    ring_bytes = B * M * SLOT * 4
    print(f"\nRing buffer size: {ring_bytes/1024:.1f} KB ({ring_bytes/1024/1024:.2f} MB)")
    print(f"T clones (forward): {ring_bytes * T / 1024 / 1024:.1f} MB")


def profile_c19_vs_relu():
    """Compare C19 activation vs standard ReLU/GELU."""
    print("\n" + "=" * 70)
    print("Activation Function Comparison")
    print("=" * 70)

    x = torch.randn(B, HIDDEN)
    rho = nn.Parameter(torch.full((HIDDEN,), 1.0))
    C_param = nn.Parameter(torch.full((HIDDEN,), 1.0))

    def do_relu():
        return F.relu(x)

    def do_gelu():
        return F.gelu(x)

    def do_c19_fixed():
        return _c19_activation(x, rho=4.0)

    def do_c19_learnable():
        return _c19_activation(x, rho=_rho_from_raw(rho), C=_C_from_raw(C_param))

    ms_relu = time_fn(do_relu, repeats=50)
    ms_gelu = time_fn(do_gelu, repeats=50)
    ms_c19_fix = time_fn(do_c19_fixed, repeats=50)
    ms_c19_learn = time_fn(do_c19_learnable, repeats=50)

    print(f"\n{'Activation':<25} {'Time (ms)':>10} {'vs ReLU':>10}")
    print("-" * 47)
    print(f"{'ReLU':<25} {ms_relu:>9.4f}ms {'1.0x':>10}")
    print(f"{'GELU':<25} {ms_gelu:>9.4f}ms {ms_gelu/ms_relu:>9.1f}x")
    print(f"{'C19 (fixed rho=4)':<25} {ms_c19_fix:>9.4f}ms {ms_c19_fix/ms_relu:>9.1f}x")
    print(f"{'C19 (learnable rho+C)':<25} {ms_c19_learn:>9.4f}ms {ms_c19_learn/ms_relu:>9.1f}x")


def profile_with_torch_profiler():
    """Use torch.profiler for detailed op-level breakdown."""
    print("\n" + "=" * 70)
    print("PyTorch Profiler — Op-Level Breakdown (1 forward pass)")
    print("=" * 70)

    model = INSTNCT(
        M=M, hidden_dim=HIDDEN, slot_dim=SLOT, N=N, R=R,
        embed_mode=True, kernel_mode='vshape',
        pointer_mode='pilot', write_mode='replace',
        embed_encoding='bitlift', output_encoding='lowrank_c19',
    )
    model.train()
    x = torch.randint(0, 256, (B, T), dtype=torch.long)

    # Warmup
    out, _ = model(x)
    loss = F.cross_entropy(out.view(-1, 256), x.view(-1))
    loss.backward()
    model.zero_grad()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        out, _ = model(x)
        loss = F.cross_entropy(out.view(-1, 256), x.view(-1))
        loss.backward()

    # Print top 30 ops by CPU time
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    return prof


if __name__ == '__main__':
    profile_components()
    profile_full_forward_backward()
    profile_loop_overhead()
    profile_c19_vs_relu()
    profile_with_torch_profiler()
