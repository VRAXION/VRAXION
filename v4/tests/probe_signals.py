"""Deep signal analysis on best checkpoint — no hooks, manual probing."""

import sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from model_factory import build_model_from_spec

# ── Load model ──
ckpt = torch.load(ROOT / 'training_output/ckpt_latest.pt', map_location='cpu', weights_only=False)
spec = ckpt['model']
model = build_model_from_spec(spec, 'cpu')
model.load_state_dict(spec['state_dict'])
model.eval()

S = model.S_raw.item()
print(f'Checkpoint: step={ckpt["step"]}, best_loss={ckpt["best_loss"]:.6f}')
print(f'S={S:.4f}, N={model.N}, M={model.M}, hidden={model.hidden_dim}, slot={model.slot_dim}, R={model.R}')

# ── Run model for a few steps to build state, then probe ──
state = None
for step in range(50):
    xb = torch.randint(0, 256, (16, 64))
    with torch.no_grad():
        pred, new_state = model(xb, state=state)
    if new_state:
        state = {k: v.detach() for k, v in new_state.items()}

# Check for NaN
if state:
    for k, v in state.items():
        if torch.is_tensor(v):
            has_nan = v.isnan().any().item()
            has_inf = v.isinf().any().item()
            print(f'state[{k}]: shape={list(v.shape)} nan={has_nan} inf={has_inf} norm={v.norm():.2f}')

ring = state['ring']  # (B, M, slot_dim)
ptr = state['ptr']    # (N, B)
hid = state['hidden'] # (N, B, hidden_dim)

if ring.isnan().any():
    print('\n*** RING HAS NaN — aborting analysis ***')
    # Try with just 1 step
    state = None
    xb = torch.randint(0, 256, (16, 64))
    with torch.no_grad():
        pred, state2 = model(xb, state=state)
    for k, v in state2.items():
        if torch.is_tensor(v):
            print(f'  fresh state[{k}]: nan={v.isnan().any().item()} norm={v.norm():.2f}')
    ring = state2['ring']
    ptr = state2['ptr']
    hid = state2['hidden']

B, M_actual, sd = ring.shape
N = model.N
R = model.R

print()
print('=' * 70)
print('MANUAL SIGNAL PROBE (last state)')
print('=' * 70)

# Input embedding for a random byte
byte_val = torch.randint(0, 256, (B,))
_bit_shifts = torch.arange(7, -1, -1)
bits = ((byte_val.unsqueeze(-1) >> _bit_shifts) & 1).float()
rho_inp = F.softplus(model.c19_rho_input)
C_inp = model.c19_C_input
inp_raw = model.inp(bits)
inp_vec = rho_inp * torch.tanh(inp_raw / rho_inp + C_inp)  # c19 activation
print(f'input (pre-c19):  {inp_raw.norm(dim=-1).mean():.4f}')
print(f'input (post-c19): {inp_vec.norm(dim=-1).mean():.4f}')

# Ring read for expert 0
i = 0
ptr_f = ptr[i]
center = ptr_f.long() % M_actual
offsets = torch.arange(-R, R + 1)
indices = (center.unsqueeze(-1) + offsets.unsqueeze(0)) % M_actual
expanded_idx = indices.unsqueeze(-1).expand(-1, -1, sd)
neighbors = ring.gather(1, expanded_idx)

# V-shape kernel
abs_off = offsets.abs().float()
w = 1.0 - abs_off / (R + 1)
w = w / w.sum()
read_vec = (w.unsqueeze(0).unsqueeze(-1) * neighbors).sum(1)  # (B, slot_dim)

ring_signal = model.read_proj[i](read_vec)  # slot_dim -> hidden_dim
blended = S * ring_signal

# Phase
theta = (ptr_f / M_actual) * (2 * math.pi)
phase = (torch.cos(theta).unsqueeze(-1) * model.phase_cos
       + torch.sin(theta).unsqueeze(-1) * model.phase_sin)

# Write
write_vec = model.write_proj[i](hid[i])

print(f'read_raw (sd=64):   {read_vec.norm(dim=-1).mean():.4f}')
print(f'ring_signal (2048):  {ring_signal.norm(dim=-1).mean():.4f}')
print(f'blended (S*sig):     {blended.norm(dim=-1).mean():.4f}')
print(f'phase:               {phase.norm(dim=-1).mean():.4f}')
print(f'hidden (carry):      {hid[i].norm(dim=-1).mean():.4f}')
print(f'write_vec (sd=64):   {write_vec.norm(dim=-1).mean():.4f}')

# Composition
inp_n = inp_vec.norm(dim=-1).mean().item()
blend_n = blended.norm(dim=-1).mean().item()
phase_n = phase.norm(dim=-1).mean().item()
hid_n = hid[i].norm(dim=-1).mean().item()
total = inp_n + blend_n + phase_n + hid_n

print()
print('HIDDEN UPDATE COMPOSITION')
print(f'  hidden_new = c19(input + S*ring_signal + phase + hidden_old)')
print(f'  input:         {inp_n:>8.2f}  ({inp_n/total*100:>5.1f}%)')
print(f'  S*ring_signal: {blend_n:>8.2f}  ({blend_n/total*100:>5.1f}%)')
print(f'  phase:         {phase_n:>8.2f}  ({phase_n/total*100:>5.1f}%)')
print(f'  hidden_old:    {hid_n:>8.2f}  ({hid_n/total*100:>5.1f}%)')
print(f'  total pre-c19: {total:>8.2f}')
print()
print(f'  Ring contribution:  {blend_n/total*100:.1f}% of hidden update')
print(f'  Ring vs input:      {blend_n/inp_n*100:.1f}%')
print(f'  Ring vs hidden:     {blend_n/hid_n*100:.1f}%')

# ── Ring state analysis ──
print()
print('=' * 70)
print('RING STATE ANALYSIS')
print('=' * 70)
slot_norms = ring.norm(dim=-1)
print(f'  slot norm:  mean={slot_norms.mean():.4f}  std={slot_norms.std():.4f}')
print(f'              min={slot_norms.min():.4f}  max={slot_norms.max():.4f}')

active = (slot_norms > 0.1).float().mean().item()
print(f'  active slots: {active*100:.1f}%')

if not ring.isnan().any():
    cos_adj = F.cosine_similarity(ring[:, :-1], ring[:, 1:], dim=-1)
    print(f'  adjacent cos_sim:  mean={cos_adj.mean():.4f}  std={cos_adj.std():.4f}')

    idx1 = torch.randint(0, M_actual, (2000,))
    idx2 = torch.randint(0, M_actual, (2000,))
    cos_rand = F.cosine_similarity(ring[0, idx1], ring[0, idx2], dim=-1)
    print(f'  random cos_sim:    mean={cos_rand.mean():.4f}  std={cos_rand.std():.4f}')

    ring_flat = ring[0]
    try:
        U, S_vals, V = torch.svd(ring_flat)
        total_var = (S_vals ** 2).sum()
        cumvar = (S_vals ** 2).cumsum(0) / total_var
        rank_90 = (cumvar < 0.90).sum().item() + 1
        rank_95 = (cumvar < 0.95).sum().item() + 1
        rank_99 = (cumvar < 0.99).sum().item() + 1
        print(f'  SVD effective rank: 90%={rank_90}  95%={rank_95}  99%={rank_99}  (max={sd})')
        print(f'  top 10 SV:    {" ".join(f"{s:.1f}" for s in S_vals[:10].tolist())}')
    except:
        print(f'  SVD failed')

# ── Output projection analysis ──
print()
print('=' * 70)
print('OUTPUT PATH ANALYSIS')
print('=' * 70)
out_w0 = model.out[0].weight.data  # (64, 2048)
out_w2 = model.out[2].weight.data  # (256, 64)
_, S0, _ = torch.svd(out_w0)
_, S2, _ = torch.svd(out_w2)
print(f'  out.0 (2048->64): condition={S0[0]/S0[-1]:.1f}x')
print(f'    top 5 SV: {" ".join(f"{s:.3f}" for s in S0[:5].tolist())}')
print(f'  out.2 (64->256):  condition={S2[0]/S2[-1]:.1f}x')
print(f'    top 5 SV: {" ".join(f"{s:.3f}" for s in S2[:5].tolist())}')

# How much info passes through the 64-dim bottleneck?
# Test: project a typical hidden through out.0, then back-project
test_hidden = hid[0]  # (B, 2048)
projected = F.linear(test_hidden, out_w0)  # (B, 64)
reconstructed = F.linear(projected, out_w0.T)  # (B, 2048)
recon_cos = F.cosine_similarity(test_hidden, reconstructed, dim=-1)
print(f'  round-trip cos_sim (2048->64->2048): {recon_cos.mean():.4f}')
info_preserved = (reconstructed.norm(dim=-1) / test_hidden.norm(dim=-1)).mean()
print(f'  round-trip norm ratio: {info_preserved:.4f}')

# ── C19 saturation check ──
print()
print('=' * 70)
print('C19 SATURATION ANALYSIS')
print('=' * 70)
rho_h = F.softplus(model.c19_rho_hidden)
C_h = model.c19_C_hidden
# c19(x) = rho * tanh(x/rho + C)
# tanh saturates when |x/rho + C| >> 1
# Check: what's the argument to tanh for a typical hidden state?
test_x = hid[0]  # (B, 2048)
tanh_arg = test_x / rho_h + C_h
tanh_out = torch.tanh(tanh_arg)
print(f'  tanh argument: mean={tanh_arg.mean():.4f}  std={tanh_arg.std():.4f}')
print(f'  tanh argument: |arg| > 2.0: {(tanh_arg.abs() > 2.0).float().mean()*100:.1f}%')
print(f'  tanh argument: |arg| > 3.0: {(tanh_arg.abs() > 3.0).float().mean()*100:.1f}%')
print(f'  tanh output:   mean={tanh_out.mean():.4f}  std={tanh_out.std():.4f}')
print(f'  saturation (|tanh| > 0.95): {(tanh_out.abs() > 0.95).float().mean()*100:.1f}%')
print(f'  rho_hidden: mean={rho_h.mean():.4f}  std={rho_h.std():.4f}')
print(f'  C_hidden:   mean={C_h.mean():.4f}  std={C_h.std():.4f}')
