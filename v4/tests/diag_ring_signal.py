"""Ring signal diagnostic — measures if the ring actually contributes to learning.

Probes:
1. Signal magnitudes: how big is ring read vs input vs hidden?
2. Gradient flow: does write_proj/read_proj get meaningful gradients?
3. Ring content: are slots diverse or collapsed?
4. Ring delta: does writing actually change the ring?
"""
import torch
import sys
from pathlib import Path

_v4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_v4 / 'model'))
from instnct import INSTNCT

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Full-scale model (same as training)
model = INSTNCT(M=1024, hidden_dim=2048, slot_dim=64, N=2, R=2,
                embed_mode=True, kernel_mode='vshape',
                embed_encoding='learned', output_encoding='lowrank_c19',
                pointer_mode='sequential')
model.to(device)
model.train()

x = torch.randint(0, 256, (16, 64), device=device)  # (B=16, T=64)

# ── Hook to capture intermediate signals ──
signals = {}

def hook_hidden_update(module, input, output):
    """Capture the hidden state update components."""
    pass  # We'll measure differently

# ── Forward pass with signal capture ──
# We need to instrument the model temporarily
# Monkey-patch _process_chunk to capture signals

_orig_process = model._process_chunk.__func__

def _instrumented_process(self, x_chunk, ring_tns, ptr_tns, hidden_tns,
                          S_flt, probs_lst, offsets_long, expert_weights):
    import math
    N = self.N
    M = self.M
    slot_dim = self.slot_dim

    ptr_tns = ptr_tns.clone()
    hidden_lst = [hidden_tns[i] for i in range(N)]
    C = x_chunk.shape[1]
    outs_lst = []

    # Accumulators for diagnostics
    input_mags = []
    read_mags = []
    hidden_mags = []
    phase_mags = []
    ring_before = ring_tns.clone()

    _bit_shifts = torch.arange(7, -1, -1, device=x_chunk.device)

    for t in range(C):
        if self._bitlift:
            byte_val = x_chunk[:, t]
            bits = ((byte_val.unsqueeze(-1) >> _bit_shifts) & 1).float()
            from instnct import _c19_activation, _rho_from_raw, _C_from_raw
            input_vec_tns = _c19_activation(self.inp(bits),
                                            rho=_rho_from_raw(self.c19_rho_input),
                                            C=_C_from_raw(self.c19_C_input))
        elif self.inp is not None:
            input_vec_tns = self.inp(x_chunk[:, t])
        else:
            input_vec_tns = self._fixed_table[x_chunk[:, t]]

        for i in range(N):
            center = ptr_tns[i].long().clamp(0, M - 1)
            indices_tns = (center.unsqueeze(1) + offsets_long) % M

            weights_tns = expert_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
            from instnct import func_softread_tns
            read_vec_tns, expanded_idx_tns = func_softread_tns(
                ring_tns, indices_tns, weights_tns, slot_dim)

            theta_tns = (ptr_tns[i] / M) * (2 * math.pi)
            phase_tns = (torch.cos(theta_tns).unsqueeze(-1) * self.phase_cos
                       + torch.sin(theta_tns).unsqueeze(-1) * self.phase_sin)

            read_signal = S_flt * self.read_proj[i](read_vec_tns)

            # ── CAPTURE MAGNITUDES ──
            input_mags.append(input_vec_tns.detach().norm(dim=-1).mean().item())
            read_mags.append(read_signal.detach().norm(dim=-1).mean().item())
            hidden_mags.append(hidden_lst[i].detach().norm(dim=-1).mean().item())
            phase_mags.append(phase_tns.detach().norm(dim=-1).mean().item())

            from instnct import _c19_activation, _rho_from_raw, _C_from_raw
            hidden_lst[i] = _c19_activation(
                input_vec_tns + read_signal + phase_tns + hidden_lst[i],
                rho=_rho_from_raw(self.c19_rho_hidden),
                C=_C_from_raw(self.c19_C_hidden),
            )

            if self.write_proj is not None:
                write_vec = self.write_proj[i](hidden_lst[i])
            else:
                write_vec = hidden_lst[i]
            from instnct import func_softwrit_tns
            ring_tns = func_softwrit_tns(ring_tns, write_vec, expanded_idx_tns, weights_tns)

            ptr_tns[i] = (ptr_tns[i] + 1) % M

        mean_hidden = torch.stack(hidden_lst).mean(0)
        if self._bitlift_out:
            bit_scores = torch.tanh(self.out(mean_hidden))
            outs_lst.append(bit_scores @ self._bit_patterns.T)
        elif self.out is not None:
            outs_lst.append(self.out(mean_hidden))
        else:
            outs_lst.append(mean_hidden @ self._fixed_output_table.T)

    # Store diagnostics
    signals['input_mag'] = sum(input_mags) / len(input_mags)
    signals['read_mag'] = sum(read_mags) / len(read_mags)
    signals['hidden_mag'] = sum(hidden_mags) / len(hidden_mags)
    signals['phase_mag'] = sum(phase_mags) / len(phase_mags)
    signals['ring_delta'] = (ring_tns - ring_before).detach().norm().item()

    hidden_tns = torch.stack(hidden_lst)
    outs_tns = torch.stack(outs_lst, dim=1)
    return ring_tns, ptr_tns, hidden_tns, outs_tns

# Bind instrumented version
import types
model._process_chunk = types.MethodType(_instrumented_process, model)

# ── Run forward + backward ──
out, state = model(x, S=0.3)
loss = torch.nn.functional.cross_entropy(out.view(-1, 256), x.view(-1))
loss.backward()

# ── Collect gradient magnitudes ──
def grad_norm(params):
    norms = []
    for p in params:
        if p.grad is not None:
            norms.append(p.grad.detach().norm().item())
    return sum(norms) / max(len(norms), 1)

read_grad = grad_norm(model.read_proj.parameters())
write_grad = grad_norm(model.write_proj.parameters()) if model.write_proj else 0
inp_grad = grad_norm(model.inp.parameters()) if model.inp is not None else 0
out_grad = grad_norm(model.out.parameters()) if model.out is not None else 0

# ── Ring content analysis ──
ring = state['ring'].detach()  # (B, M, slot_dim)
slot_norms = ring.norm(dim=-1)  # (B, M)
slot_std = slot_norms.std(dim=-1).mean().item()   # diversity across slots
slot_mean = slot_norms.mean().item()

# Cosine similarity between random slot pairs
idx1 = torch.randint(0, 1024, (100,))
idx2 = torch.randint(0, 1024, (100,))
s1 = ring[0, idx1]  # (100, slot_dim)
s2 = ring[0, idx2]  # (100, slot_dim)
cos_sim = torch.nn.functional.cosine_similarity(s1, s2, dim=-1).mean().item()

# ── Print Report ──
print("=" * 60)
print("  RING SIGNAL DIAGNOSTIC")
print("=" * 60)
print()
print("1. SIGNAL MAGNITUDES (hidden update components):")
print(f"   input_vec   |norm| = {signals['input_mag']:.4f}")
print(f"   S*read_proj |norm| = {signals['read_mag']:.4f}")
print(f"   hidden      |norm| = {signals['hidden_mag']:.4f}")
print(f"   phase       |norm| = {signals['phase_mag']:.4f}")
ratio = signals['read_mag'] / (signals['input_mag'] + 1e-8)
print(f"   ring/input ratio   = {ratio:.4f}")
if ratio < 0.01:
    print("   !! RING SIGNAL DROWNING -- too small vs input")
elif ratio < 0.1:
    print("   ~  Ring signal weak but present")
else:
    print("   OK Ring signal competitive with input")
print()

print("2. GRADIENT FLOW:")
print(f"   inp (encoder)  grad_norm = {inp_grad:.6f}")
print(f"   read_proj      grad_norm = {read_grad:.6f}")
print(f"   write_proj     grad_norm = {write_grad:.6f}")
print(f"   out (decoder)  grad_norm = {out_grad:.6f}")
if read_grad < inp_grad * 0.01:
    print("   !! READ_PROJ gets <1% of inp gradient — ring path starved")
elif read_grad < inp_grad * 0.1:
    print("   ~  read_proj gets some gradient but much less than inp")
else:
    print("   OK read_proj gradient comparable to inp")
print()

print("3. RING CONTENT:")
print(f"   slot |norm| mean  = {slot_mean:.4f}")
print(f"   slot |norm| std   = {slot_std:.4f}  (diversity)")
print(f"   cosine sim (random pairs) = {cos_sim:.4f}")
print(f"   ring write delta  = {signals['ring_delta']:.4f}")
if cos_sim > 0.9:
    print("   !! RING COLLAPSED — all slots ~identical")
elif cos_sim > 0.7:
    print("   ~  Ring partially collapsed")
else:
    print("   OK Ring has diverse content")
print()

print("4. SUMMARY:")
issues = []
if ratio < 0.01:
    issues.append("Ring signal drowned by input (S too small or read too weak)")
if read_grad < inp_grad * 0.01:
    issues.append("Ring path gets no gradient (write->ring->read->hidden chain broken)")
if cos_sim > 0.9:
    issues.append("Ring slots collapsed (writing same thing everywhere)")
if signals['ring_delta'] < 0.01:
    issues.append("Ring barely changes during forward pass")

if issues:
    print("   ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")
else:
    print("   No obvious issues — ring appears functional")
print()
