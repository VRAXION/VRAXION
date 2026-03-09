"""probe_dataflow.py — Signal trace through INSTNCT pipeline.

Feeds known inputs through the model at different training stages and
measures signal quality at each internal stage. Identifies the EXACT
stage where the signal degrades.

Three probes:
  1. Forward stream: diversity + magnitude at each pipeline stage
  2. Backward micro: gradient flow to each learnable component
  3. State health: ring SVD rank, adj_cos over streaming steps

Usage:
  python probe_dataflow.py                     # fresh init only
  python probe_dataflow.py --train-steps 500   # fresh + after 500 steps
  python probe_dataflow.py --train-steps 3000  # fresh + after 3000 steps
"""

import sys, os, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── path setup ──
V4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V4)
from model.instnct import (
    INSTNCT, _c19_activation, _rho_from_raw, _C_from_raw,
    func_softread_tns, func_softwrit_tns, func_ringstart_tns,
)

# ═══════════════════════════════════════════════════════════════
#  PROBE 1: Forward Stream — trace signal at each pipeline stage
# ═══════════════════════════════════════════════════════════════

def probe_forward_stream(model, device, state=None, label=''):
    """Trace two different inputs (A=65, B=66) through the model.
    Measure diversity (cos_sim), magnitude, saturation at each stage."""

    model.eval()
    M, N = model.M, model.N
    hidden_dim, slot_dim = model.hidden_dim, model.slot_dim

    # Two test bytes
    byte_A = torch.tensor([[65]], device=device)  # 'A'
    byte_B = torch.tensor([[66]], device=device)  # 'B'

    # State: provided or fresh
    if state is not None:
        ring = state['ring'][:1]   # take first batch element
        ptr  = state['ptr'][:, :1]
        hid  = state['hidden'][:, :1]
    else:
        ring = func_ringstart_tns(1, M, slot_dim, device)
        ptr  = torch.zeros(N, 1, device=device)
        for i in range(N):
            ptr[i] = (i * M // N) % M
        hid = torch.zeros(N, 1, hidden_dim, device=device)

    results = {}

    with torch.no_grad():
        # ── Stage 1: BITLIFT ──
        shifts = torch.arange(7, -1, -1, device=device)
        bits_A = ((byte_A.view(-1).unsqueeze(-1) >> shifts) & 1).float()
        bits_B = ((byte_B.view(-1).unsqueeze(-1) >> shifts) & 1).float()

        inp_A = _c19_activation(
            model.inp(bits_A),
            rho=_rho_from_raw(model.c19_rho_input),
            C=_C_from_raw(model.c19_C_input),
        )
        inp_B = _c19_activation(
            model.inp(bits_B),
            rho=_rho_from_raw(model.c19_rho_input),
            C=_C_from_raw(model.c19_C_input),
        )
        inp_cos = F.cosine_similarity(inp_A, inp_B, dim=-1).item()
        inp_norm_A = inp_A.norm().item()
        results['1_bitlift'] = {
            'cos_AB': inp_cos,
            'norm_A': inp_norm_A,
            'norm_B': inp_B.norm().item(),
            'sat_%': ((inp_A.abs() > 0.95).float().mean().item()) * 100,
        }

        # ── Per-expert stages ──
        # Precompute vshape weights (same as forward())
        R_effs = model._R_eff
        win = int(math.floor(R_effs.max().item()))
        offsets = torch.arange(-win, win + 1, device=device)
        abs_off = offsets.float().abs()

        if model.kernel_mode == 'vshape':
            raw_w = (1.0 - abs_off.unsqueeze(0) / R_effs.unsqueeze(1).clamp(min=0.5)).clamp(min=0)
            ew = raw_w / raw_w.sum(dim=1, keepdim=True)  # (N, 2R+1)
        else:
            ew = torch.ones(N, 2 * win + 1, device=device) / (2 * win + 1)

        S_val = model.S_raw.item() if hasattr(model, 'S_raw') else 1.0

        for i in range(N):
            pfx = f'expert{i}'
            center = ptr[i].long().clamp(0, M - 1)
            indices = (center.unsqueeze(1) + offsets) % M  # (1, 2R+1)
            weights = ew[i].unsqueeze(0)  # (1, 2R+1)

            # ── Stage 2: RING READ ──
            read_A, exp_idx = func_softread_tns(ring, indices, weights, slot_dim)
            read_B, _ = func_softread_tns(ring, indices, weights, slot_dim)
            # (read_A == read_B because same ring + same pointer position)
            read_norm = read_A.norm().item()

            # Also check: how different are nearby vs far slots?
            slot_at_ptr = ring[0, center[0].item()]
            slot_far = ring[0, (center[0].item() + M // 2) % M]
            slot_cos = F.cosine_similarity(
                slot_at_ptr.unsqueeze(0), slot_far.unsqueeze(0), dim=-1
            ).item() if slot_at_ptr.norm() > 1e-6 and slot_far.norm() > 1e-6 else 0.0

            results[f'2_ring_read_{pfx}'] = {
                'read_norm': read_norm,
                'slot_near_vs_far_cos': slot_cos,
                'ptr_pos': ptr[i].item(),
            }

            # ── Stage 3: READ PROJECTION + BLEND ──
            ring_signal_A = model.read_proj[i](read_A)
            ring_signal_B = model.read_proj[i](read_B)
            blended_A = S_val * ring_signal_A
            blended_B = S_val * ring_signal_B

            results[f'3_blend_{pfx}'] = {
                'ring_signal_norm': ring_signal_A.norm().item(),
                'blended_norm': blended_A.norm().item(),
                'input_norm': inp_norm_A,
                'blend_vs_input_ratio': blended_A.norm().item() / max(inp_norm_A, 1e-8),
            }

            # ── Stage 4: PHASE ──
            theta = (ptr[i] / M) * (2 * math.pi)
            phase = (torch.cos(theta).unsqueeze(-1) * model.phase_cos
                   + torch.sin(theta).unsqueeze(-1) * model.phase_sin)
            results[f'4_phase_{pfx}'] = {
                'phase_norm': phase.norm().item(),
                'phase_vs_input_ratio': phase.norm().item() / max(inp_norm_A, 1e-8),
            }

            # ── Stage 5: HIDDEN UPDATE ──
            new_hid_A = _c19_activation(
                inp_A + blended_A + phase + hid[i],
                rho=_rho_from_raw(model.c19_rho_hidden),
                C=_C_from_raw(model.c19_C_hidden),
            )
            new_hid_B = _c19_activation(
                inp_B + blended_B + phase + hid[i],
                rho=_rho_from_raw(model.c19_rho_hidden),
                C=_C_from_raw(model.c19_C_hidden),
            )
            hid_cos = F.cosine_similarity(new_hid_A, new_hid_B, dim=-1).item()
            sat = (new_hid_A.abs() > 0.95).float().mean().item() * 100

            results[f'5_hidden_{pfx}'] = {
                'cos_AB': hid_cos,
                'norm_A': new_hid_A.norm().item(),
                'sat_%': sat,
                'prev_hidden_norm': hid[i].norm().item(),
            }

            # ── Stage 6: WRITE ──
            if model.write_proj is not None:
                write_A = model.write_proj[i](new_hid_A)
                write_B = model.write_proj[i](new_hid_B)
            else:
                write_A, write_B = new_hid_A, new_hid_B

            write_cos = F.cosine_similarity(write_A, write_B, dim=-1).item()
            results[f'6_write_{pfx}'] = {
                'cos_AB': write_cos,
                'norm_A': write_A.norm().item(),
                'write_vs_slot_norm_ratio': write_A.norm().item() / max(
                    ring.norm(dim=-1).mean().item(), 1e-8),
            }

        # ── Stage 7: OUTPUT ──
        mean_hid_A = _c19_activation(
            inp_A + S_val * model.read_proj[0](read_A) + phase + hid[0],
            rho=_rho_from_raw(model.c19_rho_hidden),
            C=_C_from_raw(model.c19_C_hidden),
        )
        mean_hid_B = _c19_activation(
            inp_B + S_val * model.read_proj[0](read_B) + phase + hid[0],
            rho=_rho_from_raw(model.c19_rho_hidden),
            C=_C_from_raw(model.c19_C_hidden),
        )
        # Simplified: use expert 0 hidden for output (real model averages all)
        if hasattr(model, '_bitlift_out') and model._bitlift_out:
            bit_scores_A = torch.tanh(model.out(mean_hid_A))
            logits_A = bit_scores_A @ model._bit_patterns.T
            bit_scores_B = torch.tanh(model.out(mean_hid_B))
            logits_B = bit_scores_B @ model._bit_patterns.T
        elif model.out is not None:
            logits_A = model.out(mean_hid_A)
            logits_B = model.out(mean_hid_B)
        else:
            logits_A = mean_hid_A
            logits_B = mean_hid_B

        pred_A = logits_A.argmax(-1).item()
        pred_B = logits_B.argmax(-1).item()
        ent_A = -(F.softmax(logits_A, dim=-1) * F.log_softmax(logits_A, dim=-1)).sum().item()
        logit_cos = F.cosine_similarity(logits_A, logits_B, dim=-1).item()

        results['7_output'] = {
            'pred_A': pred_A,
            'pred_B': pred_B,
            'preds_different': pred_A != pred_B,
            'logit_cos_AB': logit_cos,
            'entropy_A': ent_A,
            'max_entropy': math.log(256),  # 5.545 for uniform over 256 classes
        }

    # ── Pretty print ──
    print(f'\n{"="*65}')
    print(f'  FORWARD STREAM PROBE {label}')
    print(f'{"="*65}')
    for stage, metrics in sorted(results.items()):
        print(f'\n  [{stage}]')
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f'    {k:30s} = {v:>10.4f}')
            else:
                print(f'    {k:30s} = {v}')

    return results


# ═══════════════════════════════════════════════════════════════
#  PROBE 2: Backward Micro — gradient flow check
# ═══════════════════════════════════════════════════════════════

def probe_backward(model, device, label=''):
    """One forward+backward pass with synthetic data.
    Check gradient norms at each key parameter."""

    model.train()
    model.zero_grad()

    # Synthetic input: 16 random bytes, batch=4
    B, T = 4, 16
    x = torch.randint(0, 256, (B, T), device=device)

    # Forward
    logits, _ = model(x)

    # Synthetic CE loss (random targets)
    targets = torch.randint(0, 256, (B, T), device=device)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()

    # Collect gradient norms for key parameters
    grad_report = {}

    # Bitlift input layer
    if model.inp is not None and model.inp.weight.grad is not None:
        grad_report['inp.weight'] = model.inp.weight.grad.norm().item()

    # Read projections
    for i in range(model.N):
        g = model.read_proj[i].weight.grad
        grad_report[f'read_proj[{i}].weight'] = g.norm().item() if g is not None else 0.0

    # Write projections
    if model.write_proj is not None:
        for i in range(model.N):
            g = model.write_proj[i].weight.grad
            grad_report[f'write_proj[{i}].weight'] = g.norm().item() if g is not None else 0.0

    # Output layer (may be Sequential for lowrank_c19)
    if model.out is not None:
        for pname, p in model.out.named_parameters():
            if p.grad is not None:
                grad_report[f'out.{pname}'] = p.grad.norm().item()

    # C19 learnable params
    if hasattr(model, 'c19_rho_hidden') and model.c19_rho_hidden.grad is not None:
        grad_report['c19_rho_hidden'] = model.c19_rho_hidden.grad.norm().item()
    if hasattr(model, 'c19_C_hidden') and model.c19_C_hidden.grad is not None:
        grad_report['c19_C_hidden'] = model.c19_C_hidden.grad.norm().item()

    # Phase embeddings
    if hasattr(model, 'phase_cos') and model.phase_cos.grad is not None:
        grad_report['phase_cos'] = model.phase_cos.grad.norm().item()

    # Gated write params
    if hasattr(model, 'erase_raw') and hasattr(model.erase_raw, 'grad') and model.erase_raw.grad is not None:
        grad_report['erase_raw'] = model.erase_raw.grad.norm().item()
    if hasattr(model, 'write_gate_raw') and hasattr(model.write_gate_raw, 'grad') and model.write_gate_raw.grad is not None:
        grad_report['write_gate_raw'] = model.write_gate_raw.grad.norm().item()

    # S_raw
    if hasattr(model, 'S_raw') and getattr(model.S_raw, 'grad', None) is not None:
        grad_report['S_raw'] = model.S_raw.grad.norm().item()

    model.zero_grad()

    # ── Pretty print ──
    print(f'\n{"="*65}')
    print(f'  BACKWARD MICRO PROBE {label}')
    print(f'{"="*65}')
    max_grad = max(grad_report.values()) if grad_report else 1.0
    for name, val in sorted(grad_report.items(), key=lambda x: -x[1]):
        bar_len = int(30 * val / max(max_grad, 1e-10))
        bar = '#' * bar_len
        status = 'OK' if val > 1e-8 else 'DEAD'
        print(f'  {name:30s}  {val:>10.6f}  {bar:30s}  [{status}]')

    dead = [k for k, v in grad_report.items() if v < 1e-8]
    if dead:
        print(f'\n  WARNING: Dead gradient paths: {dead}')
    else:
        print(f'\n  All gradient paths alive.')

    return grad_report


# ═══════════════════════════════════════════════════════════════
#  PROBE 3: State Health — ring quality over streaming steps
# ═══════════════════════════════════════════════════════════════

def probe_state_health(model, device, n_steps=200, label=''):
    """Stream n_steps of random input through the model (no optimizer).
    Track ring SVD rank, adj_cos, pointer coverage over time."""

    model.eval()
    M = model.M

    state = None
    history = []
    ptr_visited = set()

    with torch.no_grad():
        for step in range(n_steps):
            # Random input byte
            x = torch.randint(0, 256, (1, 1), device=device)
            logits, state = model(x, state=state)

            # Track pointer positions
            for i in range(model.N):
                pos = int(state['ptr'][i, 0].item()) % M
                ptr_visited.add(pos)

            # Ring health every 50 steps
            if (step + 1) % 50 == 0 or step == 0:
                ring = state['ring']  # (1, M, slot_dim)
                slot_norms = ring[0].norm(dim=-1)  # (M,)
                active = (slot_norms > 0.1).float().mean().item()

                # Adjacent cosine similarity
                cos_adj = F.cosine_similarity(
                    ring[:, :-1], ring[:, 1:], dim=-1
                ).mean().item()

                # SVD rank
                try:
                    _, S_vals, _ = torch.svd(ring[0])
                    total_var = (S_vals ** 2).sum()
                    cumvar = (S_vals ** 2).cumsum(0) / total_var
                    rank_90 = (cumvar < 0.90).sum().item() + 1
                except:
                    rank_90 = -1

                history.append({
                    'step': step + 1,
                    'rank_90': rank_90,
                    'adj_cos': cos_adj,
                    'active_%': active * 100,
                    'slot_norm_mean': slot_norms.mean().item(),
                    'ptr_coverage_%': len(ptr_visited) / M * 100,
                })

    # ── Pretty print ──
    print(f'\n{"="*65}')
    print(f'  STATE HEALTH PROBE ({n_steps} streaming steps) {label}')
    print(f'{"="*65}')
    print(f'  {"step":>6s} | {"rank90":>6s} | {"adj_cos":>7s} | {"active%":>7s} | {"norm":>8s} | {"ptr_cov%":>8s}')
    print(f'  {"-"*6} | {"-"*6} | {"-"*7} | {"-"*7} | {"-"*8} | {"-"*8}')
    for h in history:
        print(f'  {h["step"]:6d} | {h["rank_90"]:6d} | {h["adj_cos"]:7.4f} | '
              f'{h["active_%"]:6.1f}% | {h["slot_norm_mean"]:8.3f} | {h["ptr_coverage_%"]:7.1f}%')

    # Pass/fail gates
    last = history[-1]
    print(f'\n  PASS/FAIL GATES:')
    gates = {
        'rank_90 > 2': last['rank_90'] > 2,
        'adj_cos < 0.95': last['adj_cos'] < 0.95,
        'ptr_coverage > 30%': last['ptr_coverage_%'] > 30,
        'active_slots > 50%': last['active_%'] > 50,
    }
    for gate, passed in gates.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'    [{status:4s}] {gate}')

    return history


# ═══════════════════════════════════════════════════════════════
#  Training helper (for multi-checkpoint probing)
# ═══════════════════════════════════════════════════════════════

def train_steps(model, device, steps, lr=1e-3, period=8, seq=32, batch=32):
    """Quick training on synthetic periodic data. Returns final state."""
    from bench_fast_memory import generate_repeating_pattern

    data, mask = generate_repeating_pattern(
        B=batch, length=seq, period=period, seed=42
    )
    data = data.to(device)
    mask = mask.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    state = None

    for step in range(steps):
        x = data[:, :-1]
        y = data[:, 1:]
        m = mask[:, 1:]

        logits, state = model(x, state=state)
        state = {k: v.detach() for k, v in state.items()}

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction='none')
        loss = (loss.view(y.shape) * m).sum() / m.sum().clamp(min=1)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 200 == 0 or step == 0:
            acc = ((logits.argmax(-1) == y).float() * m).sum() / m.sum()
            print(f'    train step {step+1:5d} | loss={loss.item():.4f} | acc={acc.item()*100:.1f}%')

    return state


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def create_model(device, gated_write=False, write_mode='accumulate', kernel_mode='vshape',
                 pointer_mode='sequential', N=2):
    """Create INSTNCT with bench_fast_memory defaults."""
    return INSTNCT(
        M=256, hidden_dim=512, slot_dim=64, N=N, R=1,
        embed_mode=True, kernel_mode=kernel_mode,
        embed_encoding='bitlift', output_encoding='lowrank_c19',
        expert_weighting=False, checkpoint_chunks=0,
        bb_enabled=False, io_split_mode='off',
        pointer_mode=pointer_mode,
        gated_write=gated_write, write_mode=write_mode,
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description='INSTNCT pipeline signal probe')
    parser.add_argument('--train-steps', type=int, nargs='+', default=[],
                        help='Training checkpoints to probe (e.g., 500 3000)')
    parser.add_argument('--gated-write', action='store_true',
                        help='Enable mini head gated write (anti-blob)')
    parser.add_argument('--write-mode', default='accumulate', choices=['accumulate', 'replace'],
                        help='Write mode: accumulate (scatter_add) or replace (HDD-style)')
    parser.add_argument('--kernel-mode', default='vshape', choices=['vshape', 'topk', 'uniform', 'gaussian', 'dotprod'],
                        help='Read kernel: vshape (local), topk (global content-based), etc.')
    parser.add_argument('--pointer-mode', default='sequential', choices=['sequential', 'learned', 'pilot'],
                        help='Pointer mode: sequential (+1), learned (dir+mag), pilot (walk + novelty jump).')
    parser.add_argument('--n-experts', type=int, default=2, help='Number of experts')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'Device: {device}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Probe at t=0 (fresh init) ──
    model = create_model(device, gated_write=args.gated_write,
                          write_mode=args.write_mode, kernel_mode=args.kernel_mode,
                         pointer_mode=args.pointer_mode,
                         N=args.n_experts)
    if args.gated_write:
        print('  ** GATED WRITE (mini head) enabled **')
    if args.write_mode == 'replace':
        print('  ** HDD-STYLE REPLACE WRITE enabled **')
    if args.kernel_mode != 'vshape':
        print(f'  ** KERNEL: {args.kernel_mode} **')
    if args.pointer_mode != 'sequential':
        print(f'  ** POINTER: {args.pointer_mode} **')
    print(f'  ** N={args.n_experts} experts **')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params:,} params')

    probe_forward_stream(model, device, label='[t=0 fresh init]')
    probe_backward(model, device, label='[t=0 fresh init]')
    probe_state_health(model, device, n_steps=200, label='[t=0 fresh init]')

    # ── Probe at training checkpoints ──
    if args.train_steps:
        checkpoints = sorted(args.train_steps)
        prev_steps = 0
        state = None

        for ckpt_step in checkpoints:
            delta = ckpt_step - prev_steps
            if delta > 0:
                print(f'\n{"#"*65}')
                print(f'  TRAINING {delta} steps (total: {ckpt_step})...')
                print(f'{"#"*65}')
                state = train_steps(model, device, steps=delta)
                prev_steps = ckpt_step

            tag = f'[t={ckpt_step} after training]'
            probe_forward_stream(model, device, state=state, label=tag)
            probe_backward(model, device, label=tag)
            probe_state_health(model, device, n_steps=200, label=tag)

    print(f'\n{"="*65}')
    print('  ALL PROBES COMPLETE')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
