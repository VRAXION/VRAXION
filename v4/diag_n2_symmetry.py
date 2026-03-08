"""
N=2 Symmetry Collapse Diagnostic — All Pointer Modes

Tests sequential, learned, pilot pointer modes with N=2 experts.
Trains until plateau (loss improvement <0.01 over 50 steps, or max 500).
Logs every 10 steps: loss, per-expert metrics, grad delta.

Output: raw info dump for later refinement.
"""
import sys, time, torch, torch.nn as nn
sys.path.insert(0, r"S:\AI\_tmp\nightly_worktree\v4")
from model.instnct import INSTNCT

device = "cuda" if torch.cuda.is_available() else "cpu"
B, T = 64, 16
MAX_STEPS = 500
LOG_EVERY = 10
PLATEAU_WINDOW = 50
PLATEAU_THRESH = 0.01

POINTER_MODES = ['sequential', 'learned', 'pilot']

print(f"Device: {device}")
print(f"B={B}, T={T}, max_steps={MAX_STEPS}, plateau_window={PLATEAU_WINDOW}")
print(f"Pointer modes to test: {POINTER_MODES}")


def build_model(pointer_mode):
    m = INSTNCT(
        M=128,
        hidden_dim=4096,
        slot_dim=128,
        N=2,
        R=1,
        embed_mode=True,
        kernel_mode='vshape',
        pointer_mode=pointer_mode,
        write_mode='replace',
        embed_encoding='bitlift',
        output_encoding='lowrank_c19',
    ).to(device)
    return m


def grad_delta(model):
    """Compare gradient norms between expert 0 and expert 1."""
    expert_grads = {0: {}, 1: {}}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        for ei in (0, 1):
            if f'.{ei}.' in name:
                base = name.replace(f'.{ei}.', '.X.')
                expert_grads[ei][base] = param.grad.norm().item()
    shared = set(expert_grads[0].keys()) & set(expert_grads[1].keys())
    if not shared:
        return 0, 0.0, 0.0
    deltas = [abs(expert_grads[0][k] - expert_grads[1][k]) for k in shared]
    return len(shared), sum(deltas), max(deltas)


def run_mode(pointer_mode):
    print(f"\n{'#' * 70}")
    print(f"# POINTER MODE: {pointer_mode}")
    print(f"{'#' * 70}")

    model = build_model(pointer_mode)
    model.train()
    vocab = 256
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    best_loss = float('inf')
    t0 = time.perf_counter()

    for step in range(MAX_STEPS):
        x = torch.randint(0, vocab, (B, T), device=device)
        target = torch.randint(0, vocab, (B, T), device=device)

        is_log = (step % LOG_EVERY == 0) or (step == MAX_STEPS - 1)
        model._diag_enabled = is_log

        result = model(x)
        out = result[0] if isinstance(result, tuple) else result
        if out.dim() == 3:
            loss = criterion(out.reshape(-1, out.size(-1)), target.reshape(-1))
        else:
            loss = criterion(out, target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        loss_val = loss.item()
        loss_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if is_log:
            d = model._diag
            n_pairs, g_total, g_max = grad_delta(model)

            # Collect per-expert metrics
            metrics = {}
            for key in ['alpha', 'input_norm', 'ring_signal_norm', 'blended_norm', 'hidden_norm']:
                for suffix in ['_mean', '']:
                    v0 = d.get(f'{key}_0{suffix}')
                    v1 = d.get(f'{key}_1{suffix}')
                    if v0 is not None and v1 is not None:
                        metrics[key] = (v0, v1)
                        break
                    v0 = d.get(f'{key}_0')
                    v1 = d.get(f'{key}_1')
                    if v0 is not None and v1 is not None:
                        metrics[key] = (v0, v1)
                        break

            parts = [f"step {step:3d} | loss={loss_val:.4f} | grad_delta={g_max:.6f}"]
            for k, (v0, v1) in metrics.items():
                short = k.replace('_norm', '').replace('_signal', '_sig')
                parts.append(f"{short}:[{v0:.3f},{v1:.3f}]")
            print("  " + " | ".join(parts))

        optimizer.step()

        # Plateau check
        if step >= PLATEAU_WINDOW:
            old_avg = sum(loss_history[step - PLATEAU_WINDOW:step - PLATEAU_WINDOW + 10]) / 10
            new_avg = sum(loss_history[step - 9:step + 1]) / 10
            if old_avg - new_avg < PLATEAU_THRESH:
                elapsed = time.perf_counter() - t0
                print(f"  ** PLATEAU at step {step}: old_avg={old_avg:.4f}, new_avg={new_avg:.4f}, "
                      f"delta={old_avg - new_avg:.4f} < {PLATEAU_THRESH}")
                print(f"  ** Elapsed: {elapsed:.1f}s ({step / elapsed:.1f} step/s)")
                break

    elapsed = time.perf_counter() - t0
    print(f"\n  SUMMARY ({pointer_mode}):")
    print(f"  Steps: {min(step + 1, MAX_STEPS)}, Time: {elapsed:.1f}s, Best loss: {best_loss:.4f}")

    # Final expert divergence check with diag ON
    model._diag_enabled = True
    with torch.no_grad():
        x = torch.randint(0, vocab, (B, T), device=device)
        result = model(x)
    d = model._diag

    print(f"  Final diag keys _0: {len([k for k in d if k.endswith('_0')])}, "
          f"_1: {len([k for k in d if k.endswith('_1')])}")

    # Print all scalar expert pairs
    keys_0 = sorted([k for k in d if k.endswith('_0')])
    for k0 in keys_0:
        base = k0[:-2]
        v0, v1 = d.get(k0), d.get(base + '_1')
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            delta = abs(v0 - v1)
            sym = "SAME" if delta < 1e-4 else ("close" if delta < 0.01 else "DIVERGED")
            print(f"    {base:<35} {v0:>10.4f}  {v1:>10.4f}  d={delta:.4f}  {sym}")

    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()


# ── Run all modes ──
print(f"\n{'=' * 70}")
print(f"N=2 SYMMETRY COLLAPSE TEST — ALL POINTER MODES")
print(f"{'=' * 70}")

for mode in POINTER_MODES:
    run_mode(mode)

print(f"\n{'=' * 70}")
print("ALL DONE")
print("=" * 70)
