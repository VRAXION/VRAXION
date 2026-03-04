"""Causal validation of learnable C — three hard tests proposed by GPT-4o.

Test 1: C-SWAP
  Train add1 for N steps, train reverse for N steps.
  Then swap their learned C values (add1's C → reverse model, and vice versa).
  If accuracy DROPS after swap → C encodes task-specific knowledge.

Test 2: C-FREEZE TIMING
  Compare three regimes on add1 (1000 steps):
    a) C learnable the whole time
    b) C learnable for first 200 steps, then frozen
    c) C fixed at π the whole time
  If (a) ≈ (b) >> (c) → early C adaptation is sufficient.

Test 3: SEED ROBUSTNESS
  Run add1 and reverse on 3 different seeds.
  Check if C_input / C_hidden movement DIRECTIONS are consistent.
  If directions hold across seeds → the signal is real, not noise.

Usage:
    python tests/test_c19_causal.py              # all 3 tests
    python tests/test_c19_causal.py --test swap   # just one
"""

import sys
import time
import math
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from instnct import INSTNCT, _C_from_raw, _C19_C_MIN, _C19_C_MAX

# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

BATCH = 8
SEQ_LEN = 32
LR = 1e-3
STEPS = 500
SEED = 42

MODEL_CFG = dict(
    M=32, hidden_dim=64, slot_dim=16, N=1, R=1,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift',
    output_encoding='lowrank_c19',
    checkpoint_chunks=0,
)

C_INIT = math.pi

# ═══════════════════════════════════════════════════════════════
#  Tasks
# ═══════════════════════════════════════════════════════════════

def make_add1_batch(batch, seq_len, rng):
    x = rng.randint(0, 256, size=(batch, seq_len)).astype(np.int64)
    y = (x + 1) % 256
    return x, y

def make_reverse_batch(batch, seq_len, rng):
    block = 8
    n_blocks = seq_len // block
    actual = n_blocks * block
    x = np.zeros((batch, actual), dtype=np.int64)
    y = np.zeros((batch, actual), dtype=np.int64)
    for i in range(batch):
        for b in range(n_blocks):
            bl = rng.randint(0, 256, size=block, dtype=np.uint8)
            x[i, b*block:(b+1)*block] = bl
            y[i, b*block:(b+1)*block] = bl[::-1]
    return x, y

TASKS = {'add1': make_add1_batch, 'reverse': make_reverse_batch}

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def get_c_values(model):
    with torch.no_grad():
        c_inp = _C_from_raw(model.c19_C_input)
        c_hid = _C_from_raw(model.c19_C_hidden)
    return {
        'c_input_mean': c_inp.mean().item(),
        'c_hidden_mean': c_hid.mean().item(),
        'c_input_std': c_inp.std().item(),
        'c_hidden_std': c_hid.std().item(),
    }

def evaluate(model, task_fn, n_batches=8, device='cpu'):
    model.eval()
    rng = np.random.RandomState(9999)
    total_correct = 0
    total_count = 0
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x_np, y_np = task_fn(BATCH, SEQ_LEN, rng)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            out, _ = model(x)
            logits = out.view(-1, 256)
            targets = y.view(-1)
            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()
            total_loss += loss.item()
    return {
        'acc': total_correct / total_count,
        'loss': total_loss / n_batches,
    }

def train_model(task_fn, steps, seed, device='cpu', freeze_c_after=None, fix_c=False):
    """Train a model, optionally freezing C after N steps or fixing C entirely."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = INSTNCT(**MODEL_CFG).to(device)

    if fix_c:
        model.c19_C_input.requires_grad_(False)
        model.c19_C_hidden.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(seed)

    c_trajectory = [(0, get_c_values(model))]

    for step in range(1, steps + 1):
        # freeze C at the specified step
        if freeze_c_after is not None and step == freeze_c_after + 1:
            model.c19_C_input.requires_grad_(False)
            model.c19_C_hidden.requires_grad_(False)
            # rebuild optimizer without frozen params
            opt = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=LR
            )

        model.train()
        x_np, y_np = task_fn(BATCH, SEQ_LEN, rng)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        out, _ = model(x)
        logits = out.view(-1, 256)
        targets = y.view(-1)
        loss = F.cross_entropy(logits, targets)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if step % 100 == 0 or step == steps:
            c_trajectory.append((step, get_c_values(model)))

    return model, c_trajectory


# ═══════════════════════════════════════════════════════════════
#  TEST 1: C-SWAP
# ═══════════════════════════════════════════════════════════════

def test_swap():
    print("\n" + "=" * 78)
    print("  TEST 1: C-SWAP (cross-task C transplant)")
    print("=" * 78)

    device = 'cpu'
    t0 = time.perf_counter()

    # Train both tasks
    print("  Training add1...")
    model_add1, traj_add1 = train_model(TASKS['add1'], STEPS, SEED, device)
    print("  Training reverse...")
    model_rev, traj_rev = train_model(TASKS['reverse'], STEPS, SEED, device)

    # Evaluate BEFORE swap
    acc_add1_native = evaluate(model_add1, TASKS['add1'], device=device)
    acc_rev_native = evaluate(model_rev, TASKS['reverse'], device=device)

    print(f"\n  BEFORE swap:")
    print(f"    add1  model on add1  task: acc={acc_add1_native['acc']*100:.1f}%  loss={acc_add1_native['loss']:.4f}")
    print(f"    reverse model on reverse task: acc={acc_rev_native['acc']*100:.1f}%  loss={acc_rev_native['loss']:.4f}")

    # Swap C values
    c_add1_input = model_add1.c19_C_input.data.clone()
    c_add1_hidden = model_add1.c19_C_hidden.data.clone()
    c_rev_input = model_rev.c19_C_input.data.clone()
    c_rev_hidden = model_rev.c19_C_hidden.data.clone()

    # add1 model gets reverse's C
    model_add1.c19_C_input.data.copy_(c_rev_input)
    model_add1.c19_C_hidden.data.copy_(c_rev_hidden)
    # reverse model gets add1's C
    model_rev.c19_C_input.data.copy_(c_add1_input)
    model_rev.c19_C_hidden.data.copy_(c_add1_hidden)

    # Evaluate AFTER swap
    acc_add1_swapped = evaluate(model_add1, TASKS['add1'], device=device)
    acc_rev_swapped = evaluate(model_rev, TASKS['reverse'], device=device)

    print(f"\n  AFTER swap (add1 model has reverse's C, and vice versa):")
    print(f"    add1  model on add1  task: acc={acc_add1_swapped['acc']*100:.1f}%  loss={acc_add1_swapped['loss']:.4f}")
    print(f"    reverse model on reverse task: acc={acc_rev_swapped['acc']*100:.1f}%  loss={acc_rev_swapped['loss']:.4f}")

    # Compute degradation
    add1_loss_delta = acc_add1_swapped['loss'] - acc_add1_native['loss']
    rev_loss_delta = acc_rev_swapped['loss'] - acc_rev_native['loss']
    add1_acc_delta = acc_add1_swapped['acc'] - acc_add1_native['acc']
    rev_acc_delta = acc_rev_swapped['acc'] - acc_rev_native['acc']

    print(f"\n  DEGRADATION after swap:")
    print(f"    add1:    Δloss={add1_loss_delta:+.4f}  Δacc={add1_acc_delta*100:+.1f}%")
    print(f"    reverse: Δloss={rev_loss_delta:+.4f}  Δacc={rev_acc_delta*100:+.1f}%")

    # Also test: reset C to init (π) — "ablation"
    model_add1.c19_C_input.data.copy_(c_add1_input)  # restore first
    model_add1.c19_C_hidden.data.copy_(c_add1_hidden)
    # Now save and reset to init
    raw_init_val = model_add1.c19_C_input.data.clone()
    # We need the raw value that maps to π via sigmoid. Compute it.
    # _C_from_raw uses: C_MIN + (C_MAX - C_MIN) * sigmoid(raw)
    # So raw for π: sigmoid(raw) = (π - C_MIN) / (C_MAX - C_MIN)
    target_sig = (C_INIT - _C19_C_MIN) / (_C19_C_MAX - _C19_C_MIN)
    raw_pi = math.log(target_sig / (1 - target_sig))  # logit
    model_add1.c19_C_input.data.fill_(raw_pi)
    model_add1.c19_C_hidden.data.fill_(raw_pi)
    acc_add1_reset = evaluate(model_add1, TASKS['add1'], device=device)
    print(f"\n  ABLATION (add1 model with C reset to π):")
    print(f"    add1 model on add1 task: acc={acc_add1_reset['acc']*100:.1f}%  loss={acc_add1_reset['loss']:.4f}")
    reset_loss_delta = acc_add1_reset['loss'] - acc_add1_native['loss']
    print(f"    Δloss vs native: {reset_loss_delta:+.4f}")

    elapsed = time.perf_counter() - t0
    # Verdict
    swap_hurts = add1_loss_delta > 0.01 or rev_loss_delta > 0.01
    verdict = "PASS — swap degrades performance → C is task-specific" if swap_hurts else \
              "WEAK — swap didn't hurt much → C may not be critical yet"
    print(f"\n  VERDICT: {verdict}  [{elapsed:.1f}s]")
    return swap_hurts


# ═══════════════════════════════════════════════════════════════
#  TEST 2: C-FREEZE TIMING
# ═══════════════════════════════════════════════════════════════

def test_freeze_timing():
    print("\n" + "=" * 78)
    print("  TEST 2: C-FREEZE TIMING (learnable vs early-freeze vs fixed)")
    print("=" * 78)

    device = 'cpu'
    steps = 1000
    t0 = time.perf_counter()

    configs = [
        ("learnable (full)", dict(steps=steps, seed=SEED, freeze_c_after=None, fix_c=False)),
        ("freeze@200",       dict(steps=steps, seed=SEED, freeze_c_after=200,  fix_c=False)),
        ("freeze@50",        dict(steps=steps, seed=SEED, freeze_c_after=50,   fix_c=False)),
        ("fixed C=π",        dict(steps=steps, seed=SEED, freeze_c_after=None, fix_c=True)),
    ]

    results = []
    for name, kwargs in configs:
        print(f"\n  Training: {name}...")
        model, traj = train_model(TASKS['add1'], device=device, **kwargs)
        ev = evaluate(model, TASKS['add1'], device=device)
        c_vals = get_c_values(model)
        results.append((name, ev, c_vals, traj))
        print(f"    acc={ev['acc']*100:.1f}%  loss={ev['loss']:.4f}"
              f"  C_in={c_vals['c_input_mean']:.4f}  C_hid={c_vals['c_hidden_mean']:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"\n  COMPARISON (add1, {steps} steps):")
    print(f"  {'Regime':<20} {'Acc':>8} {'Loss':>8} {'C_input':>8} {'C_hidden':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, ev, c_vals, _ in results:
        print(f"  {name:<20} {ev['acc']*100:>7.1f}% {ev['loss']:>8.4f}"
              f" {c_vals['c_input_mean']:>8.4f} {c_vals['c_hidden_mean']:>8.4f}")

    # Check: learnable > fixed
    loss_learnable = results[0][1]['loss']
    loss_fixed = results[-1][1]['loss']
    learnable_wins = loss_learnable < loss_fixed - 0.05
    # Check: early freeze is close to full learnable
    loss_freeze200 = results[1][1]['loss']
    early_close = abs(loss_freeze200 - loss_learnable) < abs(loss_fixed - loss_learnable) * 0.5

    verdict_parts = []
    if learnable_wins:
        verdict_parts.append("learnable C beats fixed C")
    else:
        verdict_parts.append("learnable C ≈ fixed C (no clear advantage yet)")
    if early_close:
        verdict_parts.append("early freeze is viable")
    else:
        verdict_parts.append("C needs to keep learning")

    print(f"\n  VERDICT: {' + '.join(verdict_parts)}  [{elapsed:.1f}s]")
    return learnable_wins


# ═══════════════════════════════════════════════════════════════
#  TEST 3: SEED ROBUSTNESS
# ═══════════════════════════════════════════════════════════════

def test_seed_robustness():
    print("\n" + "=" * 78)
    print("  TEST 3: SEED ROBUSTNESS (3 seeds, direction stability)")
    print("=" * 78)

    device = 'cpu'
    seeds = [42, 123, 7]
    t0 = time.perf_counter()

    task_directions = {}
    for task_name, task_fn in TASKS.items():
        directions = []
        print(f"\n  Task: {task_name}")
        for seed in seeds:
            model, traj = train_model(task_fn, STEPS, seed, device)
            c_init = traj[0][1]
            c_final = traj[-1][1]
            d_in = c_final['c_input_mean'] - c_init['c_input_mean']
            d_hid = c_final['c_hidden_mean'] - c_init['c_hidden_mean']
            directions.append((d_in, d_hid))
            sign_in = '+' if d_in > 0 else '-'
            sign_hid = '+' if d_hid > 0 else '-'
            print(f"    seed={seed:>3d}: ΔC_input={d_in:+.4f} ({sign_in})"
                  f"  ΔC_hidden={d_hid:+.4f} ({sign_hid})"
                  f"  C_in_final={c_final['c_input_mean']:.4f}"
                  f"  C_hid_final={c_final['c_hidden_mean']:.4f}")

        task_directions[task_name] = directions

    elapsed = time.perf_counter() - t0

    # Check direction consistency
    print(f"\n  DIRECTION CONSISTENCY:")
    all_consistent = True
    for task_name, dirs in task_directions.items():
        # Check if all seeds agree on sign of delta
        in_signs = [1 if d[0] > 0.001 else (-1 if d[0] < -0.001 else 0) for d in dirs]
        hid_signs = [1 if d[1] > 0.001 else (-1 if d[1] < -0.001 else 0) for d in dirs]

        # Filter out near-zero (ambiguous)
        in_nonzero = [s for s in in_signs if s != 0]
        hid_nonzero = [s for s in hid_signs if s != 0]

        in_consistent = len(set(in_nonzero)) <= 1 if in_nonzero else True
        hid_consistent = len(set(hid_nonzero)) <= 1 if hid_nonzero else True

        in_dir = {1: '↑', -1: '↓', 0: '~'}.get(in_nonzero[0] if in_nonzero else 0, '~')
        hid_dir = {1: '↑', -1: '↓', 0: '~'}.get(hid_nonzero[0] if hid_nonzero else 0, '~')

        status_in = "✓" if in_consistent else "✗"
        status_hid = "✓" if hid_consistent else "✗"

        print(f"    {task_name:<12} C_input: {in_dir} [{status_in}]  C_hidden: {hid_dir} [{status_hid}]"
              f"  (signs_in={in_signs} signs_hid={hid_signs})")

        if not in_consistent or not hid_consistent:
            all_consistent = False

    # Also check: add1 and reverse have DIFFERENT direction profiles
    if 'add1' in task_directions and 'reverse' in task_directions:
        add1_mean_in = np.mean([d[0] for d in task_directions['add1']])
        add1_mean_hid = np.mean([d[1] for d in task_directions['add1']])
        rev_mean_in = np.mean([d[0] for d in task_directions['reverse']])
        rev_mean_hid = np.mean([d[1] for d in task_directions['reverse']])
        tasks_differ = (np.sign(add1_mean_in) != np.sign(rev_mean_in) or
                       np.sign(add1_mean_hid) != np.sign(rev_mean_hid))
        print(f"\n    add1 vs reverse profiles differ: {'✓ YES' if tasks_differ else '✗ NO'}")
        print(f"      add1:    mean ΔC_in={add1_mean_in:+.4f}  mean ΔC_hid={add1_mean_hid:+.4f}")
        print(f"      reverse: mean ΔC_in={rev_mean_in:+.4f}  mean ΔC_hid={rev_mean_hid:+.4f}")
    else:
        tasks_differ = True

    verdict = "PASS — directions are seed-robust" if all_consistent and tasks_differ else \
              "PARTIAL — some inconsistency" if tasks_differ else \
              "FAIL — directions are not stable"
    print(f"\n  VERDICT: {verdict}  [{elapsed:.1f}s]")
    return all_consistent


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Causal C validation tests")
    parser.add_argument('--test', type=str, default='all',
                        choices=['swap', 'freeze', 'seed', 'all'])
    args = parser.parse_args()

    print("=" * 78)
    print("  CAUSAL C VALIDATION — Hard Tests")
    print("=" * 78)

    results = {}
    if args.test in ('swap', 'all'):
        results['swap'] = test_swap()
    if args.test in ('freeze', 'all'):
        results['freeze'] = test_freeze_timing()
    if args.test in ('seed', 'all'):
        results['seed'] = test_seed_robustness()

    print("\n" + "=" * 78)
    print("  FINAL SCORECARD")
    print("=" * 78)
    for name, passed in results.items():
        print(f"    {name:<15} {'PASS ✓' if passed else 'FAIL/WEAK ✗'}")
    all_pass = all(results.values())
    print(f"\n  {'ALL TESTS PASSED — C is causally task-specific.' if all_pass else 'Some tests did not pass — see details above.'}")
    print("=" * 78)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
