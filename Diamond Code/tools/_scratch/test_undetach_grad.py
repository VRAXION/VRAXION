"""
test_undetach_grad.py -- deterministic gradient flow test for #97

Single forward+backward pass. Checks whether gradient reaches lcx_values
in detached vs un-detached config. No training, no randomness beyond seed.
"""
import sys, os
sys.path.insert(0, r'S:\AI\work\VRAXION_DEV\Diamond Code')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ── Minimal setup ────────────────────────────────────────────────────────────
D      = 128
SLOTS  = 100
TOP_K  = 2
B      = 4

keys   = torch.randn(SLOTS, D)
query  = torch.randn(B, D)

def run(detach_values, label):
    values = torch.randn(SLOTS, D, requires_grad=True)

    scores = query @ keys.T                          # [B, S]
    topk_scores, topk_idx = scores.topk(TOP_K, dim=-1)
    weights = F.softmax(topk_scores, dim=-1)         # [B, K]

    if detach_values:
        topk_vals = values.detach()[topk_idx]        # gradient BLOCKED
    else:
        topk_vals = values[topk_idx]                 # gradient FLOWS

    context = (weights.unsqueeze(-1) * topk_vals).sum(dim=1)  # [B, D]
    loss = context.sum()

    try:
        loss.backward()
        grad = values.grad
    except RuntimeError:
        grad = None  # no grad_fn at all — detach worked

    has_grad = grad is not None and grad.abs().sum().item() > 0

    print(f'  {label}:')
    print(f'    values.grad is None: {grad is None}')
    if grad is not None:
        print(f'    grad.abs().sum():    {grad.abs().sum().item():.6f}')
        print(f'    grad nonzero slots: {(grad.abs().sum(dim=1) > 0).sum().item()} / {SLOTS}')
    print(f'    GRADIENT FLOWS: {has_grad}')
    print()
    return has_grad

print('=' * 50)
print('test_undetach_grad -- gradient flow check (#97)')
print('=' * 50)
print()

a = run(detach_values=True,  label='A -- values.detach() [current]')
b = run(detach_values=False, label='B -- values (un-detached) [proposed]')

print('=' * 50)
print(f'  A detached:   grad flows = {a}  (expected: False)')
print(f'  B undetached: grad flows = {b}  (expected: True)')
if not a and b:
    print('  PASS -- un-detach opens gradient path as expected')
else:
    print('  UNEXPECTED -- check logic')
print('=' * 50)
