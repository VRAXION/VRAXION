"""Patch checkpoint: zero-init last BN layer for gate-free LCX (v3.5).

Zeroes lcx_bn_layers.2.weight and lcx_bn_layers.2.bias in the checkpoint,
so LCX output starts at zero but gradients flow freely (GPT-2 trick).
The trained LCX memory content and BN layers 0,1 are PRESERVED.
"""
import sys, os, torch

CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'curriculum_v2')
LATEST = os.path.join(CKPT_DIR, 'checkpoint_latest.pt')

def patch(path):
    print(f'Loading: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    step = ckpt.get('step', '?')

    # Show current values
    w_key = 'lcx_bn_layers.2.weight'
    b_key = 'lcx_bn_layers.2.bias'

    if w_key not in sd:
        print(f'  ERROR: {w_key} not found in checkpoint')
        return

    w = sd[w_key]
    b = sd[b_key]
    print(f'  Step: {step}')
    print(f'  {w_key}: shape={list(w.shape)}, norm={w.norm().item():.4f}, mean={w.mean().item():.6f}')
    print(f'  {b_key}: shape={list(b.shape)}, norm={b.norm().item():.4f}, mean={b.mean().item():.6f}')

    # Also show BN layers 0,1 (these are PRESERVED)
    for i in range(2):
        wk = f'lcx_bn_layers.{i}.weight'
        if wk in sd:
            print(f'  {wk}: norm={sd[wk].norm().item():.4f} (KEPT)')

    # Zero the last layer
    sd[w_key] = torch.zeros_like(w)
    sd[b_key] = torch.zeros_like(b)
    print(f'\n  PATCHED: {w_key} and {b_key} zeroed')
    print(f'  LCX output = 0 at restart, will learn to open via gradient flow')

    # Save
    torch.save(ckpt, path)
    print(f'  Saved: {path}')
    print(f'  Size: {os.path.getsize(path) / 1024 / 1024:.1f} MB')

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else LATEST
    patch(target)
    print('\nDone. Restart training to use the patched checkpoint.')
