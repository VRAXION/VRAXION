"""Validate nightly proxy_overlay correctness against dense replace writes.

Checks:
  - fixed-batch fp32 loss/logit equivalence with autocast disabled
  - 16-step deterministic forward-state equivalence (logits + ring/ptr/hidden)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from profile_sweep_step_wikitext import V4_ROOT, _load_dataset
from sweep_c19_core_geometry_wikitext import _set_determinism, build_model
from train import func_maskloss_ce


def _default_output_path() -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'dev_notes' / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'{Path(__file__).stem}_{stamp}.json'


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _run_forward(seed: int, xb: torch.Tensor, yb: torch.Tensor, mask: torch.Tensor, replace_impl: str):
    _set_determinism(seed)
    model = build_model(seed, replace_impl=replace_impl)
    model.train()
    with torch.amp.autocast('cuda', enabled=False):
        logits, state = model(xb, state=None)
        _, loss = func_maskloss_ce(logits, yb, mask)
    return logits.detach(), float(loss.detach().item()), {
        'ring': state['ring'].detach(),
        'ptr': state['ptr'].detach(),
        'hidden': state['hidden'].detach(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--chunk-steps', type=int, default=16)
    parser.add_argument('--json-out', type=str, default='')
    args = parser.parse_args()

    _set_determinism(args.seed)
    _, dataset = _load_dataset(args.seq, args.seed)
    dataset.rng = __import__('numpy').random.default_rng(args.seed)
    xb, yb, mask = dataset.sample_batch(args.batch, 'cuda')

    dense_logits, dense_loss, dense_state = _run_forward(args.seed, xb, yb, mask, 'dense')
    overlay_logits, overlay_loss, overlay_state = _run_forward(args.seed, xb, yb, mask, 'proxy_overlay')

    chunk_xb = xb[:, :args.chunk_steps].contiguous()
    chunk_yb = yb[:, :args.chunk_steps].contiguous()
    chunk_mask = mask[:, :args.chunk_steps].contiguous()
    dense_chunk_logits, _, dense_chunk_state = _run_forward(args.seed, chunk_xb, chunk_yb, chunk_mask, 'dense')
    overlay_chunk_logits, _, overlay_chunk_state = _run_forward(args.seed, chunk_xb, chunk_yb, chunk_mask, 'proxy_overlay')

    payload = {
        'script': Path(__file__).name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': args.seed,
            'batch': args.batch,
            'seq': args.seq,
            'chunk_steps': args.chunk_steps,
        },
        'fixed_batch': {
            'dense_loss': dense_loss,
            'overlay_loss': overlay_loss,
            'loss_abs_diff': abs(dense_loss - overlay_loss),
            'logit_max_abs_diff': _max_abs_diff(dense_logits, overlay_logits),
        },
        'chunk_behavior': {
            'logit_max_abs_diff': _max_abs_diff(dense_chunk_logits, overlay_chunk_logits),
            'ring_max_abs_diff': _max_abs_diff(dense_chunk_state['ring'], overlay_chunk_state['ring']),
            'ptr_max_abs_diff': _max_abs_diff(dense_chunk_state['ptr'], overlay_chunk_state['ptr']),
            'hidden_max_abs_diff': _max_abs_diff(dense_chunk_state['hidden'], overlay_chunk_state['hidden']),
        },
    }

    output_path = Path(args.json_out) if args.json_out else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('=== Proxy Overlay Validation ===')
    print(f'Saved JSON: {output_path}')
    print(
        f'fixed_batch: loss_abs_diff={payload["fixed_batch"]["loss_abs_diff"]:.8f}  '
        f'logit_max_abs_diff={payload["fixed_batch"]["logit_max_abs_diff"]:.8f}'
    )
    print(
        f'chunk_behavior: logit={payload["chunk_behavior"]["logit_max_abs_diff"]:.8f}  '
        f'ring={payload["chunk_behavior"]["ring_max_abs_diff"]:.8f}  '
        f'ptr={payload["chunk_behavior"]["ptr_max_abs_diff"]:.8f}  '
        f'hidden={payload["chunk_behavior"]["hidden_max_abs_diff"]:.8f}'
    )


if __name__ == '__main__':
    main()
