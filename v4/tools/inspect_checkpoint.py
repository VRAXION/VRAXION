"""Checkpoint V2 inspector.

Usage:
    python tools/inspect_checkpoint.py --ckpt training_output/ckpt_latest.pt
    python tools/inspect_checkpoint.py --ckpt a.pt --against b.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_TOP = (
    'ckpt_version', 'run_id', 'step', 'best_loss', 'timestamp_utc',
    'model', 'optimizer',
    'train_config_resolved', 'model_config_resolved',
    'data_state', 'sequence_state', 'rng_state', 'env',
)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f'Expected dict checkpoint, got {type(ckpt).__name__}')
    return ckpt


def _manifest_signature(manifest: list[dict[str, Any]]) -> list[tuple[str, int, int, str]]:
    sig: list[tuple[str, int, int, str]] = []
    for it in manifest:
        sig.append((
            str(it.get('path', '')),
            int(it.get('size', -1)),
            int(it.get('mtime_ns', -1)),
            str(it.get('sha256_head_1mb', '')),
        ))
    return sig


def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def _validate(ckpt: dict[str, Any]) -> list[str]:
    msgs: list[str] = []
    for k in REQUIRED_TOP:
        if k not in ckpt:
            msgs.append(f'missing top-level key: {k}')
    model_obj = ckpt.get('model', {})
    for k in ('type', 'module', 'class_name', 'build_spec', 'state_dict'):
        if isinstance(model_obj, dict) and k not in model_obj:
            msgs.append(f'missing model.{k}')
    opt_obj = ckpt.get('optimizer', {})
    for k in ('class_name', 'state_dict'):
        if isinstance(opt_obj, dict) and k not in opt_obj:
            msgs.append(f'missing optimizer.{k}')
    if isinstance(ckpt.get('rng_state'), dict) and 'dataset_rng_state' not in ckpt['rng_state']:
        msgs.append('missing rng_state.dataset_rng_state')
    return msgs


def _summarize(path: Path, ckpt: dict[str, Any]) -> None:
    model = ckpt.get('model', {})
    data_state = ckpt.get('data_state', {})
    seq_state = ckpt.get('sequence_state', {})
    rng_state = ckpt.get('rng_state', {})

    print(f'Checkpoint: {path}')
    print(f'  ckpt_version : {ckpt.get("ckpt_version")}')
    print(f'  run_id       : {ckpt.get("run_id")}')
    print(f'  step         : {ckpt.get("step")}')
    print(f'  best_loss    : {ckpt.get("best_loss")}')
    print(f'  timestamp    : {ckpt.get("timestamp_utc")}')
    print(f'  model.type   : {model.get("type")}')
    print(f'  model.class  : {model.get("class_name")}')
    print(f'  optimizer    : {ckpt.get("optimizer", {}).get("class_name")}')
    print(f'  data_dir     : {data_state.get("data_dir")}')
    print(f'  seq_len      : {data_state.get("seq_len")}')
    print(f'  batch_size   : {data_state.get("batch_size")}')
    print(f'  sequential   : {data_state.get("sequential")}')
    print(f'  manifest_n   : {len(data_state.get("file_manifest", []))}')
    print(f'  seq_state_k  : {sorted(list(seq_state.keys()))}')
    print(f'  rng_keys     : {sorted(list(rng_state.keys()))}')


def _compare(a: dict[str, Any], b: dict[str, Any]) -> None:
    a_spec = a['model']['build_spec']
    b_spec = b['model']['build_spec']
    if a_spec == b_spec:
        print('build_spec: MATCH')
    else:
        print('build_spec: DIFF')
        a_keys = set(a_spec.keys())
        b_keys = set(b_spec.keys())
        for k in sorted(a_keys | b_keys):
            av = a_spec.get(k, '<missing>')
            bv = b_spec.get(k, '<missing>')
            if av != bv:
                print(f'  - {k}: A={av!r}  B={bv!r}')

    am = _manifest_signature(a.get('data_state', {}).get('file_manifest', []))
    bm = _manifest_signature(b.get('data_state', {}).get('file_manifest', []))
    print(f'manifest: {"MATCH" if am == bm else "DIFF"}')
    if am != bm:
        print(f'  A files={len(am)}  B files={len(bm)}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Inspect VRAXION checkpoint V2 payload.')
    ap.add_argument('--ckpt', required=True, help='Checkpoint to inspect')
    ap.add_argument('--against', default=None, help='Optional second checkpoint for diff')
    ap.add_argument('--print-build-spec', action='store_true', help='Print model.build_spec JSON')
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = _load(ckpt_path)
    problems = _validate(ckpt)

    _summarize(ckpt_path, ckpt)
    if problems:
        print('\nSchema issues:')
        for p in problems:
            print(f'  - {p}')
        return 2

    if args.print_build_spec:
        print('\nmodel.build_spec:')
        print(_pretty(ckpt['model']['build_spec']))

    if args.against:
        other_path = Path(args.against)
        other = _load(other_path)
        print('\n--- Compare ---')
        _compare(ckpt, other)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
