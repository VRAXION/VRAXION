#!/usr/bin/env python3
"""Rewrite AbsoluteHallway router_map to a Fibonacci/halving frequency distribution.

This implements the "colony frequency" version of the Fibonacci idea:
- Experts remain equal sized (no architecture change).
- Routing frequency becomes unequal via router_map (address -> expert_id).

Typical use:
1) Start a run to produce an initial checkpoint (resume=0, max_steps small).
2) Rewire router_map in that checkpoint with this script.
3) Resume training (resume=1) from the rewired checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rewire router_map to a halving (fibo-style) frequency map.")
    ap.add_argument("--in-ckpt", required=True, help="Input checkpoint .pt path.")
    ap.add_argument("--out-ckpt", default="", help="Optional output checkpoint path (default: in-place overwrite).")
    ap.add_argument("--buckets", type=int, default=7, help="Number of fibo/halving buckets (default: 7).")
    ap.add_argument(
        "--use-expert-ids",
        default="",
        help="Comma-separated expert IDs to use (default: 0..buckets-1).",
    )
    ap.add_argument(
        "--permute-seed",
        type=int,
        default=12345,
        help="Deterministic shuffle seed to spread bucket IDs across the ring.",
    )
    ap.add_argument(
        "--ratio",
        default="",
        help="Optional comma weights overriding halving counts (e.g. 0.55,0.34,0.11).",
    )
    ap.add_argument("--dry-run", type=int, default=0, choices=[0, 1], help="Print hist only; do not write.")
    ap.add_argument("--print-json", type=int, default=1, choices=[0, 1], help="Emit a small JSON summary to stdout.")
    return ap.parse_args()


def _load_ckpt(path: Path) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(obj)}")
    return obj


def _find_router_key(model_sd: Dict[str, Any]) -> str:
    if "router_map" in model_sd:
        return "router_map"
    keys = [k for k in model_sd.keys() if str(k).lower().endswith("router_map")]
    if len(keys) == 1:
        return str(keys[0])
    raise KeyError("Could not uniquely locate router_map key in checkpoint model state.")


def _infer_num_experts(ckpt: Dict[str, Any], model_sd: Dict[str, Any]) -> int:
    n = ckpt.get("num_experts")
    try:
        n_int = int(n)
        if n_int > 0:
            return n_int
    except Exception:
        pass

    # Fallback: infer from router_map max.
    rkey = _find_router_key(model_sd)
    rmap = model_sd[rkey]
    if not torch.is_tensor(rmap):
        raise TypeError("router_map is not a tensor; cannot infer num_experts")
    return int(rmap.max().item()) + 1


def _parse_expert_ids(raw: str, *, buckets: int) -> List[int]:
    if not str(raw).strip():
        return list(range(int(buckets)))
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if len(out) != int(buckets):
        raise ValueError(f"--use-expert-ids must have exactly {int(buckets)} IDs (got {len(out)})")
    return out


def _counts_halving(ring_len: int, buckets: int) -> List[int]:
    if ring_len <= 0:
        raise ValueError("ring_len must be > 0")
    if buckets <= 0:
        raise ValueError("buckets must be > 0")
    # Ideal weights: 1/2, 1/4, ... 1/2^(buckets)
    counts = [int(ring_len // (2 ** (i + 1))) for i in range(int(buckets))]
    used = int(sum(counts))
    if used > ring_len:
        raise ValueError("Internal error: used > ring_len")
    # Push any rounding remainder into the final shard bucket.
    counts[-1] += int(ring_len - used)
    return counts


def _parse_ratio(raw: str) -> List[float]:
    vals = [float(p.strip()) for p in str(raw).split(",") if p.strip()]
    if not vals:
        raise ValueError("empty ratio")
    if any((not torch.isfinite(torch.tensor(v)) or float(v) <= 0.0) for v in vals):
        raise ValueError("ratio weights must be finite positive numbers")
    total = float(sum(vals))
    if total <= 0.0:
        raise ValueError("ratio sum must be > 0")
    return [float(v) / total for v in vals]


def _counts_ratio(ring_len: int, weights: List[float]) -> List[int]:
    if ring_len <= 0:
        raise ValueError("ring_len must be > 0")
    if not weights:
        raise ValueError("weights must be non-empty")
    targets = [float(ring_len) * float(w) for w in weights]
    counts = [int(t) for t in targets]
    used = int(sum(counts))
    rem = int(ring_len - used)
    if rem > 0:
        frac = sorted(
            [(targets[i] - float(int(targets[i])), i) for i in range(len(targets))],
            reverse=True,
        )
        for j in range(rem):
            counts[frac[j % len(frac)][1]] += 1
    elif rem < 0:
        frac = sorted(
            [(targets[i] - float(int(targets[i])), i) for i in range(len(targets))],
            reverse=False,
        )
        need = int(-rem)
        j = 0
        while need > 0 and j < 10000:
            idx = frac[j % len(frac)][1]
            if counts[idx] > 0:
                counts[idx] -= 1
                need -= 1
            j += 1
    return counts


def _make_router_map(
    ring_len: int,
    expert_ids: List[int],
    permute_seed: int,
    *,
    counts_override: List[int] | None = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    buckets = len(expert_ids)
    counts = list(counts_override) if counts_override is not None else _counts_halving(int(ring_len), int(buckets))
    if len(counts) != buckets:
        raise ValueError("Internal error: counts/buckets mismatch")
    if int(sum(counts)) != int(ring_len):
        raise ValueError("Internal error: counts must sum to ring_len")

    ids: List[int] = []
    for eid, n in zip(expert_ids, counts, strict=True):
        ids.extend([int(eid)] * int(n))
    if len(ids) != int(ring_len):
        raise ValueError(f"Internal error: built ids len {len(ids)} != ring_len {int(ring_len)}")

    t = torch.tensor(ids, dtype=torch.long)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(permute_seed))
    perm = torch.randperm(int(ring_len), generator=g)
    t = t[perm]

    # Histogram for reporting.
    hist: Dict[str, int] = {}
    for eid in expert_ids:
        hist[str(int(eid))] = int((t == int(eid)).sum().item())
    meta = {
        "ring_len": int(ring_len),
        "buckets": int(buckets),
        "expert_ids": [int(x) for x in expert_ids],
        "counts": [int(x) for x in counts],
        "hist": hist,
        "permute_seed": int(permute_seed),
    }
    return t, meta


def _atomic_write(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


def main() -> int:
    args = _parse_args()
    in_path = Path(str(args.in_ckpt)).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {in_path}")

    out_path = Path(str(args.out_ckpt)).resolve() if str(args.out_ckpt).strip() else in_path

    ckpt = _load_ckpt(in_path)
    model_sd = ckpt.get("model")
    if not isinstance(model_sd, dict):
        raise TypeError("Checkpoint missing 'model' state dict")

    router_key = _find_router_key(model_sd)
    old_map = model_sd.get(router_key)
    if not torch.is_tensor(old_map):
        raise TypeError("router_map in checkpoint is not a tensor")
    ring_len = int(old_map.numel())

    num_experts = _infer_num_experts(ckpt, model_sd)
    buckets = int(args.buckets)
    if buckets > num_experts:
        raise ValueError(f"Checkpoint has num_experts={num_experts}; cannot build buckets={buckets}")

    expert_ids = _parse_expert_ids(args.use_expert_ids, buckets=buckets)
    if any(int(e) < 0 or int(e) >= num_experts for e in expert_ids):
        raise ValueError(f"Expert IDs must be within [0, {num_experts - 1}]")

    ratio_weights: List[float] | None = None
    counts_override: List[int] | None = None
    if str(args.ratio).strip():
        ratio_weights = _parse_ratio(str(args.ratio))
        if len(ratio_weights) != int(buckets):
            raise ValueError(
                f"--ratio length ({len(ratio_weights)}) must match --buckets ({int(buckets)})"
            )
        counts_override = _counts_ratio(int(ring_len), ratio_weights)

    new_map, meta = _make_router_map(
        ring_len,
        expert_ids,
        int(args.permute_seed),
        counts_override=counts_override,
    )
    if ratio_weights is not None:
        meta["ratio"] = [float(x) for x in ratio_weights]

    summary = {
        "in_ckpt": str(in_path),
        "out_ckpt": str(out_path),
        "router_key": router_key,
        "num_experts": int(num_experts),
        "old_ring_len": int(ring_len),
        "new_meta": meta,
    }

    if int(args.dry_run):
        if int(args.print_json):
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    model_sd[router_key] = new_map
    ckpt["model"] = model_sd
    _atomic_write(out_path, ckpt)

    if int(args.print_json):
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
