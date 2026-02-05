"""Generate arc-safe mitosis meta from a checkpoint (static-space v1.2).

Design goals:
- Deterministic output (stable tie-breaks; stable JSON structure).
- Arc-safe by default: produce a contiguous arc and redirect ONLY addresses
  currently owned by the chosen parent expert.
- Avoid runtime paging: this tool only reads checkpoints and writes meta JSON.

NOTE: The default address policy uses pointer-visit counts from a small,
deterministic probe pass. It does *not* require loss-by-address plumbing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


def _bootstrap_paths() -> None:
    """Best-effort sys.path bootstrap for running as a standalone script."""

    draftr = Path(__file__).resolve().parents[1]  # Golden Draft/
    reproo = draftr.parent

    if str(draftr) not in sys.path:
        sys.path.insert(0, str(draftr))

    candls: list[str] = []
    for keystr in ("VRAXION_GOLDEN_SRC", "GOLDEN_CODE_ROOT", "GOLDEN_CODE_PATH", "GOLDEN_CODE_DIR"):
        envval = os.environ.get(keystr)
        if envval:
            candls.append(envval)

    candls.append(str(reproo / "Golden Code"))

    for candpt in candls:
        try:
            if candpt and os.path.isdir(candpt) and candpt not in sys.path:
                sys.path.insert(0, candpt)
                break
        except OSError:
            continue


def _infer_num_experts_from_state(state: Mapping[str, Any]) -> int:
    """Infer num_experts from state_dict key layout."""

    if any(isinstance(k, str) and k.startswith("head.single.") for k in state.keys()):
        return 1

    # Prefer expert shard keys when present.
    max_idx = -1
    prefix = "head.experts."
    for keystr in state.keys():
        if not isinstance(keystr, str) or not keystr.startswith(prefix):
            continue
        rest = keystr[len(prefix) :]
        try:
            idx = int(rest.split(".", 1)[0])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def _load_checkpoint_payload(path: Path) -> MutableMapping[str, Any]:
    """Load either a monolithic checkpoint (.pt) or a modular dir router.state."""

    from tools._checkpoint_io import safe_torch_load

    if path.is_dir():
        router_path = path / "system" / "router.state"
        if not router_path.exists():
            raise FileNotFoundError(f"modular router.state not found: {router_path}")
        payload = safe_torch_load(str(router_path))
    else:
        payload = safe_torch_load(str(path))

    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(payload).__name__}")
    return payload


def _extract_model_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint model state must be a dict, got {type(state).__name__}")
    return state


def _coerce_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return int(default)


def _auto_arc_len(router_map_len: int, num_experts: int) -> int:
    denom = 2 * max(1, int(num_experts))
    return max(16, int(router_map_len) // denom)


def best_circular_window_start(values: Sequence[int], window: int) -> tuple[int, int]:
    """Return (best_start, best_sum) for a circular window sum.

    Tie-break: smallest start index.
    """

    L = len(values)
    if L <= 0:
        raise ValueError("values must be non-empty")
    if window <= 0 or window > L:
        raise ValueError(f"window must be in [1, {L}], got {window}")

    doubled = list(values) + list(values)
    prefix = [0]
    for v in doubled:
        prefix.append(prefix[-1] + int(v))

    best_sum = None
    best_start = 0
    for start in range(L):
        cur = prefix[start + window] - prefix[start]
        if best_sum is None or cur > best_sum:
            best_sum = cur
            best_start = start
    return best_start, int(best_sum or 0)


def parent_only_addresses_in_arc(router_map: Sequence[int], *, parent: int, start: int, length: int) -> list[int]:
    """Return arc addresses that are currently owned by parent expert."""

    L = len(router_map)
    if L <= 0:
        raise ValueError("router_map must be non-empty")
    if length <= 0 or length > L:
        raise ValueError(f"length must be in [1, {L}], got {length}")
    if start < 0 or start >= L:
        raise ValueError(f"start must be in [0, {L-1}], got {start}")

    out: list[int] = []
    parent = int(parent)
    for off in range(int(length)):
        addr = (int(start) + off) % L
        if int(router_map[addr]) == parent:
            out.append(addr)
    return out


def expert_counts_from_router_map(
    router_map: Sequence[int],
    counts_by_address: Sequence[int],
    *,
    num_experts: int,
) -> list[int]:
    """Aggregate address visit counts into per-expert visit counts."""

    num_experts = int(num_experts)
    counts = [0 for _ in range(num_experts)]
    for addr, cnt in enumerate(counts_by_address):
        try:
            exp = int(router_map[addr])
        except Exception:
            continue
        if 0 <= exp < num_experts:
            counts[exp] += int(cnt)
    return counts


def _infer_input_dim(state: Mapping[str, Any]) -> int:
    w = state.get("input_proj.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", ())) == 2:
        return int(w.shape[1])
    return 1


def _infer_slot_dim(state: Mapping[str, Any]) -> int:
    w = state.get("input_proj.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", ())) == 2:
        return int(w.shape[0])
    return 16


def _infer_num_classes(state: Mapping[str, Any]) -> int:
    for key in ("head.single.weight", "head.experts.0.weight"):
        w = state.get(key)
        if hasattr(w, "shape") and len(getattr(w, "shape", ())) == 2:
            return int(w.shape[0])
    return 2


def _collect_ptr_hist_counts(
    *,
    state: Mapping[str, Any],
    router_map_len: int,
    num_experts: int,
    ptr_samples: int = 4096,
    batch_size: int = 64,
    seq_len: int = 16,
) -> list[int]:
    """Collect a deterministic pointer histogram via a small CPU probe pass."""

    import torch
    from vraxion.instnct import absolute_hallway

    from vraxion.instnct.absolute_hallway import AbsoluteHallway

    absolute_hallway.EXPERT_HEADS = int(num_experts)

    input_dim = _infer_input_dim(state)
    slot_dim = _infer_slot_dim(state)
    num_classes = _infer_num_classes(state)
    ring_len = int(router_map_len)

    # Create a model compatible with the checkpoint tensors we actually care about.
    model = AbsoluteHallway(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        ring_len=int(ring_len),
        slot_dim=int(slot_dim),
    ).cpu()
    model.eval()

    # Avoid shape-mismatch errors by not forcing head weights to match a chosen num_classes.
    filtered: dict[str, Any] = {k: v for k, v in state.items() if isinstance(k, str) and not k.startswith("head.")}
    try:
        model.load_state_dict(filtered, strict=False)
    except Exception:
        # Best-effort: this is a heuristic probe; skip weights when load fails.
        pass

    torch.manual_seed(1337)

    total_batches = int(math.ceil(int(ptr_samples) / max(1, int(batch_size))))
    counts = [0 for _ in range(int(router_map_len))]
    L = int(router_map_len)

    with torch.no_grad():
        for _ in range(total_batches):
            x = torch.randn(int(batch_size), int(seq_len), int(input_dim), dtype=torch.float32)
            _logits, _move_penalty = model(x)
            ptr = getattr(model, "last_ptr_int", None)
            if ptr is None or not hasattr(ptr, "numel"):
                raise RuntimeError("model did not expose last_ptr_int; cannot build ptr histogram")
            vals = ptr.view(-1).to(torch.int64).clamp(min=0, max=L - 1)
            binc = torch.bincount(vals, minlength=L).to(torch.int64).tolist()
            for i, v in enumerate(binc):
                counts[i] += int(v)

    return counts


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate arc-safe mitosis_meta.json from a checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Input checkpoint (.pt) or modular checkpoint directory")
    p.add_argument("--output", required=True, help="Output mitosis_meta.json path")
    p.add_argument("--address-policy", choices=("ptr_hist_arc", "topk_loss"), default="ptr_hist_arc")
    p.add_argument("--arc-len", default="auto", help="Arc length (int) or 'auto'")
    p.add_argument("--top-k", type=int, default=256, help="Top-k addresses for topk_loss mode")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap_paths()
    args = _parse_args(argv)

    ckppth = Path(os.path.expanduser(args.checkpoint)).resolve()
    outpth = Path(os.path.expanduser(args.output)).resolve()

    try:
        if args.address_policy != "ptr_hist_arc":
            raise NotImplementedError("topk_loss is deferred (requires loss-by-address telemetry); use ptr_hist_arc")

        payload = _load_checkpoint_payload(ckppth)
        state = _extract_model_state(payload)

        router_map = state.get("router_map")
        if router_map is None or not hasattr(router_map, "numel"):
            raise KeyError("checkpoint missing router_map")

        router_cpu = router_map.detach().cpu().view(-1)
        L = int(router_cpu.numel())
        if L <= 0:
            raise ValueError("router_map is empty")

        num_experts = payload.get("num_experts")
        num_experts = _coerce_int(num_experts, 0)
        if num_experts <= 0:
            num_experts = _infer_num_experts_from_state(state)
        if num_experts <= 0:
            raise ValueError("could not infer num_experts from checkpoint")

        # Determine pointer visit counts per address.
        counts_by_address = _collect_ptr_hist_counts(state=state, router_map_len=L, num_experts=num_experts)

        # Determine parent expert.
        if int(num_experts) == 1:
            parent_expert = 0
        else:
            expert_counts = expert_counts_from_router_map(router_cpu.tolist(), counts_by_address, num_experts=num_experts)
            parent_expert = int(max(range(len(expert_counts)), key=lambda i: expert_counts[i]))

        # Score arc using ONLY parent-owned addresses.
        masked_counts = [int(cnt) if int(router_cpu[idx]) == int(parent_expert) else 0 for idx, cnt in enumerate(counts_by_address)]

        if str(args.arc_len).strip().lower() == "auto":
            arc_len = _auto_arc_len(L, int(num_experts))
        else:
            arc_len = int(args.arc_len)
        arc_len = max(1, min(int(arc_len), L))

        arc_start, arc_score = best_circular_window_start(masked_counts, arc_len)

        hot_addresses = parent_only_addresses_in_arc(
            router_cpu.tolist(),
            parent=parent_expert,
            start=arc_start,
            length=arc_len,
        )

        dropped = int(arc_len) - int(len(parent_only_addresses_in_arc(router_cpu.tolist(), parent=parent_expert, start=arc_start, length=arc_len)))

        # Hard guard: never remap addresses owned by other experts.
        if any(int(router_cpu[a]) != int(parent_expert) for a in hot_addresses):
            raise RuntimeError("hot_addresses contained non-parent-owned addresses (bug)")

        expert_counts = expert_counts_from_router_map(router_cpu.tolist(), counts_by_address, num_experts=num_experts)
        total_visits = sum(expert_counts) if expert_counts else 0
        parent_share = (expert_counts[parent_expert] / total_visits) if total_visits else 0.0

        meta = {
            "schema_version": "mitosis_meta_v1",
            "parent_expert": int(parent_expert),
            "router_map_len": int(L),
            "num_experts": int(num_experts),
            "address_policy": str(args.address_policy),
            "hot_arc": {"start": int(arc_start), "len": int(arc_len)},
            "hot_addresses": [int(a) for a in hot_addresses],
            "metrics": {
                "parent_share": float(parent_share),
                "ptr_samples": int(sum(counts_by_address)),
                "arc_score": int(arc_score),
                "dropped_non_parent": int(dropped),
            },
        }

        from tools._checkpoint_io import atomic_json_dump

        atomic_json_dump(meta, outpth, indent=2)
    except Exception as exc:
        print(f"[mitosis_meta] error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
