import argparse
import json
from collections import Counter
from typing import Any, Iterable

import torch


def _iter_tensors(obj: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_tensors(v, p)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _iter_tensors(v, p)
        return


def _summarize_router_map(t: torch.Tensor, expert_heads: int | None) -> dict[str, Any]:
    flat = t.detach().to("cpu").flatten().to(torch.int64)
    uniq, counts = torch.unique(flat, return_counts=True)
    total = int(flat.numel())
    hist = {int(u.item()): int(c.item()) for u, c in zip(uniq, counts, strict=False)}
    out_of_range = []
    if expert_heads is not None:
        out_of_range = [u for u in hist.keys() if u < 0 or u >= expert_heads]
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": total,
        "unique": len(hist),
        "hist": hist,
        "out_of_range": out_of_range,
    }


def _infer_expert_heads(ckpt: dict[str, Any]) -> int | None:
    # Best-effort: some checkpoints store this explicitly.
    for key_path in (
        ("model_shape", "expert_heads"),
        ("settings", "expert_heads"),
        ("config", "expert_heads"),
        ("expert_heads",),
    ):
        cur: Any = ckpt
        ok = True
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok:
            try:
                return int(cur)
            except Exception:
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    ap.add_argument(
        "--dump-json",
        action="store_true",
        help="Emit machine-readable JSON (still prints a short human header).",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit(f"Unexpected checkpoint type: {type(ckpt)}")

    expert_heads = _infer_expert_heads(ckpt)

    router_candidates: list[tuple[str, torch.Tensor]] = []
    for name, t in _iter_tensors(ckpt):
        lname = name.lower()
        if "router_map" in lname:
            router_candidates.append((name, t))

    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "top_level_keys": sorted([str(k) for k in ckpt.keys()]),
        "expert_heads_inferred": expert_heads,
        "router_maps": {},
    }

    for name, t in router_candidates:
        summary["router_maps"][name] = _summarize_router_map(t, expert_heads)

    # Friendly header (visual thinker version).
    print(f"[router_inspect] ckpt={args.checkpoint}")
    print(f"[router_inspect] expert_heads_inferred={expert_heads}")
    if not router_candidates:
        print("[router_inspect] router_map: NOT FOUND (search key contains 'router_map').")
    else:
        for name, t in router_candidates:
            s = summary["router_maps"][name]
            # Build a compact bar chart for expert IDs.
            hist = s["hist"]
            max_c = max(hist.values()) if hist else 1
            print(f"[router_inspect] router_map={name} shape={s['shape']} unique={s['unique']}")
            for expert_id in sorted(hist.keys()):
                c = hist[expert_id]
                frac = c / s["numel"] if s["numel"] else 0.0
                bar = "#" * max(1, int(40 * (c / max_c)))
                print(f"  id={expert_id:>3}  {frac:>6.2%}  {bar}")
            if s["out_of_range"]:
                print(f"[router_inspect] WARNING out_of_range={s['out_of_range']}")

    if args.dump_json:
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

