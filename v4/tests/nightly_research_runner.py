"""Canonical nightly research runner with explicit surfaces and guardrails.

This is the only intended entrypoint for the nightly research surfaces:
    - small_wikitext_fresh
    - fast_memory_carry
    - wikitext_sequential_carry

It wraps the existing benchmark helpers behind fixed presets so nightly claims
cannot silently mix fresh-start and carry semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for subdir in ("model", "training"):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from train import ByteDataset, func_discover_dat, func_maskloss_ce  # type: ignore[import-not-found]
import instnct  # type: ignore[import-not-found]
from bench_fast_memory import _summarize_ring_trace, run_one as run_fast_memory  # type: ignore[import-not-found]
from sweep_c19_core_geometry_wikitext import (  # type: ignore[import-not-found]
    TOPK_READ_DIAG_KEYS,
    ActivationTelemetry,
    _set_determinism,
    build_model,
    make_c19_dualphi_fixed_c,
    run_one as run_wikitext_fresh,
)

PHI = (1 + math.sqrt(5)) / 2

SURFACES: dict[str, dict] = {
    "small_wikitext_fresh": {
        "state_mode": "fresh",
        "device": "cpu",
        "steps": 10000,
        "batch": 8,
        "seq": 8,
        "seed": 42,
        "hidden_dim": 32,
        "M": 64,
        "slot_dim": 8,
        "N": 1,
        "R": 1,
        "pointer_mode": "sequential",
        "pointer_interp_mode": "off",
        "fixed_C": math.pi,
        "tail_mode": "linear",
        "tail_k": 6.0,
        "reset_each_batch": True,
        "pooled_topk_read": True,
    },
    "fast_memory_carry": {
        "state_mode": "carry",
        "device": "cpu",
        "steps": 10000,
        "batch": 8,
        "seq": 8,
        "seed": 42,
        "hidden_dim": 32,
        "M": 64,
        "slot_dim": 8,
        "N": 1,
        "R": 1,
        "pointer_mode": "sequential",
        "pointer_interp_mode": "off",
        "period": 64,
        "reset_each_batch": False,
        "pooled_topk_read": True,
    },
    "wikitext_sequential_carry": {
        "state_mode": "carry",
        "device": "cpu",
        "steps": 10000,
        "batch": 8,
        "seq": 8,
        "seed": 42,
        "hidden_dim": 32,
        "M": 64,
        "slot_dim": 8,
        "N": 1,
        "R": 1,
        "pointer_mode": "sequential",
        "pointer_interp_mode": "off",
        "fixed_C": math.pi,
        "tail_mode": "linear",
        "tail_k": 6.0,
        "reset_each_batch": False,
        "pooled_topk_read": True,
        "eval_steps": 64,
    },
}

VARIANTS: dict[str, dict] = {
    "LL": {
        "read_kernel_mode": "vshape",
        "write_address_mode": "pointer",
        "read_topk_K": 2,
        "write_topk_K": 2,
    },
    "GL": {
        "read_kernel_mode": "topk",
        "write_address_mode": "pointer",
        "read_topk_K": 2,
        "write_topk_K": 2,
    },
    "GG": {
        "read_kernel_mode": "topk",
        "write_address_mode": "content_topk",
        "read_topk_K": 2,
        "write_topk_K": 2,
    },
}


def _default_json_path(surface: str, variant: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = ROOT / "dev_notes" / "telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"nightly_runner_{surface}_{variant}_{stamp}.json"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _build_meta(surface: str, variant: str, cfg: dict, overrides: dict | None = None) -> dict:
    variant_cfg = VARIANTS[variant]
    meta = {
        "surface_kind": surface,
        "variant": variant,
        "state_mode": cfg["state_mode"],
        "pointer_mode": cfg["pointer_mode"],
        "pointer_interp_mode": cfg.get("pointer_interp_mode", "off"),
        "read_mode": variant_cfg["read_kernel_mode"],
        "write_mode": variant_cfg["write_address_mode"],
        "seq": cfg["seq"],
        "steps": cfg["steps"],
        "ring_slots": cfg["M"],
        "reset_each_batch": cfg["reset_each_batch"],
        "pooled_topk_read": bool(variant_cfg["read_kernel_mode"] == "topk" and cfg.get("pooled_topk_read", False)),
        "device": cfg["device"],
        "seed": cfg["seed"],
    }
    if overrides:
        meta.update(overrides)
    return meta


def _ring_trace_guard(result: dict, batch: int, seq: int, steps: int) -> dict:
    trace = result.get("ring_trace")
    summary = result.get("ring_trace_summary")
    if not trace or not summary:
        return {"trace_present": False}
    expected_ptr_steps = steps * seq
    expected_center = steps * seq * batch
    read_width = len(trace["read_idx_trace"][0]) if trace["read_idx_trace"] else 0
    write_width = len(trace["write_idx_trace"][0]) if trace["write_idx_trace"] else 0
    center_sum = int(sum(trace["center_hist"]))
    read_sum = int(sum(trace["read_hist"]))
    write_sum = int(sum(trace["write_hist"]))
    return {
        "trace_present": True,
        "ptr_steps_ok": len(trace["ptr_trace"]) == expected_ptr_steps,
        "center_hist_ok": center_sum == expected_center,
        "read_hist_ok": read_sum == expected_center * read_width,
        "write_hist_ok": write_sum == expected_center * write_width,
        "expected_ptr_steps": expected_ptr_steps,
        "actual_ptr_steps": len(trace["ptr_trace"]),
        "expected_center_sum": expected_center,
        "actual_center_sum": center_sum,
        "expected_read_sum": expected_center * read_width,
        "actual_read_sum": read_sum,
        "expected_write_sum": expected_center * write_width,
        "actual_write_sum": write_sum,
    }


def _effective_global_flags(result: dict, variant: str) -> dict:
    if variant == "LL":
        return {"effective_global_read": False, "effective_global_write": False}
    read_outside = result.get("topk_outside_local_frac")
    write_outside = result.get("write_topk_outside_local_frac")
    ring_summary = result.get("ring_trace_summary") or {}
    read_dist = ring_summary.get("read_center_dist_mean")
    write_dist = ring_summary.get("write_center_dist_mean")
    effective_read = bool(read_outside is not None and read_outside >= 0.50 and (read_dist or 0.0) > 1.5)
    effective_write = bool(
        variant == "GG"
        and write_outside is not None
        and write_outside >= 0.50
        and (write_dist or 0.0) > 1.5
    )
    return {
        "effective_global_read": effective_read,
        "effective_global_write": effective_write,
    }


def _surface_guards(surface: str, result: dict, meta: dict) -> dict:
    guards = {
        "surface_kind": surface,
        "reset_each_batch_matches": True,
        "carry_surface_has_broader_pointer": True,
    }
    ring_summary = result.get("ring_trace_summary") or {}
    ptr_unique = ring_summary.get("ptr_unique_frac")
    if surface == "small_wikitext_fresh" and meta.get("pointer_mode") == "sequential":
        guards["fresh_pointer_coverage_capped"] = bool(
            ptr_unique is not None and ptr_unique <= (meta["seq"] / meta["ring_slots"]) + 1e-9
        )
    if surface == "wikitext_sequential_carry":
        guards["carry_pointer_coverage_exceeds_fresh_bound"] = bool(
            ptr_unique is not None and ptr_unique > (meta["seq"] / meta["ring_slots"])
        )
    return guards


def _require_topk_diag(variant: str, result: dict):
    if variant == "LL":
        return
    missing = [key for key in ("topk_mean_abs_circ_dist", "topk_outside_local_frac") if result.get(key) is None]
    if missing:
        raise RuntimeError(f"Missing required topk telemetry for {variant}: {missing}")
    if variant == "GG":
        missing_write = [
            key for key in ("write_topk_mean_abs_circ_dist", "write_topk_outside_local_frac")
            if result.get(key) is None
        ]
        if missing_write:
            raise RuntimeError(f"Missing required write-topk telemetry for GG: {missing_write}")


def _discover_dataset(seq: int, seed: int) -> ByteDataset:
    data_dir = ROOT / "training_data"
    if not data_dir.exists():
        fallback_dir = Path(r"S:\AI\work\VRAXION_DEV\v4\training_data")
        if fallback_dir.exists():
            data_dir = fallback_dir
    files = func_discover_dat(str(data_dir))
    return ByteDataset(files, seq, embed_mode=True, seed=seed)


def _run_small_wikitext_fresh(surface: str, variant: str, cfg: dict) -> dict:
    variant_cfg = VARIANTS[variant]
    dataset = _discover_dataset(cfg["seq"], cfg["seed"])
    telemetry = ActivationTelemetry(sample_per_call=1024)
    act_fn = make_c19_dualphi_fixed_c(
        c_value=cfg["fixed_C"],
        tail_mode=cfg["tail_mode"],
        tail_k=cfg["tail_k"],
        telemetry=telemetry,
    )
    act_fn._telemetry = telemetry
    result = run_wikitext_fresh(
        f"{surface}-{variant}",
        act_fn,
        dataset,
        cfg["steps"],
        cfg["batch"],
        cfg["seed"],
        kernel_mode=variant_cfg["read_kernel_mode"],
        topk_k=variant_cfg["read_topk_K"],
        replace_impl="dense",
        topk_read_diag=(variant != "LL"),
        read_kernel_mode=variant_cfg["read_kernel_mode"],
        write_address_mode=variant_cfg["write_address_mode"],
        write_topk_k=variant_cfg["write_topk_K"],
        pointer_mode=cfg["pointer_mode"],
        pointer_interp_mode=cfg["pointer_interp_mode"],
        ring_trace=True,
        device=cfg["device"],
        hidden_dim=cfg["hidden_dim"],
        M=cfg["M"],
        slot_dim=cfg["slot_dim"],
        N=cfg["N"],
        R=cfg["R"],
    )
    return result


def _run_fast_memory_carry(surface: str, variant: str, cfg: dict) -> dict:
    variant_cfg = VARIANTS[variant]
    result = run_fast_memory(
        N=cfg["N"],
        period=cfg["period"],
        steps=cfg["steps"],
        batch=cfg["batch"],
        seq=cfg["seq"],
        hidden_dim=cfg["hidden_dim"],
        M=cfg["M"],
        slot_dim=cfg["slot_dim"],
        model_type="instnct",
        device=cfg["device"],
        io_split_mode="off",
        gated_write=False,
        lr=1e-3,
        log_every=100,
        seed=cfg["seed"],
        read_kernel_mode=variant_cfg["read_kernel_mode"],
        write_address_mode=variant_cfg["write_address_mode"],
        topk_k=variant_cfg["read_topk_K"],
        ring_trace=True,
        pointer_mode=cfg["pointer_mode"],
        pointer_interp_mode=cfg["pointer_interp_mode"],
    )
    result["best_acc"] = result.get("peak_acc")
    result["time_s"] = result.get("wall_time")
    result["final_loss"] = None
    result["final_bpc"] = None
    return result


def _eval_sequential(model, dataset: ByteDataset, batch: int, seq: int, device: str, steps: int, reset_each_batch: bool) -> float:
    offsets = np.array(dataset._seq_offsets, dtype=np.int64)
    state = None
    total_correct = 0.0
    total_sup = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            xb, yb, mask = dataset.sample_batch_sequential(batch, device)
            logits, new_state = model(xb, state=None if reset_each_batch else state)
            if new_state is not None and not reset_each_batch:
                state = {k: v.detach() for k, v in new_state.items()}
            preds = logits.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            total_correct += correct.sum().item()
            total_sup += mask.sum().item()
    dataset._seq_offsets = offsets
    model.train()
    return total_correct / max(total_sup, 1.0)


def _run_wikitext_sequential_carry(surface: str, variant: str, cfg: dict) -> dict:
    variant_cfg = VARIANTS[variant]
    _set_determinism(cfg["seed"])
    dataset = _discover_dataset(cfg["seq"], cfg["seed"])
    dataset.init_sequential(cfg["batch"])
    telemetry = ActivationTelemetry(sample_per_call=1024)
    act_fn = make_c19_dualphi_fixed_c(
        c_value=cfg["fixed_C"],
        tail_mode=cfg["tail_mode"],
        tail_k=cfg["tail_k"],
        telemetry=telemetry,
    )
    act_fn._telemetry = telemetry

    orig_fn = instnct._c19_activation
    instnct._c19_activation = act_fn
    instnct.set_topk_read_diag_enabled(variant != "LL")
    instnct.set_ring_trace_enabled(True)

    model = build_model(
        cfg["seed"],
        replace_impl="dense",
        kernel_mode=variant_cfg["read_kernel_mode"],
        topk_k=variant_cfg["read_topk_K"],
        read_kernel_mode=variant_cfg["read_kernel_mode"],
        write_address_mode=variant_cfg["write_address_mode"],
        write_topk_k=variant_cfg["write_topk_K"],
        pointer_mode=cfg["pointer_mode"],
        pointer_interp_mode=cfg["pointer_interp_mode"],
        device=cfg["device"],
        hidden_dim=cfg["hidden_dim"],
        M=cfg["M"],
        slot_dim=cfg["slot_dim"],
        N=cfg["N"],
        R=cfg["R"],
    )
    for name, param in model.named_parameters():
        if any(key in name for key in ("c19_C_", "c19_rho_")):
            param.requires_grad_(False)

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    losses: list[float] = []
    accs: list[float] = []
    topk_diag_rows = {key: [] for key in TOPK_READ_DIAG_KEYS}
    ring_trace_rows = {
        "ptr_trace": [],
        "read_idx_trace": [],
        "read_weight_trace": [],
        "write_idx_trace": [],
        "write_weight_trace": [],
        "read_write_overlap_trace": [],
        "center_hist": [0 for _ in range(cfg["M"])],
        "read_hist": [0 for _ in range(cfg["M"])],
        "write_hist": [0 for _ in range(cfg["M"])],
    }
    state = None
    max_grad = 0.0
    t0 = time.time()

    for step in range(1, cfg["steps"] + 1):
        xb, yb, mask = dataset.sample_batch_sequential(cfg["batch"], cfg["device"])
        logits, new_state = model(xb, state=state)
        if new_state is not None:
            state = {k: v.detach() for k, v in new_state.items()}
        _, masked_loss = func_maskloss_ce(logits, yb, mask)
        opt.zero_grad()
        masked_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
        opt.step()

        losses.append(masked_loss.item())
        max_grad = max(max_grad, gn)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            accs.append((correct.sum() / mask.sum().clamp(min=1)).item())

        for key in TOPK_READ_DIAG_KEYS:
            value = model._diag.get(key)
            if value is not None:
                topk_diag_rows[key].append(float(value))
        trace = getattr(model, "_ring_trace", None)
        if trace is not None:
            for key in ("ptr_trace", "read_idx_trace", "read_weight_trace", "write_idx_trace", "write_weight_trace", "read_write_overlap_trace"):
                ring_trace_rows[key].extend(trace.get(key, []))
            for key in ("center_hist", "read_hist", "write_hist"):
                vals = trace.get(key, [])
                ring_trace_rows[key] = [a + int(b) for a, b in zip(ring_trace_rows[key], vals)]

        if step % 100 == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            tele = act_fn._telemetry.summary()
            elapsed = time.time() - t0
            diag_suffix = ""
            if variant != "LL":
                dist = model._diag.get("topk_mean_abs_circ_dist")
                outside = model._diag.get("topk_outside_local_frac")
                wdist = model._diag.get("write_topk_mean_abs_circ_dist")
                woutside = model._diag.get("write_topk_outside_local_frac")
                if dist is not None and outside is not None:
                    diag_suffix = f"  topk_dist={dist:.2f}  outside={outside:.3f}"
                if wdist is not None and woutside is not None:
                    diag_suffix += f"  wdist={wdist:.2f}  wout={woutside:.3f}"
            print(
                f"  [{surface}-{variant}] step {step:4d}/{cfg['steps']}  "
                f"loss={avg_loss:.4f}  bpc={avg_loss*1.4427:.3f}  "
                f"acc={avg_acc:.3f}  gnorm={gn:.1f}  "
                f"tail={tele['tail_hit_pct']:.3f}%  "
                f"p99|x|/C={tele['p99_abs_over_c']:.2f}  "
                f"p99-ring={tele['p99_ring_idx']:.2f}  "
                f"{elapsed:.0f}s{diag_suffix}"
            )

    elapsed = time.time() - t0
    carry_eval = _eval_sequential(model, dataset, cfg["batch"], cfg["seq"], cfg["device"], cfg["eval_steps"], reset_each_batch=False)
    reset_eval = _eval_sequential(model, dataset, cfg["batch"], cfg["seq"], cfg["device"], cfg["eval_steps"], reset_each_batch=True)

    instnct._c19_activation = orig_fn
    instnct.set_topk_read_diag_enabled(False)
    instnct.set_ring_trace_enabled(False)

    result = {
        "variant": f"{surface}-{variant}",
        "final_loss": sum(losses[-100:]) / min(100, len(losses)),
        "final_bpc": sum(losses[-100:]) / min(100, len(losses)) * 1.4427,
        "final_acc": sum(accs[-100:]) / min(100, len(accs)),
        "best_acc": max(accs),
        "time_s": elapsed,
        "s_per_step": elapsed / cfg["steps"],
        "max_grad": max_grad,
        "loss_curve": losses,
        "acc_curve": accs,
        "carry_eval_acc": carry_eval,
        "fresh_eval_acc": reset_eval,
        "carry_minus_reset_pp": (carry_eval - reset_eval) * 100.0,
    }
    result.update(act_fn._telemetry.summary())
    for key in TOPK_READ_DIAG_KEYS:
        rows = topk_diag_rows[key]
        result[key] = (sum(rows) / len(rows)) if rows else None
    result["ring_trace_summary"] = _summarize_ring_trace(ring_trace_rows, cfg["M"])
    result["ring_trace"] = ring_trace_rows
    return result


def run_surface(
    surface: str,
    variant: str,
    steps_override: int | None = None,
    device_override: str | None = None,
    pointer_mode_override: str | None = None,
    pointer_interp_mode_override: str | None = None,
) -> dict:
    cfg = dict(SURFACES[surface])
    if steps_override is not None:
        cfg["steps"] = int(steps_override)
    if device_override is not None:
        cfg["device"] = device_override
    if pointer_mode_override is not None:
        cfg["pointer_mode"] = pointer_mode_override
    if pointer_interp_mode_override is not None:
        cfg["pointer_interp_mode"] = pointer_interp_mode_override

    if surface == "small_wikitext_fresh":
        result = _run_small_wikitext_fresh(surface, variant, cfg)
    elif surface == "fast_memory_carry":
        result = _run_fast_memory_carry(surface, variant, cfg)
    elif surface == "wikitext_sequential_carry":
        result = _run_wikitext_sequential_carry(surface, variant, cfg)
    else:
        raise ValueError(f"Unknown surface: {surface}")

    meta = _build_meta(surface, variant, cfg)
    _require_topk_diag(variant, result)
    guards = {
        **_ring_trace_guard(result, cfg["batch"], cfg["seq"], cfg["steps"]),
        **_surface_guards(surface, result, meta),
        **_effective_global_flags(result, variant),
    }

    if (
        surface == "small_wikitext_fresh"
        and meta.get("pointer_mode") == "sequential"
        and not guards.get("fresh_pointer_coverage_capped", False)
    ):
        raise RuntimeError("small_wikitext_fresh violated fresh-start pointer coverage guard")
    if surface == "wikitext_sequential_carry" and not guards.get("carry_pointer_coverage_exceeds_fresh_bound", False):
        raise RuntimeError("wikitext_sequential_carry failed carry pointer coverage guard")
    if not guards.get("ptr_steps_ok", True) or not guards.get("center_hist_ok", True) or not guards.get("read_hist_ok", True) or not guards.get("write_hist_ok", True):
        raise RuntimeError(f"Trace consistency failed: {guards}")

    return {
        "script": Path(__file__).name,
        "timestamp": datetime.now().isoformat(),
        "meta": meta,
        "surface_config": cfg,
        "guards": guards,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Canonical nightly research runner")
    parser.add_argument("--surface", required=True, choices=sorted(SURFACES.keys()))
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS.keys()))
    parser.add_argument("--steps", type=int, default=0, help="Optional override for preset steps.")
    parser.add_argument("--device", type=str, default="", choices=["", "cpu", "cuda"])
    parser.add_argument("--pointer-mode", type=str, default="", choices=["", "sequential", "learned", "pilot"])
    parser.add_argument("--pointer-interp-mode", type=str, default="", choices=["", "off", "linear"])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    payload = run_surface(
        surface=args.surface,
        variant=args.variant,
        steps_override=(args.steps or None),
        device_override=(args.device or None),
        pointer_mode_override=(args.pointer_mode or None),
        pointer_interp_mode_override=(args.pointer_interp_mode or None),
    )

    result = payload["result"]
    guards = payload["guards"]
    meta = payload["meta"]
    print("=" * 100)
    print(f"Nightly research runner | surface={meta['surface_kind']} variant={meta['variant']}")
    print(
        f"final_acc={_fmt(result.get('final_acc'))} "
        f"best_acc={_fmt(result.get('best_acc'))} "
        f"bpc={_fmt(result.get('final_bpc'))} "
        f"time={_fmt(result.get('time_s', result.get('wall_time')), 1)}s"
    )
    trace = result.get("ring_trace_summary") or {}
    if trace:
        print(
            f"trace: ptr_unique={trace.get('ptr_unique_frac', 0):.3f} "
            f"read_unique={trace.get('read_unique_frac', 0):.3f} "
            f"write_unique={trace.get('write_unique_frac', 0):.3f} "
            f"rdist={trace.get('read_center_dist_mean', 0):.2f} "
            f"wdist={trace.get('write_center_dist_mean', 0):.2f}"
        )
    print(
        f"guards: ptr_steps={guards.get('ptr_steps_ok')} center_hist={guards.get('center_hist_ok')} "
        f"read_hist={guards.get('read_hist_ok')} write_hist={guards.get('write_hist_ok')} "
        f"effective_global_read={guards.get('effective_global_read')} "
        f"effective_global_write={guards.get('effective_global_write')}"
    )
    json_out = Path(args.json_out) if args.json_out else _default_json_path(args.surface, args.variant)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON: {json_out}")
    print("=" * 100)


if __name__ == "__main__":
    main()
