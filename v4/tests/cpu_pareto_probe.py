"""Sequential CPU Pareto probe runner for canonical nightly surfaces.

Runs a small set of candidate configs one-by-one on top of the canonical
nightly runner, first calibrating steps from a short probe and then executing
each candidate for a fixed wall-clock budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path


def _set_cpu_thread_limit(thread_limit: int) -> None:
    value = str(thread_limit)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = value


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nightly_research_runner as nrr  # noqa: E402


PRESETS: dict[str, list[dict]] = {
    "seq_tradeoff": [
        {"id": "A_seq8_b8", "seq": 8, "batch": 8},
        {"id": "B_seq16_b4", "seq": 16, "batch": 4},
    ],
    "mtaps_mixer_probe": [
        {"id": "A_current", "variant": "LLT7"},
        {"id": "B_scalar_gate", "variant": "LLT7SG"},
        {"id": "C_residual_gated", "variant": "LLT7RG"},
    ],
}


OVERRIDE_KEYS = {
    "batch",
    "seq",
    "hidden_dim",
    "M",
    "slot_dim",
    "N",
    "R",
    "pointer_mode",
    "pointer_interp_mode",
    "pointer_seam_mode",
    "fixed_C",
    "tail_mode",
    "tail_k",
    "eval_steps",
    "reset_each_batch",
    "device",
    "seed",
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _artifact_dir() -> Path:
    out = ROOT.parent / "dev_notes" / "telemetry"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_candidates(args: argparse.Namespace) -> list[dict]:
    if args.candidates_json:
        data = json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Candidate JSON must be a list of candidate dicts")
        return data
    if args.preset not in PRESETS:
        raise ValueError(f"Unknown preset: {args.preset}")
    return deepcopy(PRESETS[args.preset])


def _apply_candidate_cfg(base_cfg: dict, candidate: dict) -> dict:
    cfg = dict(base_cfg)
    for key in OVERRIDE_KEYS:
        if key in candidate:
            cfg[key] = candidate[key]
    return cfg


def _run_with_cfg(
    *,
    surface: str,
    variant: str,
    cfg: dict,
    json_out: Path,
) -> dict:
    original = deepcopy(nrr.SURFACES[surface])
    nrr.SURFACES[surface] = cfg
    try:
        payload = nrr.run_surface(
            surface=surface,
            variant=variant,
            steps_override=None,
            device_override=cfg.get("device"),
            seed_override=cfg.get("seed"),
            pointer_mode_override=cfg.get("pointer_mode"),
            pointer_interp_mode_override=cfg.get("pointer_interp_mode"),
            pointer_seam_mode_override=cfg.get("pointer_seam_mode"),
        )
    finally:
        nrr.SURFACES[surface] = original

    with open(json_out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _round_steps(steps: int) -> int:
    if steps <= 100:
        return max(steps, 50)
    return int(round(steps / 100.0) * 100)


def _candidate_summary(candidate: dict, probe_payload: dict, full_payload: dict, target_steps: int) -> dict:
    probe_result = probe_payload["result"]
    full_result = full_payload["result"]
    return {
        "id": candidate["id"],
        "candidate": candidate,
        "probe_steps": probe_payload["meta"]["steps"],
        "probe_time_s": probe_result.get("time_s"),
        "probe_s_per_step": (probe_result.get("time_s", 0.0) / max(1, probe_payload["meta"]["steps"])),
        "target_steps": target_steps,
        "final_acc": full_result.get("final_acc"),
        "best_acc": full_result.get("best_acc"),
        "final_bpc": full_result.get("final_bpc"),
        "final_loss": full_result.get("final_loss"),
        "time_s": full_result.get("time_s"),
        "s_per_step": full_result.get("time_s", 0.0) / max(1, target_steps),
        "carry_eval_acc": full_result.get("carry_eval_acc"),
        "fresh_eval_acc": full_result.get("fresh_eval_acc"),
        "carry_minus_reset_pp": full_result.get("carry_minus_reset_pp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential CPU Pareto probe runner")
    parser.add_argument("--surface", default="wikitext_sequential_carry", choices=sorted(nrr.SURFACES.keys()))
    parser.add_argument("--variant", default="LLT7", choices=sorted(nrr.VARIANTS.keys()))
    parser.add_argument("--preset", default="seq_tradeoff", choices=sorted(PRESETS.keys()))
    parser.add_argument("--candidates-json", default="")
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--time-budget-s", type=int, default=300)
    parser.add_argument("--thread-limit", type=int, default=4)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    _set_cpu_thread_limit(args.thread_limit)

    base_cfg = deepcopy(nrr.SURFACES[args.surface])
    base_cfg["device"] = "cpu"
    candidates = _load_candidates(args)
    run_stamp = _timestamp()
    out_dir = _artifact_dir()

    summaries: list[dict] = []
    print(f"CPU Pareto probe | surface={args.surface} variant={args.variant} preset={args.preset}")
    print(f"Thread limit={args.thread_limit} probe_steps={args.probe_steps} time_budget_s={args.time_budget_s}")
    for idx, candidate in enumerate(candidates, start=1):
        label = candidate["id"]
        variant = candidate.get("variant", args.variant)
        cfg_probe = _apply_candidate_cfg(base_cfg, candidate)
        cfg_probe["steps"] = args.probe_steps
        probe_json = out_dir / f"cpu_pareto_probe_{label}_probe_{run_stamp}.json"
        print(f"[{idx}/{len(candidates)}] Probe {label} ({variant}) ...")
        probe_payload = _run_with_cfg(surface=args.surface, variant=variant, cfg=cfg_probe, json_out=probe_json)
        probe_time = float(probe_payload["result"].get("time_s", 0.0))
        s_per_step = probe_time / max(1, args.probe_steps)
        target_steps = _round_steps(max(args.probe_steps, int(args.time_budget_s / max(1e-6, s_per_step))))

        cfg_full = _apply_candidate_cfg(base_cfg, candidate)
        cfg_full["steps"] = target_steps
        full_json = out_dir / f"cpu_pareto_probe_{label}_full_{run_stamp}.json"
        print(f"[{idx}/{len(candidates)}] Full {label} ({variant}): target_steps={target_steps} (~{args.time_budget_s}s) ...")
        full_payload = _run_with_cfg(surface=args.surface, variant=variant, cfg=cfg_full, json_out=full_json)
        summary = _candidate_summary(candidate, probe_payload, full_payload, target_steps)
        summary["variant"] = variant
        summary["probe_json"] = str(probe_json)
        summary["full_json"] = str(full_json)
        summaries.append(summary)
        print(
            f"{label}: acc={summary['final_acc']:.4f} "
            f"bpc={summary['final_bpc']:.4f} "
            f"time={summary['time_s']:.1f}s "
            f"s/step={summary['s_per_step']:.4f}"
        )

    payload = {
        "script": Path(__file__).name,
        "timestamp": datetime.now().isoformat(),
        "surface": args.surface,
        "variant": args.variant,
        "preset": args.preset,
        "thread_limit": args.thread_limit,
        "probe_steps": args.probe_steps,
        "time_budget_s": args.time_budget_s,
        "results": summaries,
    }
    json_out = Path(args.json_out) if args.json_out else out_dir / f"cpu_pareto_probe_{args.preset}_{run_stamp}.json"
    with open(json_out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"SUMMARY_JSON: {json_out}")


if __name__ == "__main__":
    main()
