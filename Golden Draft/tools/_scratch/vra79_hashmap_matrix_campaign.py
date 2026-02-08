#!/usr/bin/env python3
"""Run deterministic multi-seed matrix campaign and emit a single verdict."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
MATRIX_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_hashmap_matrix_speed.py"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_seed_list(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in str(raw).split(","):
        txt = chunk.strip()
        if not txt:
            continue
        try:
            out.append(int(txt))
        except Exception as exc:
            raise ValueError(f"invalid seed entry: {txt}") from exc
    if not out:
        raise ValueError("at least one seed is required")
    return out


def _arm_final_acc_delta(summary: dict[str, Any], arm_name: str) -> float | None:
    arms = summary.get("arms", [])
    if not isinstance(arms, list):
        return None
    for arm in arms:
        if not isinstance(arm, dict):
            continue
        if str(arm.get("arm", "")) != str(arm_name):
            continue
        final = arm.get("final", {})
        if not isinstance(final, dict):
            return None
        try:
            return float(final.get("acc_delta", 0.0))
        except Exception:
            return None
    return None


def _run_matrix(args: argparse.Namespace, seed: int, run_root: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(MATRIX_TOOL),
        "--run-root",
        str(run_root),
        "--label",
        str(args.label),
        "--arms",
        str(args.arms),
        "--seed",
        str(int(seed)),
        "--device",
        str(args.device),
        "--ring-len",
        str(int(args.ring_len)),
        "--slot-dim",
        str(int(args.slot_dim)),
        "--expert-heads",
        str(int(args.expert_heads)),
        "--ratio",
        str(args.ratio),
        "--batch-size",
        str(int(args.batch_size)),
        "--steps",
        str(int(args.steps)),
        "--checkpoint-every",
        str(int(args.checkpoint_every)),
        "--eval-n-realtime",
        str(int(args.eval_n_realtime)),
        "--eval-n-anchor",
        str(int(args.eval_n_anchor)),
        "--anchor-steps",
        str(args.anchor_steps),
    ]
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if int(cp.returncode) != 0:
        raise RuntimeError(
            "matrix run failed\n"
            f"seed={seed}\n"
            f"rc={cp.returncode}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}\n"
        )
    summary_path = run_root / "matrix_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing matrix summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic multi-seed matrix campaign.")
    ap.add_argument("--run-root", default="")
    ap.add_argument("--label", default="hashmap_matrix_campaign")
    ap.add_argument("--seeds", default="123,231,312")
    ap.add_argument("--arms", default="a,b,c,d")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ring-len", type=int, default=8192)
    ap.add_argument("--slot-dim", type=int, default=576)
    ap.add_argument("--expert-heads", type=int, default=3)
    ap.add_argument("--ratio", default="0.55,0.34,0.11")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--eval-n-realtime", type=int, default=64)
    ap.add_argument("--eval-n-anchor", type=int, default=256)
    ap.add_argument("--anchor-steps", default="200,400,600")
    ap.add_argument("--winner-margin", type=float, default=0.01)
    ap.add_argument("--tie-margin", type=float, default=0.005)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = _parse_seed_list(args.seeds)
    if int(args.steps) <= 0:
        raise ValueError("--steps must be > 0")
    if int(args.checkpoint_every) <= 0:
        raise ValueError("--checkpoint-every must be > 0")

    if str(args.run_root).strip():
        campaign_root = Path(str(args.run_root)).resolve()
    else:
        campaign_root = (
            REPO_ROOT
            / "bench_vault"
            / "_tmp"
            / "vra79_hashmap_campaign"
            / f"{_utc_stamp()}-{args.label}"
        )
    campaign_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    d_beats_a_count = 0
    a_beats_d_count = 0

    for seed in seeds:
        seed_root = campaign_root / f"seed{int(seed):03d}"
        seed_root.mkdir(parents=True, exist_ok=True)
        summary = _run_matrix(args, seed, seed_root)
        summaries.append(
            {
                "seed": int(seed),
                "run_root": str(seed_root),
                "summary_path": str(seed_root / "matrix_summary.json"),
            }
        )
        acc_a = _arm_final_acc_delta(summary, "arm_a_route_equal_capacity_equal")
        acc_d = _arm_final_acc_delta(summary, "arm_d_route_fibo_capacity_fibo")
        diff = None
        if acc_a is not None and acc_d is not None:
            diff = float(acc_d - acc_a)
            if diff >= float(args.winner_margin):
                d_beats_a_count += 1
            if diff <= -float(args.winner_margin):
                a_beats_d_count += 1
        rows.append(
            {
                "seed": int(seed),
                "acc_delta_arm_a": acc_a,
                "acc_delta_arm_d": acc_d,
                "d_minus_a": diff,
            }
        )

    total = max(1, len(seeds))
    verdict = "no_win"
    reason = "insufficient separation"
    if d_beats_a_count >= 2:
        verdict = "winner_fibo_d"
        reason = f"arm_d beat arm_a by >= {args.winner_margin:.4f} on {d_beats_a_count}/{total} seeds"
    elif a_beats_d_count >= 2:
        verdict = "fail_fibo_d"
        reason = f"arm_d trailed arm_a by <= -{args.winner_margin:.4f} on {a_beats_d_count}/{total} seeds"
    else:
        within_tie = 0
        for row in rows:
            diff = row.get("d_minus_a")
            if isinstance(diff, (int, float)) and abs(float(diff)) <= float(args.tie_margin):
                within_tie += 1
        if within_tie >= 2:
            verdict = "tie"
            reason = f"arm_d and arm_a stayed within +/-{args.tie_margin:.4f} on {within_tie}/{total} seeds"

    csv_path = campaign_root / "campaign_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "acc_delta_arm_a", "acc_delta_arm_d", "d_minus_a"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "campaign_root": str(campaign_root),
        "config": {
            "seeds": seeds,
            "arms": str(args.arms),
            "device": str(args.device),
            "ring_len": int(args.ring_len),
            "slot_dim": int(args.slot_dim),
            "expert_heads": int(args.expert_heads),
            "ratio": str(args.ratio),
            "batch_size": int(args.batch_size),
            "steps": int(args.steps),
            "checkpoint_every": int(args.checkpoint_every),
            "eval_n_realtime": int(args.eval_n_realtime),
            "eval_n_anchor": int(args.eval_n_anchor),
            "anchor_steps": str(args.anchor_steps),
            "winner_margin": float(args.winner_margin),
            "tie_margin": float(args.tie_margin),
        },
        "seed_runs": summaries,
        "rows": rows,
        "counts": {
            "d_beats_a": int(d_beats_a_count),
            "a_beats_d": int(a_beats_d_count),
            "num_seeds": int(total),
        },
        "verdict": str(verdict),
        "reason": str(reason),
        "campaign_summary_csv": str(csv_path),
    }
    out_path = campaign_root / "campaign_summary.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
