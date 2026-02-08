#!/usr/bin/env python3
"""Speed-first deterministic 2x2 matrix:
- routing: equal vs fibo ratio
- capacity: equal vs fibo split
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
ENTRY_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_saturation_entry.py"
ROUTER_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_router_map_fibo.py"
EVAL_TOOL = REPO_ROOT / r"Golden Draft\tools\eval_ckpt_assoc_byte.py"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ratio_len(raw: str) -> int:
    return len([p for p in str(raw).split(",") if p.strip()])


def _equal_ratio(expert_heads: int) -> str:
    return ",".join("1" for _ in range(int(expert_heads)))


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> None:
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if int(cp.returncode) != 0:
        raise RuntimeError(f"command failed rc={cp.returncode}: {' '.join(cmd)}")


def _run_json(cmd: List[str], *, env: Dict[str, str] | None = None) -> Dict[str, Any]:
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if int(cp.returncode) != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {cp.returncode}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}"
        )
    out = (cp.stdout or "").strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {"stdout": out}


def _parse_int_set(raw: str) -> set[int]:
    out: set[int] = set()
    for chunk in str(raw).split(","):
        txt = chunk.strip()
        if not txt:
            continue
        try:
            value = int(txt)
        except Exception as exc:
            raise ValueError(f"invalid integer list entry: {txt}") from exc
        if value > 0:
            out.add(value)
    return out


def _checkpoint_step(path: Path) -> int | None:
    m = re.search(r"_step_?(\d+)\.pt$", path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _list_step_checkpoints(train_root: Path, checkpoint_every: int, max_steps: int) -> List[Path]:
    out: List[Path] = []
    checkpoints_dir = train_root / "checkpoints"
    for path in sorted(checkpoints_dir.glob("checkpoint_step*.pt")):
        step = _checkpoint_step(path)
        if step is None:
            continue
        if step <= 0 or step > int(max_steps):
            continue
        if step % int(checkpoint_every) != 0:
            continue
        out.append(path)
    return out


def _eval_checkpoint(train_root: Path, ckpt_path: Path, eval_n: int) -> Dict[str, Any]:
    t0 = time.time()
    cmd = [
        sys.executable,
        str(EVAL_TOOL),
        "--run-root",
        str(train_root),
        "--checkpoint",
        str(ckpt_path),
        "--eval-samples",
        str(int(eval_n)),
        "--batch-size",
        "32",
        "--device",
        "cpu",
        "--heartbeat-s",
        "20",
        "--force-eval-disjoint",
    ]
    _run(cmd)
    elapsed = float(time.time() - t0)
    report_path = train_root / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing report.json after eval: {train_root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    eval_blk = report.get("eval", {}) if isinstance(report.get("eval"), dict) else {}
    settings = report.get("settings", {}) if isinstance(report.get("settings"), dict) else {}
    val_range = int(settings.get("val_range", 256))
    eval_acc = float(eval_blk.get("eval_acc", 0.0))
    acc_delta = float(eval_acc - (1.0 / float(val_range)))
    return {
        "report_path": str(report_path),
        "eval_seconds": float(elapsed),
        "eval_n": int(eval_blk.get("eval_n", int(eval_n))),
        "eval_acc": float(eval_acc),
        "eval_loss": float(eval_blk.get("eval_loss", 0.0)),
        "acc_delta": float(acc_delta),
    }


def _slope(points: List[Dict[str, Any]], key: str) -> float:
    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        if key not in p:
            continue
        try:
            xs.append(float(p["step"]))
            ys.append(float(p[key]))
        except Exception:
            continue
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / float(n)
    mean_y = sum(ys) / float(n)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return float(num / den)


def _arm_train(
    *,
    arm_label: str,
    root: Path,
    seed: int,
    device: str,
    ring_len: int,
    slot_dim: int,
    expert_heads: int,
    batch_size: int,
    ratio_fibo: str,
    router_equal: bool,
    capacity_equal: bool,
    steps: int,
    checkpoint_every: int,
    eval_n_realtime: int,
    eval_n_anchor: int,
    anchor_steps: set[int],
) -> Dict[str, Any]:
    run_root = root / arm_label
    train_root = run_root / "train"
    run_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["VAR_RUN_SEED"] = str(int(seed))
    env["VRX_SYNTH_ONCE"] = "1"

    init_cmd = [
        sys.executable,
        "-u",
        str(ENTRY_TOOL),
        "--run-root",
        str(run_root),
        "--device",
        str(device),
        "--ring-len",
        str(int(ring_len)),
        "--slot-dim",
        str(int(slot_dim)),
        "--expert-heads",
        str(int(expert_heads)),
        "--batch-size",
        str(int(batch_size)),
        "--synth-len",
        "256",
        "--assoc-keys",
        "64",
        "--assoc-pairs",
        "4",
        "--assoc-val-range",
        "256",
        "--max-steps",
        "2",
        "--ignore-max-steps",
        "0",
        "--ignore-wall-clock",
        "1",
        "--resume",
        "0",
        "--save-every-steps",
        "1",
        "--save-history",
        "0",
        "--eval-every-steps",
        "0",
        "--eval-at-checkpoint",
        "0",
        "--eval-samples",
        "0",
        "--offline-only",
        "1",
    ]
    # Keep expert *architecture* constant across arms by always enabling adapter experts.
    # Capacity "equal" vs "fibo" is expressed only via the split weights.
    cap_ratio = _equal_ratio(int(expert_heads)) if bool(capacity_equal) else str(ratio_fibo)
    init_cmd += [
        "--expert-capacity-split",
        str(cap_ratio),
        "--expert-capacity-total-mult",
        "1.0",
        "--expert-capacity-min-hidden",
        "8",
    ]
    _run(init_cmd, env=env)

    ckpt = train_root / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing init checkpoint: {ckpt}")

    route_ratio = _equal_ratio(int(expert_heads)) if bool(router_equal) else str(ratio_fibo)
    router_cmd = [
        sys.executable,
        str(ROUTER_TOOL),
        "--in-ckpt",
        str(ckpt),
        "--buckets",
        str(int(expert_heads)),
        "--ratio",
        str(route_ratio),
        "--permute-seed",
        "12345",
    ]
    rewire = _run_json(router_cmd, env=env)

    main_cmd = [
        sys.executable,
        "-u",
        str(ENTRY_TOOL),
        "--run-root",
        str(run_root),
        "--device",
        str(device),
        "--ring-len",
        str(int(ring_len)),
        "--slot-dim",
        str(int(slot_dim)),
        "--expert-heads",
        str(int(expert_heads)),
        "--batch-size",
        str(int(batch_size)),
        "--synth-len",
        "256",
        "--assoc-keys",
        "64",
        "--assoc-pairs",
        "4",
        "--assoc-val-range",
        "256",
        "--max-steps",
        str(int(steps)),
        "--ignore-max-steps",
        "0",
        "--ignore-wall-clock",
        "1",
        "--resume",
        "1",
        "--save-every-steps",
        str(int(checkpoint_every)),
        "--save-history",
        "1",
        "--eval-every-steps",
        "0",
        "--eval-at-checkpoint",
        "0",
        "--eval-samples",
        "0",
        "--offline-only",
        "1",
    ]
    main_cmd += [
        "--expert-capacity-split",
        str(cap_ratio),
        "--expert-capacity-total-mult",
        "1.0",
        "--expert-capacity-min-hidden",
        "8",
    ]
    _run(main_cmd, env=env)

    points: List[Dict[str, Any]] = []
    step_ckpts = _list_step_checkpoints(train_root, int(checkpoint_every), int(steps))
    final_ckpt = train_root / "checkpoint.pt"
    if final_ckpt.exists():
        final_str = str(final_ckpt.resolve())
        known = {str(p.resolve()) for p in step_ckpts}
        if final_str not in known:
            step_ckpts.append(final_ckpt)
    if not step_ckpts:
        if final_ckpt.exists():
            step_ckpts = [final_ckpt]

    for step_ckpt in step_ckpts:
        step = _checkpoint_step(step_ckpt)
        if step is None:
            step = int(steps)
        eval_out = _eval_checkpoint(train_root, step_ckpt, int(eval_n_realtime))
        points.append(
            {
                "step": int(step),
                "checkpoint": str(step_ckpt),
                "eval_acc": float(eval_out["eval_acc"]),
                "acc_delta": float(eval_out["acc_delta"]),
                "eval_loss": float(eval_out["eval_loss"]),
                "eval_n": int(eval_out["eval_n"]),
                "eval_seconds": float(eval_out["eval_seconds"]),
                "eval_tier": "realtime",
            }
        )
        should_anchor = int(eval_n_anchor) > 0 and int(step) in anchor_steps
        if should_anchor:
            anchor_out = _eval_checkpoint(train_root, step_ckpt, int(eval_n_anchor))
            points.append(
                {
                    "step": int(step),
                    "checkpoint": str(step_ckpt),
                    "eval_acc": float(anchor_out["eval_acc"]),
                    "acc_delta": float(anchor_out["acc_delta"]),
                    "eval_loss": float(anchor_out["eval_loss"]),
                    "eval_n": int(anchor_out["eval_n"]),
                    "eval_seconds": float(anchor_out["eval_seconds"]),
                    "eval_tier": "anchor",
                }
            )

    points = sorted(points, key=lambda x: (int(x["step"]), str(x.get("eval_tier", ""))))
    anchor_points = [p for p in points if str(p.get("eval_tier")) == "anchor"]
    realtime_points = [p for p in points if str(p.get("eval_tier")) == "realtime"]
    final_point = anchor_points[-1] if anchor_points else (realtime_points[-1] if realtime_points else {})
    slope_points = anchor_points if len(anchor_points) >= 2 else realtime_points
    sec_points = realtime_points if realtime_points else points
    return {
        "arm": str(arm_label),
        "run_root": str(run_root),
        "router_mode": "equal" if bool(router_equal) else "fibo",
        "capacity_mode": "equal" if bool(capacity_equal) else "fibo",
        "rewire": rewire,
        "points": points,
        "final_eval_tier": str(final_point.get("eval_tier", "realtime")) if final_point else None,
        "slope_acc_delta_per_step": float(_slope(slope_points, "acc_delta")),
        "avg_eval_seconds": float(sum(float(p["eval_seconds"]) for p in sec_points) / max(1, len(sec_points))),
        "final": final_point,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic speed-first 2x2 matrix runner.")
    ap.add_argument("--run-root", default="", help="Optional explicit output folder.")
    ap.add_argument("--label", default="hashmap_matrix_speed")
    ap.add_argument(
        "--arms",
        default="a,b,c,d",
        help="Comma subset from {a,b,c,d}; example: c,d",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ring-len", type=int, default=8192)
    ap.add_argument("--slot-dim", type=int, default=576)
    ap.add_argument("--expert-heads", type=int, default=3)
    ap.add_argument("--ratio", default="0.55,0.34,0.11")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=64, help="Deprecated alias for --eval-n-realtime.")
    ap.add_argument("--eval-n-realtime", type=int, default=64)
    ap.add_argument("--eval-n-anchor", type=int, default=256)
    ap.add_argument("--anchor-steps", default="")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    ratio_count = _ratio_len(args.ratio)
    if int(ratio_count) != int(args.expert_heads):
        raise ValueError(
            f"--ratio count ({ratio_count}) must equal --expert-heads ({int(args.expert_heads)})"
        )
    if int(args.steps) < int(args.checkpoint_every):
        raise ValueError("--steps must be >= --checkpoint-every")
    anchor_steps = _parse_int_set(args.anchor_steps)
    if not anchor_steps:
        anchor_steps = {int(args.steps)}
    eval_n_realtime = int(args.eval_n_realtime)
    # Backward compatibility: older callers may only pass --eval-n.
    if int(args.eval_n_realtime) == 64 and int(args.eval_n) != 64:
        eval_n_realtime = int(args.eval_n)
    eval_n_anchor = int(args.eval_n_anchor)
    if eval_n_realtime <= 0:
        raise ValueError("--eval-n-realtime must be > 0")
    if eval_n_anchor < 0:
        raise ValueError("--eval-n-anchor must be >= 0")

    if str(args.run_root).strip():
        root = Path(str(args.run_root)).resolve()
    else:
        stamp = _utc_stamp()
        root = (
            REPO_ROOT
            / "bench_vault"
            / "_tmp"
            / "vra79_hashmap_matrix"
            / f"{stamp}-{args.label}_seed{int(args.seed)}"
        )
    root.mkdir(parents=True, exist_ok=True)

    arm_map = {
        "a": ("arm_a_route_equal_capacity_equal", True, True),
        "b": ("arm_b_route_fibo_capacity_equal", False, True),
        "c": ("arm_c_route_equal_capacity_fibo", True, False),
        "d": ("arm_d_route_fibo_capacity_fibo", False, False),
    }
    selected = [p.strip().lower() for p in str(args.arms).split(",") if p.strip()]
    if not selected:
        raise ValueError("--arms must include at least one entry from {a,b,c,d}")
    if any(k not in arm_map for k in selected):
        raise ValueError(f"invalid --arms value: {args.arms}")

    existing_path = root / "matrix_summary.json"
    merged: Dict[str, Dict[str, Any]] = {}
    if existing_path.exists():
        try:
            prev = json.loads(existing_path.read_text(encoding="utf-8"))
            prev_arms = prev.get("arms", [])
            if isinstance(prev_arms, list):
                for arm in prev_arms:
                    if isinstance(arm, dict) and "arm" in arm:
                        merged[str(arm["arm"])] = arm
        except Exception:
            pass

    for key in selected:
        arm_label, router_equal, capacity_equal = arm_map[key]
        last_err: Exception | None = None
        for _attempt in (1, 2):
            try:
                merged[arm_label] = _arm_train(
                    arm_label=arm_label,
                    root=root,
                    seed=int(args.seed),
                    device=str(args.device),
                    ring_len=int(args.ring_len),
                    slot_dim=int(args.slot_dim),
                    expert_heads=int(args.expert_heads),
                    batch_size=int(args.batch_size),
                    ratio_fibo=str(args.ratio),
                    router_equal=bool(router_equal),
                    capacity_equal=bool(capacity_equal),
                    steps=int(args.steps),
                    checkpoint_every=int(args.checkpoint_every),
                    eval_n_realtime=int(eval_n_realtime),
                    eval_n_anchor=int(eval_n_anchor),
                    anchor_steps=anchor_steps,
                )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
        if last_err is not None:
            raise last_err

    ordered = [
        "arm_a_route_equal_capacity_equal",
        "arm_b_route_fibo_capacity_equal",
        "arm_c_route_equal_capacity_fibo",
        "arm_d_route_fibo_capacity_fibo",
    ]
    arm_results: List[Dict[str, Any]] = [merged[name] for name in ordered if name in merged]

    baseline_acc = 0.0
    for arm in arm_results:
        if str(arm.get("arm")) == "arm_a_route_equal_capacity_equal":
            baseline_acc = float(arm.get("final", {}).get("eval_acc", 0.0))
            break

    for arm in arm_results:
        final_acc = float(arm.get("final", {}).get("eval_acc", 0.0))
        arm["delta_vs_baseline_final_acc"] = float(final_acc - baseline_acc)

    rows: List[Dict[str, Any]] = []
    for arm in arm_results:
        for p in arm.get("points", []):
            rows.append(
                {
                    "arm": arm.get("arm"),
                    "router_mode": arm.get("router_mode"),
                    "capacity_mode": arm.get("capacity_mode"),
                    "step": p.get("step"),
                    "eval_acc": p.get("eval_acc"),
                    "acc_delta": p.get("acc_delta"),
                    "eval_loss": p.get("eval_loss"),
                    "eval_n": p.get("eval_n"),
                    "eval_seconds": p.get("eval_seconds"),
                    "eval_tier": p.get("eval_tier"),
                }
            )

    rows = sorted(rows, key=lambda x: (str(x["arm"]), int(x["step"])))
    csv_path = root / "matrix_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "router_mode",
                "capacity_mode",
                "step",
                "eval_acc",
                "acc_delta",
                "eval_loss",
                "eval_n",
                "eval_seconds",
                "eval_tier",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    winner = None
    if arm_results:
        winner = max(
            arm_results,
            key=lambda a: (
                float(a.get("final", {}).get("acc_delta", -1e9)),
                float(a.get("slope_acc_delta_per_step", -1e9)),
            ),
        )

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_root": str(root),
        "config": {
            "seed": int(args.seed),
            "device": str(args.device),
            "ring_len": int(args.ring_len),
            "slot_dim": int(args.slot_dim),
            "expert_heads": int(args.expert_heads),
            "ratio": str(args.ratio),
            "batch_size": int(args.batch_size),
            "steps": int(args.steps),
            "checkpoint_every": int(args.checkpoint_every),
            "eval_n_realtime": int(eval_n_realtime),
            "eval_n_anchor": int(eval_n_anchor),
            "anchor_steps": sorted(int(x) for x in anchor_steps),
        },
        "arms": arm_results,
        "winner_by_final_acc_delta_then_slope": winner.get("arm") if isinstance(winner, dict) else None,
        "matrix_summary_csv": str(csv_path),
    }
    out_path = root / "matrix_summary.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
