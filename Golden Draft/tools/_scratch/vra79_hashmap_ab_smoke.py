#!/usr/bin/env python3
"""Deterministic A/B smoke for equal-vs-hashmap capacity experts.

A-arm:
  - 3 experts, equal internal capacity.
  - Router ratio fixed (default 55/34/11).

B-arm:
  - Same router ratio.
  - Non-equal expert internal capacities (default 55/34/11).

Both arms use the same seed, same synthetic task, same max steps, and the same
postmortem eval tool to produce comparable evidence quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
ENTRY_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_saturation_entry.py"
ROUTER_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_router_map_fibo.py"
EVAL_TOOL = REPO_ROOT / r"Golden Draft\tools\eval_ckpt_assoc_byte.py"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ratio_len(raw: str) -> int:
    return len([p for p in str(raw).split(",") if p.strip()])


def _run(cmd: list[str], *, env: Dict[str, str] | None = None) -> None:
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if int(cp.returncode) != 0:
        raise RuntimeError(f"command failed rc={cp.returncode}: {' '.join(cmd)}")


def _run_json(cmd: list[str], *, env: Dict[str, str] | None = None) -> Dict[str, Any]:
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


def _post_eval(train_root: Path, eval_n: int) -> Dict[str, Any]:
    ckpt = train_root / "checkpoint_last_good.pt"
    if not ckpt.exists():
        ckpt = train_root / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint for eval: {train_root}")

    cmd = [
        sys.executable,
        str(EVAL_TOOL),
        "--run-root",
        str(train_root),
        "--checkpoint",
        str(ckpt),
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
    report = train_root / "report.json"
    if not report.exists():
        raise FileNotFoundError(f"missing report.json after eval: {train_root}")
    return json.loads(report.read_text(encoding="utf-8"))


def _run_arm(
    *,
    arm_label: str,
    root: Path,
    seed: int,
    device: str,
    ring_len: int,
    slot_dim: int,
    expert_heads: int,
    ratio: str,
    steps: int,
    eval_n: int,
    use_capacity_split: bool,
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
        "1",
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
        "--eval-every-steps",
        "0",
        "--eval-at-checkpoint",
        "0",
        "--eval-samples",
        "0",
        "--offline-only",
        "1",
    ]
    if use_capacity_split:
        init_cmd += [
            "--expert-capacity-split",
            str(ratio),
            "--expert-capacity-total-mult",
            "1.0",
            "--expert-capacity-min-hidden",
            "8",
        ]
    _run(init_cmd, env=env)

    ckpt = train_root / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing init checkpoint: {ckpt}")

    router_cmd = [
        sys.executable,
        str(ROUTER_TOOL),
        "--in-ckpt",
        str(ckpt),
        "--buckets",
        str(int(_ratio_len(ratio))),
        "--ratio",
        str(ratio),
        "--permute-seed",
        "12345",
    ]
    rewire_info = _run_json(router_cmd, env=env)

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
        "1",
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
        "10",
        "--eval-every-steps",
        "0",
        "--eval-at-checkpoint",
        "0",
        "--eval-samples",
        str(int(eval_n)),
        "--offline-only",
        "1",
    ]
    if use_capacity_split:
        main_cmd += [
            "--expert-capacity-split",
            str(ratio),
            "--expert-capacity-total-mult",
            "1.0",
            "--expert-capacity-min-hidden",
            "8",
        ]
    _run(main_cmd, env=env)

    report = _post_eval(train_root, eval_n=int(eval_n))
    return {
        "arm": str(arm_label),
        "run_root": str(run_root),
        "rewire": rewire_info,
        "report": report,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run deterministic A/B smoke for hashmap-capacity experts.")
    ap.add_argument("--run-root", default="", help="Optional explicit output folder.")
    ap.add_argument("--label", default="hashmap_ab_smoke")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ring-len", type=int, default=8192)
    ap.add_argument("--slot-dim", type=int, default=576)
    ap.add_argument("--expert-heads", type=int, default=3)
    ap.add_argument("--ratio", default="0.55,0.34,0.11")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=64)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    ratio_count = _ratio_len(args.ratio)
    if int(ratio_count) != int(args.expert_heads):
        raise ValueError(
            f"--ratio count ({ratio_count}) must equal --expert-heads ({int(args.expert_heads)})"
        )

    if str(args.run_root).strip():
        root = Path(str(args.run_root)).resolve()
    else:
        stamp = _utc_stamp()
        root = (
            REPO_ROOT
            / "bench_vault"
            / "_tmp"
            / "vra79_hashmap_ab"
            / f"{stamp}-{args.label}_seed{int(args.seed)}"
        )
    root.mkdir(parents=True, exist_ok=True)

    arm_a = _run_arm(
        arm_label="arm_a_equal",
        root=root,
        seed=int(args.seed),
        device=str(args.device),
        ring_len=int(args.ring_len),
        slot_dim=int(args.slot_dim),
        expert_heads=int(args.expert_heads),
        ratio=str(args.ratio),
        steps=int(args.steps),
        eval_n=int(args.eval_n),
        use_capacity_split=False,
    )
    arm_b = _run_arm(
        arm_label="arm_b_hashmap_capacity",
        root=root,
        seed=int(args.seed),
        device=str(args.device),
        ring_len=int(args.ring_len),
        slot_dim=int(args.slot_dim),
        expert_heads=int(args.expert_heads),
        ratio=str(args.ratio),
        steps=int(args.steps),
        eval_n=int(args.eval_n),
        use_capacity_split=True,
    )

    rep_a = arm_a.get("report", {}) if isinstance(arm_a, dict) else {}
    rep_b = arm_b.get("report", {}) if isinstance(arm_b, dict) else {}
    eval_a = rep_a.get("eval", {}) if isinstance(rep_a.get("eval"), dict) else {}
    eval_b = rep_b.get("eval", {}) if isinstance(rep_b.get("eval"), dict) else {}
    acc_a = float(eval_a.get("eval_acc", 0.0))
    acc_b = float(eval_b.get("eval_acc", 0.0))
    n_a = int(eval_a.get("eval_n", 0))
    n_b = int(eval_b.get("eval_n", 0))
    delta = float(acc_b - acc_a)

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
            "steps": int(args.steps),
            "eval_n": int(args.eval_n),
        },
        "arms": [arm_a, arm_b],
        "comparison": {
            "eval_acc_a_equal": float(acc_a),
            "eval_acc_b_hashmap_capacity": float(acc_b),
            "eval_n_a": int(n_a),
            "eval_n_b": int(n_b),
            "eval_acc_delta_b_minus_a": float(delta),
        },
    }
    out_path = root / "ab_summary.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

