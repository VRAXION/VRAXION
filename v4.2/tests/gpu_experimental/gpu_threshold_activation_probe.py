"""Deterministic GPU probe for English threshold/signal calibration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.gpu_experimental.gpu_english_common import (
    DEFAULT_DATA_PATH,
    DEFAULT_TICKS,
    PROBE_BYTES,
    eval_sequence_batch,
    gather_seq_batch,
    load_all_data,
    load_checkpoint_init,
    make_fixed_eval_sequences,
    make_output_path,
    probe_threshold_dynamics,
    resolve_checkpoint_path,
    resolve_data_path,
    seq_batch_to_device,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="", help="Directory containing english_768n checkpoints")
    ap.add_argument(
        "--checkpoints",
        default="english_768n_step1000.npz,english_768n_step3000.npz,english_768n_step5000.npz",
        help="Comma-separated checkpoint file names",
    )
    ap.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--thresholds", default="0.0,0.1,0.25,0.5")
    ap.add_argument("--ticks-list", default="1,3,6")
    ap.add_argument("--probe-bytes", default="32,97,101,116")
    ap.add_argument("--eval-seed", type=int, default=9999)
    ap.add_argument("--output", default="")
    return ap.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_step(name: str) -> int:
    stem = Path(name).stem
    for part in stem.split("_"):
        if part.startswith("step"):
            return int(part.replace("step", ""))
    raise ValueError(f"Could not parse step from checkpoint name: {name}")


def runs_equal(a: dict[str, object], b: dict[str, object]) -> bool:
    return (
        abs(float(a["eval_acc"]) - float(b["eval_acc"])) <= 1e-9
        and abs(float(a["avg_target_prob"]) - float(b["avg_target_prob"])) <= 1e-9
        and a["active_neurons_per_tick"] == b["active_neurons_per_tick"]
        and a["mask_hash"] == b["mask_hash"]
    )


def run_case(
    *,
    checkpoint_name: str,
    checkpoint_dir: str,
    data_path: Path,
    threshold: float,
    ticks: int,
    probe_bytes: list[int],
    eval_seed: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_name, checkpoint_dir)
    init = load_checkpoint_init(checkpoint_path, device=device)
    all_data = load_all_data(data_path)
    eval_batch_np = make_fixed_eval_sequences(all_data, seed=eval_seed)
    eval_batch = seq_batch_to_device(eval_batch_np, device)

    eval_metrics = eval_sequence_batch(
        mask=init.mask,
        w_in=init.w_in,
        w_out=init.w_out,
        bp=init.bp,
        bp_norm=init.bp_norm,
        seq_batch=eval_batch,
        retention=init.retention,
        threshold=threshold,
        ticks=ticks,
    )
    probe_metrics = probe_threshold_dynamics(
        mask=init.mask,
        w_in=init.w_in,
        bp=init.bp,
        probe_bytes=probe_bytes,
        retention=init.retention,
        threshold=threshold,
        ticks=ticks,
    )
    result = {
        "checkpoint": checkpoint_name,
        "checkpoint_step": parse_step(checkpoint_name),
        "threshold": threshold,
        "ticks": ticks,
        "eval_acc": eval_metrics["acc"],
        "avg_target_prob": eval_metrics["avg_target_prob"],
        "charge_max_per_tick": probe_metrics["charge_max_per_tick"],
        "act_max_per_tick": probe_metrics["act_max_per_tick"],
        "active_neurons_per_tick": probe_metrics["active_neurons_per_tick"],
        "active_ratio_per_tick": probe_metrics["active_ratio_per_tick"],
        "newly_active_after_tick0": probe_metrics["newly_active_after_tick0"],
        "mask_hash": init.mask_hash,
        "probe_details": probe_metrics["probe_details"],
    }
    return result, {
        "retention": init.retention,
        "neurons": init.neurons,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = resolve_data_path(args.data_path)
    checkpoints = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    thresholds = parse_csv_floats(args.thresholds)
    ticks_list = parse_csv_ints(args.ticks_list)
    probe_bytes = parse_csv_ints(args.probe_bytes) if args.probe_bytes else PROBE_BYTES

    results: list[dict[str, object]] = []
    meta = None
    for checkpoint_name in checkpoints:
        for threshold in thresholds:
            for ticks in ticks_list:
                first, info = run_case(
                    checkpoint_name=checkpoint_name,
                    checkpoint_dir=args.checkpoint_dir,
                    data_path=data_path,
                    threshold=threshold,
                    ticks=ticks,
                    probe_bytes=probe_bytes,
                    eval_seed=args.eval_seed,
                    device=device,
                )
                second, _ = run_case(
                    checkpoint_name=checkpoint_name,
                    checkpoint_dir=args.checkpoint_dir,
                    data_path=data_path,
                    threshold=threshold,
                    ticks=ticks,
                    probe_bytes=probe_bytes,
                    eval_seed=args.eval_seed,
                    device=device,
                )
                row = dict(first)
                row["deterministic"] = runs_equal(first, second)
                results.append(row)
                meta = info
                print(
                    f"{checkpoint_name} thr={threshold:.2f} ticks={ticks} "
                    f"eval={row['eval_acc']*100:.2f}% active_t0={row['active_ratio_per_tick'][0]*100:.2f}% "
                    f"active_tlast={row['active_ratio_per_tick'][-1]*100:.2f}% det={row['deterministic']}"
                )

    latest_checkpoint = max(checkpoints, key=parse_step)
    baseline_rows = [
        r for r in results if r["checkpoint"] == latest_checkpoint and r["threshold"] == 0.5 and r["ticks"] == 6
    ]
    if not baseline_rows:
        raise RuntimeError("Missing baseline row for latest checkpoint at threshold=0.5 ticks=6")
    latest_baseline = baseline_rows[0]
    candidate_rows = [
        r
        for r in results
        if r["checkpoint"] == latest_checkpoint
        and r["ticks"] == 6
        and r["threshold"] != 0.5
        and r["active_ratio_per_tick"][-1] >= 0.05
        and r["eval_acc"] >= latest_baseline["eval_acc"] - 0.005
    ]
    candidate_rows.sort(key=lambda r: (r["eval_acc"], r["active_ratio_per_tick"][-1]), reverse=True)

    threshold_problem_votes = 0
    baseline_rows_all = [r for r in results if r["threshold"] == 0.5]
    for checkpoint_name in checkpoints:
        row = next(r for r in baseline_rows_all if r["checkpoint"] == checkpoint_name and r["ticks"] == 6)
        act_quiet = all(float(x) <= 1e-6 for x in row["act_max_per_tick"])
        sparse = all(float(x) < 0.01 for x in row["active_ratio_per_tick"])
        if act_quiet and sparse:
            threshold_problem_votes += 1

    verdict = {
        "threshold_problem_confirmed": threshold_problem_votes >= 2,
        "threshold_problem_votes": threshold_problem_votes,
        "candidate_thresholds": [float(r["threshold"]) for r in candidate_rows[:2]],
        "latest_checkpoint": latest_checkpoint,
        "baseline_latest_eval_acc": latest_baseline["eval_acc"],
    }

    output = {
        "branch_intent": "threshold_signal_calibration",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "device": str(device),
        "args": {
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoints": checkpoints,
            "data_path": str(data_path),
            "thresholds": thresholds,
            "ticks_list": ticks_list,
            "probe_bytes": probe_bytes,
            "eval_seed": args.eval_seed,
        },
        "meta": meta or {},
        "results": results,
        "verdict": verdict,
    }

    output_path = make_output_path(args.output, "gpu_threshold_activation_probe.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path}")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
