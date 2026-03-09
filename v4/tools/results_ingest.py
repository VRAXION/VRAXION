from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_OUTPUT = REPO_ROOT / "training_output"
RESULTS_ROOT = REPO_ROOT / "results"
DERIVED_ROOT = RESULTS_ROOT / "derived"
MASTER_CSV = RESULTS_ROOT / "runs_master.csv"
GOLDEN_CSV = DERIVED_ROOT / "runs_golden.csv"
QUARANTINED_CSV = DERIVED_ROOT / "runs_quarantined.csv"


MASTER_FIELDS = [
    "run_root",
    "family",
    "comparison_group",
    "golden_status",
    "repo_head",
    "train_log_path",
    "ckpt_path",
    "final_step",
    "elapsed_s",
    "sec_per_step",
    "samples_seen",
    "final_mask_frac",
    "best_raw_loss",
    "best_masked_loss",
    "final_raw_loss",
    "final_masked_loss",
    "best_accuracy",
    "best_masked_acc",
    "final_accuracy",
    "final_masked_acc",
    "final_alpha_0_mean",
    "final_ring_norm",
    "final_ring_slot_mean",
    "final_ptr_pos_0",
    "params_total",
    "checkpoint_best_loss",
    "checkpoint_step",
    "checkpoint_timestamp_utc",
    "hidden_dim",
    "slot_dim",
    "M",
    "N",
    "R",
    "embed_encoding",
    "output_encoding",
    "pointer_mode",
    "write_mode",
    "replace_impl",
    "min_write_strength",
    "mtaps_enabled",
    "mtaps_mixer_mode",
    "bb_enabled",
    "notes",
]


def git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        )
    except Exception:
        return ""
    return out.strip()


def family_for_run(run_root: str) -> str:
    if run_root.startswith("ab_cpu_"):
        return "cpu_ab"
    if run_root.startswith("needle_"):
        return "needle"
    if run_root.startswith("long_run_"):
        return "long_run"
    if run_root.startswith("test_"):
        return "test"
    return "scratch"


def comparison_group_for_run(run_root: str) -> str:
    if run_root in {"ab_cpu_bitlift", "ab_cpu_learned"}:
        return "cpu_embed_ab"
    if run_root in {"ab_cpu_current_seq", "ab_cpu_current_pilot"}:
        return "cpu_pointer_ab"
    if run_root == "long_run_configC":
        return "long_run_configC"
    if run_root in {"needle_N1", "needle_N2", "needle_N3"}:
        return "needle_expert_count"
    if run_root == "needle_RING_OFF":
        return "needle_ring_ablation"
    if run_root == "needle_S0_nulltest":
        return "needle_read_gate_ablation"
    if run_root.startswith("needle_slot"):
        return "needle_slot_width"
    if run_root.startswith("needle_pilot_m") or run_root == "needle_maxjump64":
        return "needle_pointer_memory"
    if run_root.startswith("test_"):
        return "test"
    return "scratch"


def golden_status_for_run(run_root: str) -> str:
    if run_root in {
        "ab_cpu_bitlift",
        "ab_cpu_learned",
        "ab_cpu_current_seq",
        "ab_cpu_current_pilot",
        "long_run_configC",
    }:
        return "golden"
    if run_root.startswith("needle_"):
        return "golden"
    return "quarantine"


def read_train_log(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def count_params(state_dict: dict[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel())
    return total


def summarize_run(run_dir: Path, head_sha: str) -> dict[str, Any] | None:
    train_log = run_dir / "train_log.csv"
    ckpt = run_dir / "ckpt_latest.pt"
    if not train_log.exists() or not ckpt.exists():
        return None

    rows = read_train_log(train_log)
    if not rows:
        return None

    last = rows[-1]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_rec = ck.get("model", {})
    build_spec = model_rec.get("build_spec", {})
    state_dict = model_rec.get("state_dict", {})

    best_raw_loss = min(parse_float(r.get("raw_loss")) for r in rows)
    best_masked_loss = min(parse_float(r.get("masked_loss")) for r in rows)
    best_accuracy = max(parse_float(r.get("accuracy")) for r in rows)
    best_masked_acc = max(parse_float(r.get("masked_acc")) for r in rows)

    elapsed_s = parse_float(last.get("elapsed_s"))
    final_step = parse_int(last.get("step"))
    sec_per_step = elapsed_s / final_step if elapsed_s > 0 and final_step > 0 else 0.0

    run_root = run_dir.name

    return {
        "run_root": run_root,
        "family": family_for_run(run_root),
        "comparison_group": comparison_group_for_run(run_root),
        "golden_status": golden_status_for_run(run_root),
        "repo_head": head_sha,
        "train_log_path": str(train_log.relative_to(REPO_ROOT)).replace("\\", "/"),
        "ckpt_path": str(ckpt.relative_to(REPO_ROOT)).replace("\\", "/"),
        "final_step": final_step,
        "elapsed_s": f"{elapsed_s:.4f}",
        "sec_per_step": f"{sec_per_step:.4f}",
        "samples_seen": parse_int(last.get("samples_seen")),
        "final_mask_frac": f"{parse_float(last.get('mask_frac')):.6f}",
        "best_raw_loss": f"{best_raw_loss:.6f}",
        "best_masked_loss": f"{best_masked_loss:.6f}",
        "final_raw_loss": f"{parse_float(last.get('raw_loss')):.6f}",
        "final_masked_loss": f"{parse_float(last.get('masked_loss')):.6f}",
        "best_accuracy": f"{best_accuracy:.6f}",
        "best_masked_acc": f"{best_masked_acc:.6f}",
        "final_accuracy": f"{parse_float(last.get('accuracy')):.6f}",
        "final_masked_acc": f"{parse_float(last.get('masked_acc')):.6f}",
        "final_alpha_0_mean": f"{parse_float(last.get('alpha_0_mean')):.6f}",
        "final_ring_norm": f"{parse_float(last.get('ring_norm')):.6f}",
        "final_ring_slot_mean": f"{parse_float(last.get('ring_slot_mean')):.6f}",
        "final_ptr_pos_0": f"{parse_float(last.get('ptr_pos_0')):.2f}",
        "params_total": count_params(state_dict),
        "checkpoint_best_loss": f"{parse_float(ck.get('best_loss')):.6f}",
        "checkpoint_step": parse_int(ck.get("step")),
        "checkpoint_timestamp_utc": ck.get("timestamp_utc", ""),
        "hidden_dim": build_spec.get("hidden_dim", ""),
        "slot_dim": build_spec.get("slot_dim", ""),
        "M": build_spec.get("M", ""),
        "N": build_spec.get("N", ""),
        "R": build_spec.get("R", ""),
        "embed_encoding": build_spec.get("embed_encoding", ""),
        "output_encoding": build_spec.get("output_encoding", ""),
        "pointer_mode": build_spec.get("pointer_mode", ""),
        "write_mode": build_spec.get("write_mode", ""),
        "replace_impl": build_spec.get("replace_impl", ""),
        "min_write_strength": build_spec.get("min_write_strength", ""),
        "mtaps_enabled": build_spec.get("mtaps_enabled", ""),
        "mtaps_mixer_mode": build_spec.get("mtaps_mixer_mode", ""),
        "bb_enabled": build_spec.get("bb_enabled", ""),
        "notes": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    head_sha = git_head()
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(TRAINING_OUTPUT.iterdir()):
        if not run_dir.is_dir():
            continue
        row = summarize_run(run_dir, head_sha)
        if row is not None:
            rows.append(row)

    rows.sort(
        key=lambda r: (
            r["golden_status"] != "golden",
            r["family"],
            r["comparison_group"],
            r["run_root"],
        )
    )

    write_csv(MASTER_CSV, rows)
    write_csv(GOLDEN_CSV, [r for r in rows if r["golden_status"] == "golden"])
    write_csv(QUARANTINED_CSV, [r for r in rows if r["golden_status"] != "golden"])

    print(f"wrote {MASTER_CSV}")
    print(f"wrote {GOLDEN_CSV}")
    print(f"wrote {QUARANTINED_CSV}")
    print(f"rows={len(rows)} golden={sum(r['golden_status']=='golden' for r in rows)}")


if __name__ == "__main__":
    main()
