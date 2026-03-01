#!/usr/bin/env python3
"""Consolidate all ARCHIVE/ training runs into a single flat archive.csv.

Usage:
    python tools/archive_csv.py                # archive only
    python tools/archive_csv.py --include-active   # also include training_output/

Output: ARCHIVE/archive.csv  (overwritten each run — idempotent)
"""

import csv
import os
import sys
from pathlib import Path

_V4_ROOT = Path(__file__).resolve().parent.parent          # v4/
_ARCHIVE_DIR = _V4_ROOT / "ARCHIVE"
_ACTIVE_DIR = _V4_ROOT / "training_output"
_OUT_CSV = _ARCHIVE_DIR / "archive.csv"

# ── column order ────────────────────────────────────────────────────
RUN_COLS = [
    "run_id", "run_folder", "timestamp", "total_steps", "best_loss",
    "n_params", "hidden_dim", "slot_dim", "M", "N",
    "embed_encoding", "output_encoding",
    "lr_max", "batch_size", "seq_len", "warmup_steps", "data_type", "notes",
]
STEP_COLS = [
    "step", "raw_loss", "masked_loss", "accuracy", "masked_acc",
    "lr_actual", "elapsed_s", "samples_seen", "mask_frac",
]
ALL_COLS = RUN_COLS + STEP_COLS


# ── helpers ─────────────────────────────────────────────────────────

def _find_first_checkpoint(folder: Path):
    """Return path to earliest .pt checkpoint, or None."""
    pts = sorted(folder.glob("ckpt_step_*.pt"))
    return pts[0] if pts else None


def _infer_encoding(state_dict: dict, table_key: str, learned_key: str) -> str:
    """Infer 'learned', 'hadamard', 'sincos', or 'unknown'."""
    if learned_key in state_dict:
        return "learned"
    if table_key not in state_dict:
        return "unknown"
    import torch
    table = state_dict[table_key]
    abs_vals = table.abs()
    if torch.allclose(abs_vals, abs_vals[0, 0].expand_as(abs_vals), atol=1e-5):
        return "hadamard"
    return "sincos"


def _infer_data_type(config: dict) -> str:
    data_dir = config.get("data_dir", "")
    low = str(data_dir).lower().replace("\\", "/")
    if "echo" in low:
        return "echo"
    if "real" in low or "code" in low or "shard" in low:
        return "real_code"
    # fallback: check if the default training_data/ had echo or real files
    return "unknown"


def infer_model_meta(ckpt_path: Path) -> dict:
    """Load checkpoint and extract run-level metadata."""
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", {})
    cfg = ckpt.get("config", {})

    # architecture from tensor shapes
    hidden_dim = ""
    slot_dim = ""
    M = ""
    N = 0
    for k in sd:
        if k.startswith("read_proj.") and k.endswith(".weight"):
            idx = int(k.split(".")[1])
            N = max(N, idx + 1)
            hidden_dim = sd[k].shape[0]
            slot_dim = sd[k].shape[1]
    if "dests" in sd:
        M = sd["dests"].shape[1]

    # param count (only learnable — exclude buffers that are integer or bool)
    n_params = 0
    for v in sd.values():
        if v.is_floating_point():
            n_params += v.numel()

    return {
        "timestamp": ckpt.get("timestamp", ""),
        "best_loss": ckpt.get("best_loss", ""),
        "n_params": n_params,
        "hidden_dim": hidden_dim,
        "slot_dim": slot_dim,
        "M": M,
        "N": N or "",
        "embed_encoding": _infer_encoding(sd, "_fixed_table", "inp.weight"),
        "output_encoding": _infer_encoding(sd, "_fixed_output_table", "out.weight"),
        "lr_max": cfg.get("lr", ""),
        "batch_size": cfg.get("batch_size", ""),
        "seq_len": cfg.get("seq_len", ""),
        "warmup_steps": cfg.get("warmup_steps", ""),
        "data_type": _infer_data_type(cfg),
    }


def read_train_log(csv_path: Path) -> list[dict]:
    """Read train_log.csv, return list of row dicts with renamed lr column."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step": row.get("step", ""),
                "raw_loss": row.get("raw_loss", ""),
                "masked_loss": row.get("masked_loss", ""),
                "accuracy": row.get("accuracy", ""),
                "masked_acc": row.get("masked_acc", ""),
                "lr_actual": row.get("lr", ""),
                "elapsed_s": row.get("elapsed_s", ""),
                "samples_seen": row.get("samples_seen", ""),
                "mask_frac": row.get("mask_frac", ""),
            })
    return rows


def read_notes(folder: Path) -> str:
    notes_file = folder / "notes.txt"
    if notes_file.exists():
        return notes_file.read_text(encoding="utf-8").strip().split("\n")[0]
    return ""


def discover_runs(include_active: bool = False):
    """Yield (folder_name, csv_path, ckpt_path_or_None, notes) for each run."""
    if not _ARCHIVE_DIR.exists():
        return
    for entry in sorted(_ARCHIVE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        csv_path = entry / "train_log.csv"
        if not csv_path.exists():
            continue
        yield (entry.name, csv_path, _find_first_checkpoint(entry), read_notes(entry))

    if include_active and _ACTIVE_DIR.exists():
        csv_path = _ACTIVE_DIR / "train_log.csv"
        if csv_path.exists():
            yield ("(active)", csv_path, _find_first_checkpoint(_ACTIVE_DIR), "")


def main():
    include_active = "--include-active" in sys.argv

    # collect all runs
    runs = []
    for folder_name, csv_path, ckpt_path, notes in discover_runs(include_active):
        meta = {}
        if ckpt_path:
            print(f"  Loading checkpoint: {ckpt_path.name} ...", end=" ", flush=True)
            meta = infer_model_meta(ckpt_path)
            print("OK")
        else:
            print(f"  No checkpoint in {folder_name}, using CSV only")
            meta = {k: "" for k in [
                "timestamp", "best_loss", "n_params", "hidden_dim", "slot_dim",
                "M", "N", "embed_encoding", "output_encoding",
                "lr_max", "batch_size", "seq_len", "warmup_steps", "data_type",
            ]}

        step_rows = read_train_log(csv_path)
        total_steps = int(step_rows[-1]["step"]) if step_rows else 0

        runs.append({
            "folder_name": folder_name,
            "notes": notes,
            "meta": meta,
            "total_steps": total_steps,
            "step_rows": step_rows,
        })

    # sort by timestamp (earliest first), fallback to folder name
    runs.sort(key=lambda r: r["meta"].get("timestamp", "") or r["folder_name"])

    # write CSV
    total_rows = 0
    with open(_OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()

        for run_id, run in enumerate(runs, start=1):
            run_meta = {
                "run_id": run_id,
                "run_folder": run["folder_name"],
                "timestamp": run["meta"].get("timestamp", ""),
                "total_steps": run["total_steps"],
                "best_loss": run["meta"].get("best_loss", ""),
                "n_params": run["meta"].get("n_params", ""),
                "hidden_dim": run["meta"].get("hidden_dim", ""),
                "slot_dim": run["meta"].get("slot_dim", ""),
                "M": run["meta"].get("M", ""),
                "N": run["meta"].get("N", ""),
                "embed_encoding": run["meta"].get("embed_encoding", ""),
                "output_encoding": run["meta"].get("output_encoding", ""),
                "lr_max": run["meta"].get("lr_max", ""),
                "batch_size": run["meta"].get("batch_size", ""),
                "seq_len": run["meta"].get("seq_len", ""),
                "warmup_steps": run["meta"].get("warmup_steps", ""),
                "data_type": run["meta"].get("data_type", ""),
                "notes": run["notes"],
            }

            for step_row in run["step_rows"]:
                row = {**run_meta, **step_row}
                writer.writerow(row)
                total_rows += 1

    print(f"\nWrote {_OUT_CSV}")
    print(f"  {len(runs)} runs, {total_rows} total rows")


if __name__ == "__main__":
    print(f"archive_csv — scanning {_ARCHIVE_DIR}\n")
    main()
