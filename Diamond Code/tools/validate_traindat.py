#!/usr/bin/env python3
"""Validate traindat files against model config before training.

Checks:
  1. Every .traindat has a .meta.json sidecar
  2. Metadata is well-formed (required fields present)
  3. Minimum seq_len for each task vs model's actual seq_len
  4. Active data_weights in controls.json don't include impossible tasks
  5. Corpus size is sufficient for seq_len * num_bits

Usage:
  python tools/validate_traindat.py data/traindat/ --seq_len 16 --num_bits 8
  python tools/validate_traindat.py data/traindat/ --controls logs/swarm/controls.json
  python tools/validate_traindat.py data/traindat/  # uses defaults from run_d128.bat
"""

import json
import os
import sys
from pathlib import Path

# ── Minimum seq_len per task ──────────────────────────────────────────────
# Computed from generator logic: what's the minimum context window needed
# for the model to have ANY chance of learning the pattern?

def compute_min_seq_len(meta: dict) -> int:
    """Return minimum seq_len needed to learn this task's pattern.

    IMPORTANT: This model is AUTOREGRESSIVE (causal). At position t, the model
    only sees inputs[0:t] through ring memory — NOT the current position's target.
    For block-based tasks, the model needs to have seen a FULL block in the PAST
    before it can predict the next block. This means min_seq_len = 2 * block_size
    for most block-based tasks (not 1 * block_size as a bidirectional model would need).
    """
    task = meta.get("task", "")
    block_size = meta.get("block_size", 16)
    gap_bytes = meta.get("gap_bytes", 0)

    if task in ("constant", "copy_echo"):
        return 1  # trivial: predict same byte (or zero)
    elif task == "echo":
        return 2 * block_size  # autoregressive: need full period in PAST context
    elif task in ("bitwise_not", "denoise"):
        return 2 * block_size  # need source block fully in past to compute NOT/denoise
    elif task == "counter":
        return 2 * block_size  # need previous block fully in past to increment
    elif task == "fibonacci":
        return 3 * block_size  # needs block[t-2] and block[t-1] fully in past
    elif task == "byte_rotation":
        return 2 * block_size + 1  # rotation needs past block + boundary
    elif task == "arithmetic":
        # [A:8B][B:8B][(A+B):8B][|A-B|:8B] → need A and B fully in past
        return block_size * 4  # need A and B blocks complete before predicting A+B
    elif task == "delay_echo":
        return gap_bytes + 2 * block_size  # original must be fully in past window
    else:
        return 2 * block_size  # conservative default for autoregressive


def compute_min_ring_size(meta: dict) -> int:
    """Return minimum ring/memory_size needed to hold task state.

    The ring memory is a circular buffer of the last N hidden states.
    For block-based tasks, the ring must hold at least one full block
    so the model can reference all positions within a block period.
    For tasks needing multiple blocks (fibonacci), ring must hold more.
    """
    task = meta.get("task", "")
    block_size = meta.get("block_size", 16)

    if task in ("constant", "copy_echo"):
        return 1  # no history needed
    elif task == "fibonacci":
        return 2 * block_size  # needs 2 prior blocks in memory
    elif task == "arithmetic":
        return 2 * block_size  # needs A and B blocks in memory
    elif task == "delay_echo":
        gap_bytes = meta.get("gap_bytes", 0)
        return gap_bytes + block_size  # must span the gap + original block
    else:
        return block_size  # most block-based tasks need 1 full block in ring


def difficulty_label(min_seq: int, model_seq: int) -> str:
    """Human-readable difficulty relative to model's seq_len."""
    if min_seq > model_seq:
        return "IMPOSSIBLE"
    ratio = min_seq / model_seq
    if ratio <= 0.1:
        return "TRIVIAL"
    elif ratio <= 0.5:
        return "EASY"
    elif ratio <= 0.8:
        return "MODERATE"
    else:
        return "HARD"


# ── Validation checks ────────────────────────────────────────────────────

def validate_file(traindat_path: Path, model_seq_len: int, num_bits: int,
                   model_memory_size: int = 0):
    """Validate a single .traindat file. Returns (ok, warnings, errors, meta, min_seq, diff)."""
    warnings = []
    errors = []
    name = traindat_path.name
    meta_path = traindat_path.with_suffix(".meta.json")

    # Check 1: meta.json exists
    if not meta_path.exists():
        errors.append(f"missing {meta_path.name}")
        return False, warnings, errors

    # Check 2: meta.json is valid JSON with required fields
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        errors.append(f"bad metadata: {e}")
        return False, warnings, errors

    required = ["task", "generator"]
    for field in required:
        if field not in meta:
            errors.append(f"missing field '{field}' in metadata")

    if errors:
        return False, warnings, errors

    # Check 3: corpus size sufficient
    file_size = traindat_path.stat().st_size
    min_corpus = (model_seq_len + 1) * num_bits + num_bits
    if file_size < min_corpus:
        errors.append(
            f"too small: {file_size:,} bytes < {min_corpus:,} needed "
            f"(seq_len={model_seq_len}, num_bits={num_bits})"
        )

    # Check 4: task solvability at model's seq_len
    min_seq = compute_min_seq_len(meta)
    diff = difficulty_label(min_seq, model_seq_len)

    if min_seq > model_seq_len:
        errors.append(
            f"IMPOSSIBLE at seq_len={model_seq_len}: "
            f"task '{meta['task']}' needs min_seq_len={min_seq} "
            f"(gap={meta.get('gap_bytes', 'N/A')}, block={meta.get('block_size', 'N/A')})"
        )

    # Check 5: ring memory large enough to hold task state
    if model_memory_size > 0:
        min_ring = compute_min_ring_size(meta)
        if min_ring > model_memory_size:
            errors.append(
                f"RING TOO SMALL: memory_size={model_memory_size} < {min_ring} needed "
                f"for task '{meta['task']}' (block={meta.get('block_size', 'N/A')})"
            )

    return len(errors) == 0, warnings, errors, meta, min_seq, diff


def validate_controls(controls_path: Path, file_results: dict, model_seq_len: int):
    """Check if any active data_weights point to impossible tasks."""
    errors = []
    warnings = []

    if not controls_path.exists():
        warnings.append(f"controls.json not found at {controls_path}")
        return warnings, errors

    try:
        with open(controls_path, "r") as f:
            controls = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        errors.append(f"bad controls.json: {e}")
        return warnings, errors

    data_weights = controls.get("data_weights", {})
    active = {k: v for k, v in data_weights.items() if v > 0}

    if not active:
        warnings.append("no active data_weights (all weights are 0)")
        return warnings, errors

    for fname, weight in active.items():
        if fname in file_results:
            result = file_results[fname]
            if not result["ok"]:
                errors.append(
                    f"ACTIVE in controls.json (weight={weight}) but IMPOSSIBLE: "
                    f"{fname} needs min_seq_len={result['min_seq']} > {model_seq_len}"
                )
        else:
            warnings.append(f"active weight for '{fname}' but file not found in data_dir")

    return warnings, errors


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate traindat files against model config"
    )
    parser.add_argument("data_dir", type=str, help="Path to traindat directory")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Model seq_len (default: 16)")
    parser.add_argument("--num_bits", type=int, default=8,
                        help="Model num_bits (default: 8)")
    parser.add_argument("--memory_size", type=int, default=0,
                        help="Model memory_size / ring length (0=skip ring check)")
    parser.add_argument("--controls", type=str, default=None,
                        help="Path to controls.json (checks active weights)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit code 1 if any active task is impossible")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    # Find all .traindat files
    traindat_files = sorted(data_dir.glob("*.traindat"))
    if not traindat_files:
        print(f"ERROR: no .traindat files in {data_dir}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  TRAINDAT VALIDATION")
    print(f"  data_dir:      {data_dir}")
    print(f"  seq_len:       {args.seq_len}")
    print(f"  num_bits:      {args.num_bits}")
    if args.memory_size > 0:
        print(f"  memory_size:   {args.memory_size}")
    print(f"{'='*70}")
    print()

    # Validate each file
    file_results = {}
    n_ok = 0
    n_fail = 0

    # Header
    print(f"  {'File':<30s} {'Task':<15s} {'MinSeq':>6s} {'Diff':<12s} {'Status'}")
    print(f"  {'-'*30} {'-'*15} {'-'*6} {'-'*12} {'-'*10}")

    for fp in traindat_files:
        result = validate_file(fp, args.seq_len, args.num_bits, args.memory_size)
        if len(result) == 6:
            ok, warns, errs, meta, min_seq, diff = result
        else:
            ok, warns, errs = result
            meta, min_seq, diff = {}, 0, "?"

        task = meta.get("task", "???")
        status = "OK" if ok else "FAIL"
        marker = " " if ok else "X"

        file_results[fp.name] = {
            "ok": ok, "meta": meta, "min_seq": min_seq,
            "diff": diff, "errors": errs, "warnings": warns
        }

        if ok:
            n_ok += 1
        else:
            n_fail += 1

        print(f"  [{marker}] {fp.name:<27s} {task:<15s} {min_seq:>6d} {diff:<12s} {status}")
        for e in errs:
            print(f"      ERROR: {e}")
        for w in warns:
            print(f"      WARN:  {w}")

    print()

    # Validate controls.json if provided
    controls_fail = False
    if args.controls:
        controls_path = Path(args.controls)
        c_warns, c_errs = validate_controls(controls_path, file_results, args.seq_len)

        if c_errs or c_warns:
            print(f"  CONTROLS CHECK ({controls_path.name}):")
            for e in c_errs:
                print(f"      ERROR: {e}")
                controls_fail = True
            for w in c_warns:
                print(f"      WARN:  {w}")
            print()

    # Summary
    print(f"{'='*70}")
    total = n_ok + n_fail
    if n_fail == 0 and not controls_fail:
        print(f"  ALL CLEAR: {n_ok}/{total} files valid at seq_len={args.seq_len}")
    else:
        print(f"  RESULT: {n_ok}/{total} files valid, {n_fail} impossible at seq_len={args.seq_len}")
        if controls_fail:
            print(f"  WARNING: Active data_weights include impossible tasks!")

    # Recommendation
    solvable = [
        (r["min_seq"], name)
        for name, r in file_results.items()
        if r["ok"]
    ]
    solvable.sort()
    if solvable:
        print()
        print(f"  RECOMMENDED training order (easiest first):")
        for min_s, name in solvable:
            r = file_results[name]
            print(f"    {r['diff']:<12s} {name:<30s} (min_seq={min_s})")

    print(f"{'='*70}")

    if args.strict and (n_fail > 0 or controls_fail):
        sys.exit(1)


if __name__ == "__main__":
    main()
