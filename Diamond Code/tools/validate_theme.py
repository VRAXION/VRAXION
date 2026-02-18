#!/usr/bin/env python3
"""Validate JSONL theme files for Diamond Code paired training.

Schema (line 1 = header, lines 2+ = pairs):
  Header:  {"_meta":true, "theme":"...", "version":1, "encoding":"utf8|hex|base64",
            "gist_ratio":0.10, "active":true}
  Pair:    {"id":"aa-00001", "s":"c|g", "d":1-6, "in":"...", "out":"..."}
  Optional: "set":"group-id"

Checks:
  1. Every line is valid JSON
  2. Line 1 has _meta:true with required fields
  3. All pairs have required fields (id, s, d, in, out)
  4. No duplicate IDs
  5. s is "c" (curriculum) or "g" (gist)
  6. d (difficulty) is integer 1-6
  7. in/out fit within 773 bytes when UTF-8 encoded (num_bits=6184)
  8. Encoding field is valid (utf8, hex, base64)
  9. hex/base64 fields decode cleanly if that encoding is declared
  10. Counts summary matches actual

Usage:
  python tools/validate_theme.py data/themes/arithmetic_add.jsonl
  python tools/validate_theme.py data/themes/  # validate all .jsonl in dir
  python tools/validate_theme.py data/themes/ --max_bytes 773
"""

import json
import sys
import base64
from pathlib import Path

MAX_BYTES_DEFAULT = 773  # num_bits=6184 -> 773 bytes per position
VALID_STATES = {"c", "g"}
VALID_ENCODINGS = {"utf8", "hex", "base64"}
VALID_DIFFICULTY = {1, 2, 3, 4, 5, 6}

REQUIRED_META_FIELDS = {"_meta", "theme", "version", "encoding", "gist_ratio"}
REQUIRED_PAIR_FIELDS = {"id", "s", "d", "in", "out"}


def encode_field(value: str, encoding: str) -> bytes:
    """Convert a field value to bytes using the declared encoding."""
    if encoding == "utf8":
        return value.encode("utf-8")
    elif encoding == "hex":
        return bytes.fromhex(value)
    elif encoding == "base64":
        return base64.b64decode(value)
    else:
        raise ValueError(f"unknown encoding: {encoding}")


def validate_theme_file(filepath: Path, max_bytes: int = MAX_BYTES_DEFAULT) -> dict:
    """Validate a single .jsonl theme file. Returns result dict."""
    errors = []
    warnings = []
    ids_seen = set()
    sets_seen = set()
    counts = {"c": 0, "g": 0}
    difficulty_counts = {i: 0 for i in range(1, 7)}
    meta = None
    n_pairs = 0
    encoding = "utf8"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError) as e:
        return {"ok": False, "errors": [f"cannot read file: {e}"],
                "warnings": [], "meta": None, "counts": counts,
                "n_pairs": 0, "filepath": str(filepath)}

    if len(lines) == 0:
        return {"ok": False, "errors": ["empty file"],
                "warnings": [], "meta": None, "counts": counts,
                "n_pairs": 0, "filepath": str(filepath)}

    # --- Line 1: header ---
    try:
        meta = json.loads(lines[0])
    except json.JSONDecodeError as e:
        errors.append(f"line 1: invalid JSON: {e}")
        return {"ok": False, "errors": errors, "warnings": warnings,
                "meta": None, "counts": counts, "n_pairs": 0,
                "filepath": str(filepath)}

    if not meta.get("_meta"):
        errors.append("line 1: missing _meta:true (not a valid header)")

    for field in REQUIRED_META_FIELDS:
        if field not in meta:
            errors.append(f"header: missing required field '{field}'")

    if "encoding" in meta:
        encoding = meta["encoding"]
        if encoding not in VALID_ENCODINGS:
            errors.append(f"header: invalid encoding '{encoding}' "
                          f"(must be one of {VALID_ENCODINGS})")

    if "gist_ratio" in meta:
        gr = meta["gist_ratio"]
        if not isinstance(gr, (int, float)) or gr < 0 or gr > 1:
            errors.append(f"header: gist_ratio must be 0.0-1.0, got {gr}")

    if "version" in meta:
        if not isinstance(meta["version"], int) or meta["version"] < 1:
            errors.append(f"header: version must be positive integer, got {meta['version']}")

    # --- Lines 2+: pairs ---
    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue  # skip blank lines

        try:
            pair = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {line_num}: invalid JSON: {e}")
            continue

        # Skip if accidentally another meta line
        if pair.get("_meta"):
            errors.append(f"line {line_num}: unexpected _meta line (only line 1 should be header)")
            continue

        n_pairs += 1

        # Required fields
        missing = REQUIRED_PAIR_FIELDS - set(pair.keys())
        if missing:
            errors.append(f"line {line_num}: missing fields: {missing}")
            continue

        # ID uniqueness
        pid = pair["id"]
        if pid in ids_seen:
            errors.append(f"line {line_num}: duplicate id '{pid}'")
        ids_seen.add(pid)

        # State
        state = pair["s"]
        if state not in VALID_STATES:
            errors.append(f"line {line_num} ({pid}): invalid state '{state}' "
                          f"(must be 'c' or 'g')")
        else:
            counts[state] += 1

        # Difficulty
        diff = pair["d"]
        if not isinstance(diff, int) or diff not in VALID_DIFFICULTY:
            errors.append(f"line {line_num} ({pid}): invalid difficulty {diff} "
                          f"(must be integer 1-6)")
        else:
            difficulty_counts[diff] += 1

        # Set tracking
        if "set" in pair:
            sets_seen.add(pair["set"])

        # Content size check
        for field_name in ("in", "out"):
            value = pair[field_name]
            if not isinstance(value, str):
                errors.append(f"line {line_num} ({pid}): '{field_name}' must be string, "
                              f"got {type(value).__name__}")
                continue
            try:
                encoded = encode_field(value, encoding)
                if len(encoded) > max_bytes:
                    errors.append(
                        f"line {line_num} ({pid}): '{field_name}' is {len(encoded)} bytes "
                        f"(max {max_bytes}). Content: {value[:50]}..."
                    )
            except (ValueError, base64.binascii.Error) as e:
                errors.append(
                    f"line {line_num} ({pid}): '{field_name}' cannot encode as "
                    f"{encoding}: {e}"
                )

    # --- Summary warnings ---
    if counts["g"] == 0:
        warnings.append("no gist pairs (s='g') — recommended to have some")

    if n_pairs == 0:
        errors.append("no data pairs found (file has only header)")

    gist_ratio = meta.get("gist_ratio", 0) if meta else 0
    if n_pairs > 0 and counts["g"] > 0:
        actual_ratio = counts["g"] / n_pairs
        if gist_ratio > 0 and abs(actual_ratio - gist_ratio) > 0.05:
            warnings.append(
                f"gist_ratio in header ({gist_ratio:.2f}) doesn't match "
                f"actual ratio ({actual_ratio:.2f} = {counts['g']}/{n_pairs})"
            )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "meta": meta,
        "counts": counts,
        "difficulty_counts": difficulty_counts,
        "n_pairs": n_pairs,
        "n_sets": len(sets_seen),
        "filepath": str(filepath),
    }


def print_result(result: dict):
    """Print validation result for a single file."""
    fp = Path(result["filepath"]).name
    ok = result["ok"]
    meta = result["meta"]
    theme = meta.get("theme", "???") if meta else "???"

    status = "PASS" if ok else "FAIL"
    marker = " " if ok else "X"

    print(f"  [{marker}] {fp:<35s} theme={theme:<20s} {status}")

    if meta:
        c = result["counts"]
        dc = result["difficulty_counts"]
        print(f"      pairs: {result['n_pairs']} "
              f"(curriculum={c['c']}, gist={c['g']}, sets={result['n_sets']})")
        diff_str = " ".join(f"d{k}={v}" for k, v in sorted(dc.items()) if v > 0)
        if diff_str:
            print(f"      difficulty: {diff_str}")

    for e in result["errors"]:
        print(f"      ERROR: {e}")
    for w in result["warnings"]:
        print(f"      WARN:  {w}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate JSONL theme files")
    parser.add_argument("path", type=str,
                        help="Path to .jsonl file or directory of .jsonl files")
    parser.add_argument("--max_bytes", type=int, default=MAX_BYTES_DEFAULT,
                        help=f"Max bytes per in/out field (default: {MAX_BYTES_DEFAULT})")
    parser.add_argument("--strict", action="store_true",
                        help="Exit code 1 if any file fails")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("*.jsonl"))
        if not files:
            print(f"ERROR: no .jsonl files in {target}")
            sys.exit(1)
    else:
        print(f"ERROR: path not found: {target}")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"  THEME VALIDATION")
    print(f"  path:       {target}")
    print(f"  max_bytes:  {args.max_bytes}")
    print(f"  files:      {len(files)}")
    print(f"{'=' * 70}")
    print()

    n_ok = 0
    n_fail = 0
    total_pairs = 0

    for fp in files:
        result = validate_theme_file(fp, args.max_bytes)
        print_result(result)
        print()

        if result["ok"]:
            n_ok += 1
        else:
            n_fail += 1
        total_pairs += result["n_pairs"]

    print(f"{'=' * 70}")
    if n_fail == 0:
        print(f"  ALL CLEAR: {n_ok}/{n_ok + n_fail} files valid, "
              f"{total_pairs} total pairs")
    else:
        print(f"  RESULT: {n_ok}/{n_ok + n_fail} valid, {n_fail} failed, "
              f"{total_pairs} total pairs")
    print(f"{'=' * 70}")

    if args.strict and n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
