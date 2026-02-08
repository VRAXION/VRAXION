#!/usr/bin/env python3
"""Finalize hashmap A/B summary from existing arm report files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_fields(report: Dict[str, Any]) -> tuple[float, int]:
    eval_obj = report.get("eval") if isinstance(report.get("eval"), dict) else {}
    acc = float(eval_obj.get("eval_acc", 0.0))
    n = int(eval_obj.get("eval_n", 0))
    return acc, n


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build ab_summary.json from existing A/B arm reports.")
    ap.add_argument("--run-root", required=True, help="Root containing arm_a_equal/ and arm_b_hashmap_capacity/")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.run_root).resolve()
    rep_a_path = root / "arm_a_equal" / "train" / "report.json"
    rep_b_path = root / "arm_b_hashmap_capacity" / "train" / "report.json"
    rep_a = _load_json(rep_a_path)
    rep_b = _load_json(rep_b_path)
    acc_a, n_a = _eval_fields(rep_a)
    acc_b, n_b = _eval_fields(rep_b)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_root": str(root),
        "arms": [
            {"arm": "arm_a_equal", "report_path": str(rep_a_path), "report": rep_a},
            {"arm": "arm_b_hashmap_capacity", "report_path": str(rep_b_path), "report": rep_b},
        ],
        "comparison": {
            "eval_acc_a_equal": float(acc_a),
            "eval_acc_b_hashmap_capacity": float(acc_b),
            "eval_n_a": int(n_a),
            "eval_n_b": int(n_b),
            "eval_acc_delta_b_minus_a": float(acc_b - acc_a),
        },
    }
    out_path = root / "ab_summary.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

