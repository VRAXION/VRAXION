#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pocket_library


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--summary-path", default="target/pilot_wave/pocket_library_registry_check.json")
    args = parser.parse_args()
    result = pocket_library.validate_registry()
    if args.write_summary:
        pocket_library.write_json(Path(args.summary_path), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
