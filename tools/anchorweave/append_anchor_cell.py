#!/usr/bin/env python3
"""Append exactly one validated AnchorWeave cell to a JSONL database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from validate_anchor_cells import (
    AnchorWeaveError,
    duplicate_ids,
    load_schema,
    read_json_documents,
    validate_cell,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one AnchorWeave cell to JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Path to one JSON AnchorCell.")
    parser.add_argument("--db", required=True, type=Path, help="Target append-only JSONL database.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Optional path to anchor_cell_v1.schema.json.",
    )
    return parser.parse_args(argv)


def load_single_cell(path: Path) -> dict:
    documents = read_json_documents(path)
    if len(documents) != 1:
        raise AnchorWeaveError(f"--input must contain exactly one cell, found {len(documents)}")
    _, cell = documents[0]
    if not isinstance(cell, dict):
        raise AnchorWeaveError("--input cell must be a JSON object")
    return cell


def existing_cell_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    documents = read_json_documents(path)
    duplicates = duplicate_ids(documents)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise AnchorWeaveError(f"target database already contains duplicate cell_id values: {duplicate_list}")

    ids: set[str] = set()
    for label, cell in documents:
        if not isinstance(cell, dict):
            raise AnchorWeaveError(f"target database contains a non-object record: {label}")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise AnchorWeaveError(f"target database contains a record without a valid cell_id: {label}")
        ids.add(cell_id)
    return ids


def append_cell(cell: dict, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with db_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(cell, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        schema = load_schema(args.schema)
        cell = load_single_cell(args.input)
        errors, mode = validate_cell(cell, schema)
        if errors:
            print(f"error: input cell failed validation ({mode})", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

        cell_id = cell.get("cell_id")
        ids = existing_cell_ids(args.db)
        if cell_id in ids:
            raise AnchorWeaveError(f"refusing duplicate cell_id: {cell_id}")

        append_cell(cell, args.db)
        print(f"appended cell_id: {cell_id}")
        print(f"validation mode: {mode}")
    except AnchorWeaveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
