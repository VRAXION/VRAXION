#!/usr/bin/env python3
"""Export minimal derived AnchorWeave views from canonical JSON/JSONL cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from validate_anchor_cells import AnchorWeaveError, read_json_documents


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export derived AnchorWeave views.")
    parser.add_argument("--db", required=True, type=Path, help="Path to canonical JSONL database or JSON cell.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for derived views.")
    return parser.parse_args(argv)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def as_cells(documents: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for label, cell in documents:
        if not isinstance(cell, dict):
            raise AnchorWeaveError(f"record is not an object: {label}")
        cells.append(cell)
    return cells


def episode_prompt(cell: dict[str, Any]) -> str:
    episode = cell.get("episode", {})
    domain = episode.get("domain", "<unknown>") if isinstance(episode, dict) else "<unknown>"
    title = episode.get("title", "") if isinstance(episode, dict) else ""
    scene = episode.get("scene", "") if isinstance(episode, dict) else ""
    return (
        "Given this AnchorWeave episode, identify the anchor, high-salience "
        "relations, and supported/rejected symbol binding.\n\n"
        f"Domain: {domain}\n"
        f"Title: {title}\n"
        f"Episode: {scene}"
    )


def export_sft(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        human = cell.get("human_annotation", {})
        abstraction = cell.get("abstraction", {})
        if not isinstance(human, dict):
            human = {}
        if not isinstance(abstraction, dict):
            abstraction = {}

        claim_boundary = human.get("claim_boundary") or abstraction.get("claim_boundary") or ""
        completion = f"{human.get('best_summary', '')}\nClaim boundary: {claim_boundary}".strip()
        rows.append(
            {
                "cell_id": cell.get("cell_id"),
                "prompt": episode_prompt(cell),
                "completion": completion,
            }
        )
    return rows


def export_dpo(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        candidates = cell.get("candidate_outputs")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            chosen = candidate.get("chosen")
            rejected = candidate.get("rejected")
            if chosen is None or rejected is None:
                continue
            rows.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "prompt": candidate.get("prompt") or episode_prompt(cell),
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
    return rows


def export_reward(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        binding = cell.get("symbol_binding", {})
        if not isinstance(binding, dict):
            continue
        for decision in ["attach", "reject"]:
            items = binding.get(decision, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "score" not in item:
                    continue
                rows.append(
                    {
                        "cell_id": cell.get("cell_id"),
                        "decision": decision,
                        "symbol": item.get("symbol"),
                        "score": item.get("score"),
                        "rationale": item.get("rationale", ""),
                    }
                )
    return rows


def export_graph(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        graph = cell.get("relational_graph", {})
        if not isinstance(graph, dict):
            continue
        edges = graph.get("edges", [])
        if not isinstance(edges, list):
            continue
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            rows.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "relation": edge.get("relation"),
                    "evidence": edge.get("evidence", ""),
                    "salience_weight": edge.get("salience_weight"),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        cells = as_cells(read_json_documents(args.db))

        outputs = {
            "sft/anchorweave_sft.jsonl": export_sft(cells),
            "dpo/anchorweave_dpo.jsonl": export_dpo(cells),
            "reward/anchorweave_reward.jsonl": export_reward(cells),
            "graph/anchorweave_edges.jsonl": export_graph(cells),
        }

        for relative_path, rows in outputs.items():
            write_jsonl(args.out / relative_path, rows)
            print(f"wrote {relative_path}: {len(rows)} records")

        if not outputs["dpo/anchorweave_dpo.jsonl"]:
            print("DPO export skipped records: no candidate_outputs found")
    except AnchorWeaveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
