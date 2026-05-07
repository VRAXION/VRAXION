#!/usr/bin/env python3
"""Validate AnchorWeave JSON or JSONL cells."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA_VERSION = "AnchorWeave-v1.0"

REQUIRED_TOP_LEVEL = [
    "schema_version",
    "cell_id",
    "revision",
    "created_at",
    "provenance",
    "episode",
    "relational_graph",
    "salience",
    "actions",
    "outcomes",
    "counterfactuals",
    "memory_hooks",
    "abstraction",
    "symbol_binding",
    "human_annotation",
    "outcome_followup",
]

GROUNDING_JUDGMENTS = {
    "valid_anchor",
    "weak_anchor",
    "invalid_anchor",
    "ambiguous_anchor",
}

ABSTRACTION_LEVELS = {
    "raw_episode",
    "episodic_affordance",
    "causal_relation",
    "outcome_pattern",
    "cross_domain_abstraction",
    "symbol_binding",
}


class AnchorWeaveError(Exception):
    """Raised for invalid input files or command usage."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_schema_path() -> Path:
    return repo_root() / "data" / "anchorweave" / "schemas" / "anchor_cell_v1.schema.json"


def load_schema(schema_path: Path | None = None) -> dict[str, Any]:
    path = schema_path or default_schema_path()
    try:
        with path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
    except FileNotFoundError as exc:
        raise AnchorWeaveError(f"Schema not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnchorWeaveError(f"Schema is invalid JSON: {path}:{exc.lineno}: {exc.msg}") from exc
    if not isinstance(schema, dict):
        raise AnchorWeaveError(f"Schema must be a JSON object: {path}")
    return schema


def read_json_documents(path: Path) -> list[tuple[str, Any]]:
    """Read a single JSON cell, a JSON array, or a JSONL database."""
    if not path.exists():
        raise AnchorWeaveError(f"Database path does not exist: {path}")

    documents: list[tuple[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    documents.append((f"{path}:{line_no}", json.loads(stripped)))
                except json.JSONDecodeError as exc:
                    raise AnchorWeaveError(
                        f"{path}:{line_no}: invalid JSON: {exc.msg}"
                    ) from exc
        return documents

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise AnchorWeaveError(f"{path}:{exc.lineno}: invalid JSON: {exc.msg}") from exc

    if isinstance(payload, list):
        for index, item in enumerate(payload, start=1):
            documents.append((f"{path}[{index}]", item))
    else:
        documents.append((str(path), payload))

    return documents


def _jsonschema_available() -> bool:
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


def _jsonschema_validate(cell: Any, schema: dict[str, Any]) -> list[str] | None:
    try:
        import jsonschema
    except ImportError:
        return None

    try:
        validator_cls = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())
        errors = sorted(validator.iter_errors(cell), key=lambda error: list(error.path))
    except Exception as exc:  # jsonschema uses several exception classes.
        return [f"jsonschema validator failure: {exc}"]

    formatted_errors: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.path) or "<root>"
        formatted_errors.append(f"{location}: {error.message}")
    return formatted_errors


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def minimal_validate_cell(cell: Any) -> list[str]:
    """Fallback structural checks used when jsonschema is unavailable."""
    errors: list[str] = []
    if not isinstance(cell, dict):
        return ["cell must be a JSON object"]

    for field in REQUIRED_TOP_LEVEL:
        if field not in cell:
            errors.append(f"missing required field: {field}")

    if cell.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    cell_id = cell.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        errors.append("cell_id must be a non-empty string")

    revision = cell.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        errors.append("revision must be an integer >= 1")

    if not isinstance(cell.get("created_at"), str) or not cell.get("created_at"):
        errors.append("created_at must be a non-empty string")

    expected_objects = [
        "provenance",
        "episode",
        "relational_graph",
        "salience",
        "actions",
        "outcomes",
        "abstraction",
        "symbol_binding",
        "human_annotation",
        "outcome_followup",
    ]
    for field in expected_objects:
        if field in cell and not isinstance(cell[field], dict):
            errors.append(f"{field} must be an object")

    expected_arrays = ["counterfactuals", "memory_hooks"]
    for field in expected_arrays:
        if field in cell and not isinstance(cell[field], list):
            errors.append(f"{field} must be an array")

    provenance = cell.get("provenance")
    if isinstance(provenance, dict):
        for field in [
            "source",
            "author",
            "privacy_tier",
            "collection_method",
            "contains_private_data",
        ]:
            if field not in provenance:
                errors.append(f"provenance missing required field: {field}")
        if "contains_private_data" in provenance and not isinstance(
            provenance["contains_private_data"], bool
        ):
            errors.append("provenance.contains_private_data must be a boolean")

    episode = cell.get("episode")
    if isinstance(episode, dict):
        for field in ["domain", "title", "scene", "temporal_context", "entities", "cues"]:
            if field not in episode:
                errors.append(f"episode missing required field: {field}")
        if "domain" in episode and not isinstance(episode["domain"], str):
            errors.append("episode.domain must be a string")
        for field in ["entities", "cues"]:
            if field in episode and not isinstance(episode[field], list):
                errors.append(f"episode.{field} must be an array")

    relational_graph = cell.get("relational_graph")
    if isinstance(relational_graph, dict):
        for field in ["nodes", "edges"]:
            if field not in relational_graph:
                errors.append(f"relational_graph missing required field: {field}")
            elif not isinstance(relational_graph[field], list):
                errors.append(f"relational_graph.{field} must be an array")

    salience = cell.get("salience")
    if isinstance(salience, dict):
        for field in ["high", "medium", "low"]:
            if field not in salience:
                errors.append(f"salience missing required field: {field}")
            elif not isinstance(salience[field], list):
                errors.append(f"salience.{field} must be an array")

    actions = cell.get("actions")
    if isinstance(actions, dict):
        for field in ["available", "taken"]:
            if field not in actions:
                errors.append(f"actions missing required field: {field}")
            elif not isinstance(actions[field], list):
                errors.append(f"actions.{field} must be an array")

    outcomes = cell.get("outcomes")
    if isinstance(outcomes, dict):
        for field in ["predicted", "actual"]:
            if field not in outcomes:
                errors.append(f"outcomes missing required field: {field}")
            elif not isinstance(outcomes[field], list):
                errors.append(f"outcomes.{field} must be an array")

    abstraction = cell.get("abstraction")
    if isinstance(abstraction, dict):
        for field in ["level", "pre_symbol_statement", "claim_boundary", "not_this"]:
            if field not in abstraction:
                errors.append(f"abstraction missing required field: {field}")
        if "level" in abstraction and abstraction["level"] not in ABSTRACTION_LEVELS:
            errors.append("abstraction.level is not in taxonomy_v1 abstraction_levels")
        if "not_this" in abstraction and not isinstance(abstraction["not_this"], list):
            errors.append("abstraction.not_this must be an array")

    symbol_binding = cell.get("symbol_binding")
    if isinstance(symbol_binding, dict):
        for field in ["attach", "reject"]:
            if field not in symbol_binding:
                errors.append(f"symbol_binding missing required field: {field}")
            elif not isinstance(symbol_binding[field], list):
                errors.append(f"symbol_binding.{field} must be an array")

    human_annotation = cell.get("human_annotation")
    if isinstance(human_annotation, dict):
        for field in [
            "grounding_judgment",
            "confidence",
            "best_summary",
            "error_tags",
            "positive_tags",
            "claim_boundary",
        ]:
            if field not in human_annotation:
                errors.append(f"human_annotation missing required field: {field}")
        judgment = human_annotation.get("grounding_judgment")
        if judgment is not None and judgment not in GROUNDING_JUDGMENTS:
            errors.append("human_annotation.grounding_judgment is not in taxonomy_v1")
        confidence = human_annotation.get("confidence")
        if confidence is not None and (
            not _is_number(confidence) or confidence < 0 or confidence > 1
        ):
            errors.append("human_annotation.confidence must be a number in [0, 1]")
        for field in ["error_tags", "positive_tags"]:
            if field in human_annotation and not isinstance(human_annotation[field], list):
                errors.append(f"human_annotation.{field} must be an array")

    outcome_followup = cell.get("outcome_followup")
    if isinstance(outcome_followup, dict):
        for field in ["status", "required_followup", "observations"]:
            if field not in outcome_followup:
                errors.append(f"outcome_followup missing required field: {field}")
        if "required_followup" in outcome_followup and not isinstance(
            outcome_followup["required_followup"], bool
        ):
            errors.append("outcome_followup.required_followup must be a boolean")
        if "observations" in outcome_followup and not isinstance(
            outcome_followup["observations"], list
        ):
            errors.append("outcome_followup.observations must be an array")

    return errors


def validate_cell(cell: Any, schema: dict[str, Any] | None = None) -> tuple[list[str], str]:
    if schema is not None:
        schema_errors = _jsonschema_validate(cell, schema)
        if schema_errors is not None:
            return schema_errors, "jsonschema"
    return minimal_validate_cell(cell), "minimal-structural"


def duplicate_ids(documents: list[tuple[str, Any]]) -> Counter[str]:
    seen: set[str] = set()
    duplicates: Counter[str] = Counter()
    for _, cell in documents:
        if not isinstance(cell, dict):
            continue
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            continue
        if cell_id in seen:
            duplicates[cell_id] += 1
        else:
            seen.add(cell_id)
    return duplicates


def print_summary(
    documents: list[tuple[str, Any]],
    invalid: list[tuple[str, list[str]]],
    duplicates: Counter[str],
    mode: str,
) -> None:
    domains: Counter[str] = Counter()
    judgments: Counter[str] = Counter()

    for _, cell in documents:
        if not isinstance(cell, dict):
            continue
        episode = cell.get("episode")
        if isinstance(episode, dict) and isinstance(episode.get("domain"), str):
            domains[episode["domain"]] += 1
        human_annotation = cell.get("human_annotation")
        if isinstance(human_annotation, dict) and isinstance(
            human_annotation.get("grounding_judgment"), str
        ):
            judgments[human_annotation["grounding_judgment"]] += 1

    total = len(documents)
    invalid_count = len(invalid)
    valid_count = total - invalid_count

    print("AnchorWeave validation summary")
    if mode == "jsonschema":
        print("schema validation: jsonschema")
    elif _jsonschema_available():
        print("schema validation: minimal structural fallback")
    else:
        print("schema validation: skipped (jsonschema not installed); using minimal structural validation")
    print(f"total cells: {total}")
    print(f"valid cells: {valid_count}")
    print(f"invalid cells: {invalid_count}")
    print(f"duplicate ids: {sum(duplicates.values())}")

    print("domains:")
    if domains:
        for domain, count in sorted(domains.items()):
            print(f"  {domain}: {count}")
    else:
        print("  <none>")

    print("grounding_judgment counts:")
    if judgments:
        for judgment, count in sorted(judgments.items()):
            print(f"  {judgment}: {count}")
    else:
        print("  <none>")

    if duplicates:
        print("duplicate cell_id values:")
        for cell_id, count in sorted(duplicates.items()):
            print(f"  {cell_id}: {count + 1} occurrences")

    if invalid:
        print("invalid records:")
        for label, errors in invalid:
            print(f"  {label}:")
            for error in errors:
                print(f"    - {error}")


def validate_documents(
    documents: list[tuple[str, Any]], schema: dict[str, Any] | None
) -> tuple[list[tuple[str, list[str]]], str]:
    invalid: list[tuple[str, list[str]]] = []
    mode = "jsonschema" if _jsonschema_available() and schema is not None else "minimal-structural"

    for label, cell in documents:
        errors, mode = validate_cell(cell, schema)
        if errors:
            invalid.append((label, errors))

    return invalid, mode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AnchorWeave JSON/JSONL cells.")
    parser.add_argument("--db", required=True, type=Path, help="Path to JSONL database or single JSON cell.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Optional path to anchor_cell_v1.schema.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        schema = load_schema(args.schema)
        documents = read_json_documents(args.db)
        invalid, mode = validate_documents(documents, schema)
        duplicates = duplicate_ids(documents)
        print_summary(documents, invalid, duplicates, mode)
    except AnchorWeaveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 1 if invalid or duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
