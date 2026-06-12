#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {"candidate", "staging", "stable", "core", "deprecated", "banned"}
LOAD_ALLOWED_STATUSES = {"stable", "core"}
ACTIVE_ABI_VERSION = "PocketABI-v1"
REQUIRED_STABLE_FILES = [
    "pocket_manifest.json",
    "pocket_contract.md",
    "frozen_params.json",
    "lineage.json",
    "source_metrics.json",
    "transfer_tests.json",
    "safety_report.json",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = repo_root()
REGISTRY_PATH = REPO_ROOT / "docs" / "research" / "pocket_library" / "registry.json"
ARCHIVE_ROOT = REPO_ROOT / "docs" / "research" / "pocket_archive"
ECOLOGY_ROOT = REPO_ROOT / "docs" / "research" / "pocket_ecology"
TRAINING_LOCK_PATH = REPO_ROOT / "docs" / "research" / "pocket_library" / "training_lock_v1.json"


def json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if not str(path).startswith(str(REPO_ROOT.resolve())):
        raise ValueError(f"path escapes repository root: {value}")
    return path


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return read_json(path)


def save_registry(registry: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    write_json(path, registry)


def pocket_entries(registry: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    registry = registry or load_registry()
    entries = registry.get("pockets", {})
    if not isinstance(entries, dict):
        raise ValueError("registry.pockets must be an object")
    return entries


def stable_pocket_ids(registry: dict[str, Any] | None = None, pocket_type: str | None = None) -> list[str]:
    out: list[str] = []
    for pocket_id, entry in pocket_entries(registry).items():
        if entry.get("status") not in LOAD_ALLOWED_STATUSES:
            continue
        if entry.get("load_allowed") is not True:
            continue
        if pocket_type is not None and entry.get("pocket_type") != pocket_type:
            continue
        out.append(pocket_id)
    return sorted(out)


def load_pocket_entry(pocket_id: str, registry: dict[str, Any] | None = None, require_load_allowed: bool = True) -> dict[str, Any]:
    entries = pocket_entries(registry)
    if pocket_id not in entries:
        raise KeyError(f"pocket not found in registry: {pocket_id}")
    entry = entries[pocket_id]
    if require_load_allowed:
        if entry.get("status") not in LOAD_ALLOWED_STATUSES or entry.get("load_allowed") is not True:
            raise ValueError(f"pocket is not load-allowed: {pocket_id} status={entry.get('status')}")
    return entry


def load_frozen_params(pocket_id: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = load_pocket_entry(pocket_id, registry=registry, require_load_allowed=True)
    params_path = resolve_repo_path(str(entry["frozen_params_path"]))
    params = read_json(params_path)
    expected = entry.get("frozen_params_digest")
    observed = json_digest(params)
    if expected and expected != observed:
        raise ValueError(f"frozen param digest mismatch for {pocket_id}: {observed} != {expected}")
    return params


def stage_candidate_guard(pocket_id: str, archive_dir: str, registry: dict[str, Any] | None = None) -> None:
    entries = pocket_entries(registry)
    target = resolve_repo_path(archive_dir)
    if not str(target).startswith(str(ARCHIVE_ROOT.resolve())):
        raise ValueError(f"candidate archive dir must live under {rel(ARCHIVE_ROOT)}")
    if pocket_id in entries and entries[pocket_id].get("status") in LOAD_ALLOWED_STATUSES:
        raise ValueError(f"refusing to overwrite stable/core pocket anchor: {pocket_id}")
    manifest = target / "pocket_manifest.json"
    frozen = target / "frozen_params.json"
    if manifest.exists() or frozen.exists():
        raise ValueError(f"candidate archive dir already contains anchor files: {rel(target)}")


def validate_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    failures: list[str] = []
    if not path.exists():
        return {"passed": False, "failure_count": 1, "failures": [f"missing registry {path}"]}
    try:
        registry = load_registry(path)
    except Exception as exc:
        return {"passed": False, "failure_count": 1, "failures": [f"registry parse failed: {exc}"]}

    roots = registry.get("canonical_roots", {})
    if roots.get("archive_root") != "docs/research/pocket_archive":
        failures.append("canonical archive root mismatch")
    if roots.get("ecology_root") != "docs/research/pocket_ecology":
        failures.append("canonical ecology root mismatch")
    lock = registry.get("training_lock", {})
    if lock.get("status") != "active" or lock.get("abi_version") != ACTIVE_ABI_VERSION:
        failures.append("active PocketABI-v1 training lock missing")
    if lock.get("lock_json") != "docs/research/pocket_library/training_lock_v1.json":
        failures.append("training lock json path mismatch")
    if not TRAINING_LOCK_PATH.exists():
        failures.append("missing training_lock_v1.json")
    else:
        training_lock = read_json(TRAINING_LOCK_PATH)
        if training_lock.get("active") is not True:
            failures.append("training lock is not active")
        if training_lock.get("abi_version") != ACTIVE_ABI_VERSION:
            failures.append("training lock ABI mismatch")
        if training_lock.get("stable_anchor_overwrite_allowed") is not False:
            failures.append("training lock permits stable anchor overwrite")

    entries = registry.get("pockets", {})
    if not isinstance(entries, dict) or not entries:
        failures.append("registry has no pocket entries")
        entries = {}

    stable_ids: list[str] = []
    for pocket_id, entry in entries.items():
        status = entry.get("status")
        if status not in ALLOWED_STATUSES:
            failures.append(f"{pocket_id}: invalid status {status}")
        if entry.get("load_allowed") is True and status not in LOAD_ALLOWED_STATUSES:
            failures.append(f"{pocket_id}: load_allowed true while status={status}")
        if status in LOAD_ALLOWED_STATUSES:
            stable_ids.append(pocket_id)
            for key in [
                "abi_version",
                "archive_dir",
                "manifest_path",
                "frozen_params_path",
                "frozen_params_digest",
                "input_contract",
                "output_contract",
                "allowed_side_effects",
                "compatible_families",
                "known_bottlenecks",
                "requires_adapter",
            ]:
                if not entry.get(key):
                    failures.append(f"{pocket_id}: missing {key}")
            if entry.get("abi_version") != ACTIVE_ABI_VERSION:
                failures.append(f"{pocket_id}: ABI version mismatch")
            archive_dir = resolve_repo_path(str(entry.get("archive_dir", "")))
            if not str(archive_dir).startswith(str(ARCHIVE_ROOT.resolve())):
                failures.append(f"{pocket_id}: archive dir is outside canonical archive root")
            for name in REQUIRED_STABLE_FILES:
                if not (archive_dir / name).exists():
                    failures.append(f"{pocket_id}: missing archive file {name}")
            manifest_path = resolve_repo_path(str(entry.get("manifest_path", "")))
            frozen_path = resolve_repo_path(str(entry.get("frozen_params_path", "")))
            if manifest_path.exists() and frozen_path.exists():
                manifest = read_json(manifest_path)
                params = read_json(frozen_path)
                if manifest.get("frozen_params_sha256") != entry.get("frozen_params_digest"):
                    failures.append(f"{pocket_id}: registry digest differs from pocket manifest")
                if json_digest(params) != entry.get("frozen_params_digest"):
                    failures.append(f"{pocket_id}: frozen params digest mismatch")
                if manifest.get("version") != entry.get("version"):
                    failures.append(f"{pocket_id}: version mismatch")
    if not stable_ids:
        failures.append("no stable/core loadable pockets")

    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "stable_pocket_ids": sorted(stable_ids),
        "registry_path": rel(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-stable", action="store_true")
    parser.add_argument("--load-pocket")
    parser.add_argument("--write-summary")
    args = parser.parse_args()

    if args.check:
        result = validate_registry()
        if args.write_summary:
            write_json(Path(args.write_summary), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.list_stable:
        print(json.dumps({"stable_pocket_ids": stable_pocket_ids()}, indent=2, sort_keys=True))
        return 0
    if args.load_pocket:
        params = load_frozen_params(args.load_pocket)
        print(json.dumps({"pocket_id": args.load_pocket, "param_keys": sorted(params), "param_digest": json_digest(params)}, indent=2, sort_keys=True))
        return 0
    raise SystemExit("use --check, --list-stable, or --load-pocket")


if __name__ == "__main__":
    raise SystemExit(main())
