#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pocket_library


DECISION_PASS = "e37_pocket_abi_v1_registry_lock_confirmed"
DECISION_FAIL = "e37_pocket_abi_v1_registry_lock_failed"


def expect_fail(label: str, fn, failures: list[str]) -> None:
    try:
        fn()
    except Exception:
        return
    failures.append(f"{label}: expected failure but call passed")


def run_checks() -> dict[str, object]:
    failures: list[str] = []
    registry_result = pocket_library.validate_registry()
    failures.extend([f"registry: {item}" for item in registry_result["failures"]])
    registry = pocket_library.load_registry()

    stable_ids = pocket_library.stable_pocket_ids(registry)
    if stable_ids != ["protocol_framing_ingress_v001"]:
        failures.append(f"unexpected stable pocket ids: {stable_ids}")

    for pocket_id in ["dirty_start_only_decoder", "dormant_unused_pocket", "protocol_framing_no_adapter", "wrong_rotated_codebook_pocket"]:
        expect_fail(
            f"{pocket_id} unsafe load",
            lambda pocket_id=pocket_id: pocket_library.load_pocket_entry(pocket_id, registry=registry, require_load_allowed=True),
            failures,
        )

    expect_fail(
        "banned pocket frozen params load",
        lambda: pocket_library.load_frozen_params("wrong_rotated_codebook_pocket", registry=registry),
        failures,
    )
    expect_fail(
        "adapter-required target import without adapter",
        lambda: pocket_library.load_for_target("protocol_framing_ingress_v001", adapter_declared=False, registry=registry),
        failures,
    )
    try:
        imported = pocket_library.load_for_target("protocol_framing_ingress_v001", adapter_declared=True, registry=registry)
        if "feature_bias" not in imported["frozen_params"]:
            failures.append("adapter-declared import returned malformed params")
    except Exception as exc:
        failures.append(f"adapter-declared import failed: {exc}")

    mutated_registry = copy.deepcopy(registry)
    mutated_registry["pockets"]["protocol_framing_ingress_v001"]["abi_version"] = "PocketABI-v999"
    expect_fail(
        "unknown ABI major version",
        lambda: pocket_library.load_pocket_entry("protocol_framing_ingress_v001", registry=mutated_registry, require_load_allowed=True),
        failures,
    )
    expect_fail(
        "stable anchor overwrite guard",
        lambda: pocket_library.stage_candidate_guard(
            "protocol_framing_ingress_v001",
            "docs/research/pocket_archive/e35_transfer_smoke/binary_ingress/protocol_framing_ingress_v001",
            registry=registry,
        ),
        failures,
    )

    result = {
        "decision": DECISION_PASS if not failures else DECISION_FAIL,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "registry_result": registry_result,
        "stable_pocket_ids": stable_ids,
        "checker_failure_count": len(failures),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--summary-path", default="target/pilot_wave/e37_pocket_abi_v1_registry_check.json")
    args = parser.parse_args()
    result = run_checks()
    if args.write_summary:
        pocket_library.write_json(Path(args.summary_path), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
