"""Static guard for the active legacy-harness consolidation pass."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive" / "legacy_harness"
ACTIVE = [
    ROOT / "mutation_policy_ab.py",
    ROOT / "permutation_curve_sweeps.py",
]
SHARED = [
    ROOT / "harness" / "__init__.py",
    ROOT / "harness" / "permutation_harness.py",
    ROOT / "harness" / "policy_adapters.py",
]
ARCHIVED_BASENAMES = [
    "addrem_ab_test.py",
    "bool_mood_ab.py",
    "mode_ab_test.py",
    "strategy_10seed_ab.py",
    "window_strategy_ab.py",
    "benchmark_convergence.py",
    "budget_scaling_ab.py",
    "patience_sweep.py",
    "patience_fine_sweep.py",
    "patience_8_9.py",
]
FORBIDDEN = [
    "net.signal",
    "net.grow",
    "net.intensity",
    "net.mood",
    "net.mood_x",
    "net.mood_z",
    "mutate_with_mood",
]


def main():
    errors = []

    for path in SHARED + ACTIVE:
        if not path.exists():
            errors.append(f"Missing expected active harness file: {path.name}")

    for name in ARCHIVED_BASENAMES:
        active_path = ROOT / name
        archived_path = ARCHIVE / name
        if active_path.exists():
            errors.append(f"Legacy harness still active at root: {name}")
        if not archived_path.exists():
            errors.append(f"Archived harness missing: {name}")

    for path in ACTIVE:
        text = path.read_text(encoding="utf-8")
        if "tests.harness" not in text:
            errors.append(f"{path.name} does not import the shared harness package")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                errors.append(f"{path.name} still contains forbidden legacy mutation surface: {forbidden}")

    if errors:
        for err in errors:
            print(f"[X] {err}")
        return 1

    print("[+] Legacy harness merge guard passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

