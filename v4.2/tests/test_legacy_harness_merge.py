"""Static guard for active harness and CPU legacy-controller consolidation."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive" / "legacy_harness"
CPU_ARCHIVE = ROOT / "archive" / "legacy_cpu_controller"
ACTIVE = [
    ROOT / "mutation_policy_ab.py",
    ROOT / "permutation_curve_sweeps.py",
]
ACTIVE_CPU_SWEEPS = [
    ROOT / "charge_rate_sweep.py",
    ROOT / "dynamic_threshold_sweep.py",
    ROOT / "leak_discrete_sweep.py",
]
CANON_CPU_SUPPORT = [
    ROOT / "graph_baseline_loader.py",
    ROOT / "test_model.py",
]
SHARED = [
    ROOT / "harness" / "__init__.py",
    ROOT / "harness" / "permutation_harness.py",
    ROOT / "harness" / "policy_adapters.py",
    ROOT / "harness" / "cpu_parameter_sweeps.py",
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
ARCHIVED_CPU_CONTROLLER_BASENAMES = [
    "ab_random_source.py",
    "ab_rng_knee.py",
    "benchmark_expressiveness.py",
    "computed_leak_test.py",
    "conn_budget_plateau.py",
    "cr_autotune_test.py",
    "density_plateau_v128.py",
    "drive_ab_test.py",
    "final_three_sweep.py",
    "fix_gain_sweep.py",
    "frozen_mutation_test.py",
    "health_score_test.py",
    "int8_matmul_test.py",
    "leak_static_vs_learn.py",
    "learnable_leak_gain.py",
    "mega_sweep.py",
    "nv_capacity_test.py",
    "nv_ratio_sweep.py",
    "predictive_eval_sweep.py",
    "pruner_ab_test.py",
    "remaining_params_sweep.py",
    "self_conn_sweep.py",
    "sparse_scaling_benchmark.py",
    "v64sparse_4zone_test.py",
    "weight_extend_sweep.py",
    "weight_fine_sweep.py",
    "weight_sweep.py",
    "weight_with_learnable_leak.py",
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
    for path in ACTIVE_CPU_SWEEPS:
        if not path.exists():
            errors.append(f"Missing expected active CPU sweep: {path.name}")

    for name in ARCHIVED_BASENAMES:
        active_path = ROOT / name
        archived_path = ARCHIVE / name
        if active_path.exists():
            errors.append(f"Legacy harness still active at root: {name}")
        if not archived_path.exists():
            errors.append(f"Archived harness missing: {name}")
    for name in ARCHIVED_CPU_CONTROLLER_BASENAMES:
        active_path = ROOT / name
        archived_path = CPU_ARCHIVE / name
        if active_path.exists():
            errors.append(f"Legacy CPU controller test still active at root: {name}")
        if not archived_path.exists():
            errors.append(f"Archived CPU controller test missing: {name}")

    for path in ACTIVE:
        text = path.read_text(encoding="utf-8")
        if "tests.harness" not in text:
            errors.append(f"{path.name} does not import the shared harness package")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                errors.append(f"{path.name} still contains forbidden legacy mutation surface: {forbidden}")

    for path in ACTIVE_CPU_SWEEPS:
        text = path.read_text(encoding="utf-8")
        if "tests.harness" not in text:
            errors.append(f"{path.name} does not import the shared harness package")
        if "def eval_b" in text:
            errors.append(f"{path.name} still defines a local eval_b loop")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                errors.append(f"{path.name} still contains forbidden legacy mutation surface: {forbidden}")

    for path in CANON_CPU_SUPPORT:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                errors.append(f"{path.name} still contains forbidden legacy mutation surface: {forbidden}")
                break

    for path in ROOT.glob("*.py"):
        if path.name in {"test_legacy_harness_merge.py", "test_model.py", "graph_baseline_loader.py"}:
            continue
        if path.name.startswith("gpu_"):
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                errors.append(f"{path.name} still contains forbidden active CPU legacy surface: {forbidden}")
                break

    if errors:
        for err in errors:
            print(f"[X] {err}")
        return 1

    print("[+] Legacy harness merge guard passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
