"""Multi-seed H sweep and Phase B microprobe driver.

Default mode preserves the original H-dimensionality sweep. `--phase-b` runs
the preregistered H=384 confound-vs-intrinsic arms for `evolve_mutual_inhibition`
and writes candidate logs, checkpoints, run metadata, and panel summaries.
`--phase-b1` runs the horizon x accept_ties follow-up on the same fixture.
`--phase-d1` runs the acceptance-aperture zero-p policy follow-up.
`--phase-d2` runs the cross-H validation for the D1 activation winner.
`--phase-d3-klock` runs the coarse K-lock sweep for the Search Aperture Function.
`--phase-d3-fine-k` runs the H=256 fine K-lock sweep.
`--phase-d4-softness` runs the locked-K softness sweep.
`--phase-d7-bandit` runs the Safe Operator Bandit over locked SAF v1.
`--phase-d8-instrumentation` runs D8.3 instrumentation-only over locked SAF v1.
`--phase-d8-archive-microprobe` runs D8.4a live archive-parent switching microprobe.
`--phase-d8-p2-microprobe` runs D8.4b live P2_PSI_CONF archive-parent microprobe.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "instnct-core" / "tests" / "fixtures" / "alice_corpus.txt"
DEFAULT_PACKED = REPO_ROOT / "output" / "block_c_bytepair_champion" / "packed.bin"
DEFAULT_D6_OPERATOR_SUMMARY = (
    REPO_ROOT / "output" / "phase_d6_trajectory_field_20260427" / "analysis" / "operator_field_summary.csv"
)

CANONICAL_OPERATORS = [
    ("add_edge", 0.22),
    ("remove_edge", 0.13),
    ("rewire", 0.09),
    ("reverse", 0.13),
    ("mirror", 0.06),
    ("enhance", 0.07),
    ("theta", 0.05),
    ("channel", 0.10),
    ("loop2", 0.05),
    ("loop3", 0.05),
    ("projection_weight", 0.05),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixtures", default="mutual_inhibition,bytepair_proj",
                   help="comma-separated list of fixtures to run")
    p.add_argument("--H-values", default="128,192,256,384",
                   help="comma-separated list of H values to sweep")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds per cell")
    p.add_argument("--steps", type=int, default=20000, help="base mutation steps per run")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="corpus text file")
    p.add_argument("--packed", default=str(DEFAULT_PACKED), help="VCBP packed embedding table")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--resume", action="store_true", help="skip cells already in results.json")
    p.add_argument("--dry-run", action="store_true", help="print cells but do not execute")
    p.add_argument("--phase-b", action="store_true",
                   help="run preregistered Phase B arms for evolve_mutual_inhibition")
    p.add_argument("--phase-b1", action="store_true",
                   help="run Phase B.1 horizon x accept_ties arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d1", action="store_true",
                   help="run Phase D1 acceptance-aperture zero-p arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d2", action="store_true",
                   help="run Phase D2 cross-H activation validation arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d3-klock", action="store_true",
                   help="run Phase D3 coarse K-lock strict arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d3-fine-k", action="store_true",
                   help="run Phase D3.1 fine K-lock strict arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d4-softness", action="store_true",
                   help="run Phase D4 locked-K softness arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d7-bandit", action="store_true",
                   help="run Phase D7.1 safe operator-bandit arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d8-instrumentation", action="store_true",
                   help="run Phase D8.3 instrumentation-only locked SAF v1 arms for evolve_mutual_inhibition")
    p.add_argument("--phase-d8-archive-microprobe", action="store_true",
                   help="run Phase D8.4a archive-parent switching microprobe for evolve_mutual_inhibition")
    p.add_argument("--phase-d8-p2-microprobe", action="store_true",
                   help="run Phase D8.4b P2_PSI_CONF archive-parent microprobe for evolve_mutual_inhibition")
    p.add_argument("--arms", default="B0,B1,B2,B3,B4",
                   help="comma-separated Phase B arms to run")
    p.add_argument("--panel-interval", type=int, default=None,
                   help="write Phase B panel_timeseries.csv every N steps")
    p.add_argument("--heartbeat-interval", type=int, default=None,
                   help="write heartbeat.json/log under --out every N seconds for long phase sweeps")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel Phase B cells to run at once")
    return p.parse_args()


def seed_from_idx(idx: int) -> int:
    return 42 + idx * 1000


def parse_summary_line(stdout: str) -> dict | None:
    summaries = []
    for line in stdout.splitlines():
        if line.startswith("SUMMARY "):
            summaries.append(line[len("SUMMARY "):])
    if len(summaries) != 1:
        print(f"  !! expected exactly one SUMMARY line, got {len(summaries)}", file=sys.stderr)
        return None
    try:
        return json.loads(summaries[0])
    except json.JSONDecodeError as e:
        print(f"  !! failed to decode SUMMARY line: {e}", file=sys.stderr)
        print(f"     line: {summaries[0]!r}", file=sys.stderr)
        return None


def example_binary_path(example: str) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    target_dir = Path(os.environ.get("VRAXION_TARGET_DIR", str(REPO_ROOT / "target")))
    if not target_dir.is_absolute():
        target_dir = REPO_ROOT / target_dir
    return target_dir / "release" / "examples" / f"{example}{suffix}"


def build_release_example(example: str) -> int:
    cmd = [
        "cargo", "build", "--release", "--example", example,
        "--manifest-path", str(REPO_ROOT / "instnct-core" / "Cargo.toml"),
    ]
    env = os.environ.copy()
    if "VRAXION_TARGET_DIR" in env:
        target_dir = Path(env["VRAXION_TARGET_DIR"])
        if not target_dir.is_absolute():
            target_dir = REPO_ROOT / target_dir
        env["CARGO_TARGET_DIR"] = str(target_dir)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"  !! cargo build --example {example} exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-4000:], file=sys.stderr)
    return proc.returncode


def prebuild_phase_examples(fixtures: list[str]) -> int:
    examples = [f"evolve_{fixture}" for fixture in fixtures] + ["diag_phase_b_panel"]
    for example in dict.fromkeys(examples):
        rc = build_release_example(example)
        if rc != 0:
            return rc
    return 0


def cargo_example_cmd(example: str, corpus: str, packed: str, *, prebuilt: bool = False) -> list[str]:
    if prebuilt:
        exe = example_binary_path(example)
        if not exe.exists():
            raise FileNotFoundError(f"prebuilt example missing: {exe}")
        return [str(exe), corpus, packed]
    return [
        "cargo", "run", "--release", "--example", example,
        "--manifest-path", str(REPO_ROOT / "instnct-core" / "Cargo.toml"),
        "--", corpus, packed,
    ]


def run_cell(fixture: str, h: int, seed: int, steps: int, corpus: str, packed: str) -> tuple[dict | None, int, float]:
    cmd = cargo_example_cmd(f"evolve_{fixture}", corpus, packed) + [
        "--steps", str(steps),
        "--seed", str(seed),
        "--H", str(h),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"  !! cargo exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return None, proc.returncode, wall
    summary = parse_summary_line(proc.stdout)
    return summary, 0, wall


def phase_b_arm_config(arm: str, base_steps: int) -> dict:
    configs = {
        "B0": {"steps": base_steps, "jackpot": 9, "ticks": None, "input_scatter": False, "accept_ties": None},
        "B1": {"steps": base_steps * 2, "jackpot": 9, "ticks": None, "input_scatter": False, "accept_ties": None},
        "B2": {"steps": base_steps, "jackpot": 18, "ticks": None, "input_scatter": False, "accept_ties": None},
        "B3": {"steps": base_steps, "jackpot": 9, "ticks": 12, "input_scatter": False, "accept_ties": None},
        "B4": {"steps": base_steps, "jackpot": 9, "ticks": None, "input_scatter": True, "accept_ties": None},
    }
    if arm not in configs:
        raise ValueError(f"unknown Phase B arm: {arm}")
    return configs[arm]


def phase_b1_arm_config(arm: str, base_steps: int) -> dict:
    configs = {}
    for label, multiplier in [("S20", 1), ("S40", 2), ("S80", 4)]:
        configs[f"B1_{label}_STRICT"] = {
            "steps": base_steps * multiplier,
            "jackpot": 9,
            "ticks": None,
            "input_scatter": False,
            "accept_ties": False,
        }
        configs[f"B1_{label}_TIES"] = {
            "steps": base_steps * multiplier,
            "jackpot": 9,
            "ticks": None,
            "input_scatter": False,
            "accept_ties": True,
        }
    if arm not in configs:
        raise ValueError(f"unknown Phase B.1 arm: {arm}")
    return configs[arm]


def phase_d1_arm_config(arm: str, base_steps: int) -> dict:
    def cfg(jackpot: int, accept_policy: str, neutral_p: float | None = None) -> dict:
        return {
            "steps": base_steps,
            "jackpot": jackpot,
            "ticks": None,
            "input_scatter": False,
            "accept_ties": False,
            "accept_policy": accept_policy,
            "neutral_p": neutral_p,
            "accept_epsilon": None,
        }

    configs = {}
    for jackpot in [1, 3, 9]:
        configs[f"D1_K{jackpot}_STRICT"] = cfg(jackpot, "strict")
        configs[f"D1_K{jackpot}_ZERO_P03"] = cfg(jackpot, "zero-p", 0.3)
        configs[f"D1_K{jackpot}_ZERO_P10"] = cfg(jackpot, "zero-p", 1.0)

    # Backward-compatible aliases for older Phase D1 probes. New default D1
    # runs use the explicit K-qualified arms above.
    configs.update({
        "D1_STRICT": configs["D1_K9_STRICT"],
        "D1_ZERO_P01": cfg(9, "zero-p", 0.1),
        "D1_ZERO_P03": configs["D1_K9_ZERO_P03"],
        "D1_ZERO_P06": cfg(9, "zero-p", 0.6),
        "D1_ZERO_P10": configs["D1_K9_ZERO_P10"],
        "D1_TIES_LEGACY": {
            "steps": base_steps,
            "jackpot": 9,
            "ticks": None,
            "input_scatter": False,
            "accept_ties": True,
            "accept_policy": None,
            "neutral_p": None,
            "accept_epsilon": None,
        },
    })
    if arm not in configs:
        raise ValueError(f"unknown Phase D1 arm: {arm}")
    return configs[arm]


def phase_d2_arm_config(arm: str, base_steps: int) -> dict:
    configs = {
        "D2_K1_STRICT": phase_d1_arm_config("D1_K1_STRICT", base_steps),
        "D2_K1_ZERO_P10": phase_d1_arm_config("D1_K1_ZERO_P10", base_steps),
        "D2_K3_STRICT": phase_d1_arm_config("D1_K3_STRICT", base_steps),
        "D2_K3_ZERO_P10": phase_d1_arm_config("D1_K3_ZERO_P10", base_steps),
        "D2_K9_STRICT": phase_d1_arm_config("D1_K9_STRICT", base_steps),
        "D2_K9_ZERO_P10": phase_d1_arm_config("D1_K9_ZERO_P10", base_steps),
    }
    if arm not in configs:
        raise ValueError(f"unknown Phase D2 arm: {arm}")
    return configs[arm]


def phase_d3_klock_arm_config(arm: str, base_steps: int) -> dict:
    configs = {
        "D3_K5_STRICT": {
            **phase_d1_arm_config("D1_K1_STRICT", base_steps),
            "jackpot": 5,
        },
        "D3_K13_STRICT": {
            **phase_d1_arm_config("D1_K1_STRICT", base_steps),
            "jackpot": 13,
        },
        "D3_K18_STRICT": {
            **phase_d1_arm_config("D1_K1_STRICT", base_steps),
            "jackpot": 18,
        },
    }
    if arm not in configs:
        raise ValueError(f"unknown Phase D3 K-lock arm: {arm}")
    return configs[arm]


def phase_d3_fine_k_arm_config(arm: str, base_steps: int) -> dict:
    configs = {}
    for jackpot in [15, 18, 21, 24]:
        configs[f"D3F_K{jackpot}_STRICT"] = {
            **phase_d1_arm_config("D1_K1_STRICT", base_steps),
            "jackpot": jackpot,
        }
    if arm not in configs:
        raise ValueError(f"unknown Phase D3 fine-K arm: {arm}")
    return configs[arm]


D4_SOFTNESS_ARMS = [
    "D4_H128_K9_STRICT",
    "D4_H128_K9_ZERO_P03",
    "D4_H128_K9_ZERO_P10",
    "D4_H256_K18_STRICT",
    "D4_H256_K18_ZERO_P03",
    "D4_H256_K18_ZERO_P10",
    "D4_H384_K9_STRICT",
    "D4_H384_K9_ZERO_P03",
    "D4_H384_K9_ZERO_P10",
]


def phase_d4_softness_arm_config(arm: str, base_steps: int) -> dict:
    """Locked-K softness arms. Each arm owns exactly one H/K pair."""
    configs = {}
    for h, jackpot in [(128, 9), (256, 18), (384, 9)]:
        prefix = f"D4_H{h}_K{jackpot}"
        configs[f"{prefix}_STRICT"] = {
            **phase_d1_arm_config("D1_K1_STRICT", base_steps),
            "H": h,
            "jackpot": jackpot,
        }
        configs[f"{prefix}_ZERO_P03"] = {
            **phase_d1_arm_config("D1_K1_ZERO_P03", base_steps),
            "H": h,
            "jackpot": jackpot,
        }
        configs[f"{prefix}_ZERO_P10"] = {
            **phase_d1_arm_config("D1_K1_ZERO_P10", base_steps),
            "H": h,
            "jackpot": jackpot,
        }
    if arm not in configs:
        raise ValueError(f"unknown Phase D4 softness arm: {arm}")
    return configs[arm]


D7_BANDIT_ARMS = [
    "D7_BASELINE",
    "D7_STATIC_PRIOR",
    "D7_PRIOR_EWMA",
]


def phase_d7_bandit_arm_config(arm: str, base_steps: int, h: int | None = None, prior_path: Path | None = None) -> dict:
    if h is None:
        raise ValueError("Phase D7 arm config requires H")
    k_by_h = {128: 9, 256: 18, 384: 9}
    if h not in k_by_h:
        raise ValueError(f"Phase D7 supports only H in {sorted(k_by_h)}, got {h}")
    cfg = {
        **phase_d1_arm_config("D1_K1_STRICT", base_steps),
        "H": h,
        "jackpot": k_by_h[h],
        "operator_policy": "baseline",
        "operator_prior": None,
        "operator_epsilon_random": 0.15,
        "operator_weight_floor": 0.25,
        "operator_weight_cap": 4.0,
        "operator_ewma_alpha": 0.05,
    }
    if arm == "D7_BASELINE":
        return cfg
    if arm == "D7_STATIC_PRIOR":
        cfg.update({"operator_policy": "static-prior", "operator_prior": str(prior_path)})
        return cfg
    if arm == "D7_PRIOR_EWMA":
        cfg.update({"operator_policy": "prior-ewma", "operator_prior": str(prior_path)})
        return cfg
    raise ValueError(f"unknown Phase D7 bandit arm: {arm}")


D8_INSTRUMENTATION_ARMS = [
    "D8_INSTRUMENTED",
]


def phase_d8_instrumentation_arm_config(arm: str, base_steps: int, h: int | None = None) -> dict:
    """D8.3 instrumentation-only locked SAF v1 config.

    This intentionally preserves baseline operator sampling, strict acceptance,
    and K(H). The only behavior change is extra logging requested via
    --d8-state-log.
    """
    if h is None:
        raise ValueError("Phase D8 instrumentation config requires H")
    k_by_h = {128: 9, 256: 18, 384: 9}
    if h not in k_by_h:
        raise ValueError(f"Phase D8 instrumentation supports only H in {sorted(k_by_h)}, got {h}")
    if arm != "D8_INSTRUMENTED":
        raise ValueError(f"unknown Phase D8 instrumentation arm: {arm}")
    return {
        **phase_d1_arm_config("D1_K1_STRICT", base_steps),
        "H": h,
        "jackpot": k_by_h[h],
        "operator_policy": None,
        "operator_prior": None,
        "operator_epsilon_random": None,
        "operator_weight_floor": None,
        "operator_weight_cap": None,
        "operator_ewma_alpha": None,
        "d8_state_log": True,
        "instrumentation_schema_version": "d8_state_log_v1",
    }


D8_ARCHIVE_MICROPROBE_ARMS = [
    "D8A_CURRENT_BEST",
    "D8A_RANDOM_ARCHIVE_PARENT",
    "D8A_SCORE_ARCHIVE_PARENT",
]

D8_P2_MICROPROBE_ARMS = [
    "D8B_CURRENT_BEST",
    "D8B_P2_PSI_CONF_LOW_DUTY",
    "D8B_P2_PSI_CONF_MED_DUTY",
]


def phase_d8_archive_microprobe_arm_config(arm: str, base_steps: int, h: int | None = None) -> dict:
    """D8.4a live archive-parent switching over locked SAF v1.

    This changes only which saved panel state is restored as the next parent.
    SAF v1 stays strict with K(H), baseline operators, and identical budget.
    """
    if h is None:
        raise ValueError("Phase D8 archive microprobe config requires H")
    k_by_h = {128: 9, 256: 18, 384: 9}
    if h not in k_by_h:
        raise ValueError(f"Phase D8 archive microprobe supports only H in {sorted(k_by_h)}, got {h}")
    policy_by_arm = {
        "D8A_CURRENT_BEST": "current-best",
        "D8A_RANDOM_ARCHIVE_PARENT": "random-archive",
        "D8A_SCORE_ARCHIVE_PARENT": "score-archive",
    }
    if arm not in policy_by_arm:
        raise ValueError(f"unknown Phase D8 archive microprobe arm: {arm}")
    return {
        **phase_d1_arm_config("D1_K1_STRICT", base_steps),
        "H": h,
        "jackpot": k_by_h[h],
        "operator_policy": None,
        "operator_prior": None,
        "operator_epsilon_random": None,
        "operator_weight_floor": None,
        "operator_weight_cap": None,
        "operator_ewma_alpha": None,
        "d8_state_log": True,
        "instrumentation_schema_version": "d8_state_log_v1",
        "archive_parent_policy": policy_by_arm[arm],
        "archive_parent_log": True,
        "archive_max_size": 64,
        "archive_switch_interval_panels": 1,
    }


def phase_d8_p2_microprobe_arm_config(arm: str, base_steps: int, h: int | None = None, model_path: Path | None = None) -> dict:
    """D8.4b live P2_PSI_CONF archive-parent switching.

    P2 score is psi_pred * cell_confidence. Switching is deliberately
    conservative versus D8.4a: low duty every 5 panels with confidence >= 0.8,
    medium duty every 3 panels with confidence >= 0.5.
    """
    if h is None:
        raise ValueError("Phase D8 P2 microprobe config requires H")
    k_by_h = {128: 9, 256: 18, 384: 9}
    if h not in k_by_h:
        raise ValueError(f"Phase D8 P2 microprobe supports only H in {sorted(k_by_h)}, got {h}")
    base = {
        **phase_d1_arm_config("D1_K1_STRICT", base_steps),
        "H": h,
        "jackpot": k_by_h[h],
        "operator_policy": None,
        "operator_prior": None,
        "operator_epsilon_random": None,
        "operator_weight_floor": None,
        "operator_weight_cap": None,
        "operator_ewma_alpha": None,
        "d8_state_log": True,
        "instrumentation_schema_version": "d8_state_log_v1",
        "archive_parent_log": True,
        "archive_max_size": 64,
    }
    if arm == "D8B_CURRENT_BEST":
        return {
            **base,
            "archive_parent_policy": "current-best",
            "archive_switch_interval_panels": 1,
            "archive_min_cell_confidence": 0.0,
            "archive_p2_model": None,
        }
    if arm == "D8B_P2_PSI_CONF_LOW_DUTY":
        return {
            **base,
            "archive_parent_policy": "p2-psi-conf",
            "archive_switch_interval_panels": 5,
            "archive_min_cell_confidence": 0.8,
            "archive_p2_model": str(model_path),
        }
    if arm == "D8B_P2_PSI_CONF_MED_DUTY":
        return {
            **base,
            "archive_parent_policy": "p2-psi-conf",
            "archive_switch_interval_panels": 3,
            "archive_min_cell_confidence": 0.5,
            "archive_p2_model": str(model_path),
        }
    raise ValueError(f"unknown Phase D8 P2 microprobe arm: {arm}")


def generate_d8_p2_model(out_dir: Path) -> Path:
    model_path = out_dir / "d8_p2_model.json"
    if model_path.exists():
        return model_path
    script = REPO_ROOT / "tools" / "export_phase_d8_p2_model.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--out", str(model_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    (out_dir / "d8_p2_model_stdout.txt").write_text(proc.stdout)
    (out_dir / "d8_p2_model_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"D8 P2 model export failed with rc={proc.returncode}")
    return model_path


def generate_operator_prior(out_dir: Path) -> Path:
    """Generate D7 operator prior from the D6.1 operator summary."""
    if not DEFAULT_D6_OPERATOR_SUMMARY.exists():
        raise FileNotFoundError(f"missing D6.1 operator summary: {DEFAULT_D6_OPERATOR_SUMMARY}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_path = out_dir / "operator_prior_by_H.csv"
    table_path = out_dir / "canonical_operator_table.csv"

    with table_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["operator_id", "baseline_probability"])
        w.writeheader()
        for operator_id, baseline in CANONICAL_OPERATORS:
            w.writerow({"operator_id": operator_id, "baseline_probability": baseline})

    usefulness: dict[tuple[int, str], float] = {}
    with DEFAULT_D6_OPERATOR_SUMMARY.open(newline="") as f:
        for row in csv.DictReader(f):
            h = int(float(row["H"]))
            operator_id = row["operator_id"]
            usefulness[(h, operator_id)] = float(row["usefulness_weighted"])

    fields = [
        "H",
        "operator_id",
        "usefulness_weighted",
        "baseline_probability",
        "raw_multiplier",
        "clipped_multiplier",
        "final_probability",
    ]
    with prior_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for h in [128, 256, 384]:
            missing = [op for op, _ in CANONICAL_OPERATORS if (h, op) not in usefulness]
            if missing:
                raise ValueError(f"D6.1 summary missing H={h} operators: {missing}")
            raw_rows = []
            # Use weighted usefulness relative to H-local baseline-weighted mean.
            weighted_mean = sum(usefulness[(h, op)] * base for op, base in CANONICAL_OPERATORS)
            if weighted_mean <= 0:
                raise ValueError(f"non-positive usefulness mean for H={h}")
            for op, base in CANONICAL_OPERATORS:
                raw_multiplier = usefulness[(h, op)] / weighted_mean
                clipped = min(4.0, max(0.25, raw_multiplier))
                raw_rows.append((op, usefulness[(h, op)], base, raw_multiplier, clipped, base * clipped))
            norm = sum(r[-1] for r in raw_rows)
            final_rows = []
            for op, u, base, raw_multiplier, clipped, weighted in raw_rows:
                weighted_prob = weighted / norm
                final_probability = 0.85 * weighted_prob + 0.15 * base
                final_rows.append((op, u, base, raw_multiplier, clipped, final_probability))
            total = sum(r[-1] for r in final_rows)
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"H={h} probabilities do not sum to 1: {total}")
            for op, u, base, raw_multiplier, clipped, final_probability in final_rows:
                if final_probability <= 0:
                    raise ValueError(f"H={h} operator={op} has non-positive final probability")
                final_multiplier = final_probability / base
                if final_multiplier < 0.25 - 1e-9 or final_multiplier > 4.0 + 1e-9:
                    raise ValueError(
                        f"H={h} operator={op} final multiplier violates floor/cap: {final_multiplier}"
                    )
                w.writerow({
                    "H": h,
                    "operator_id": op,
                    "usefulness_weighted": f"{u:.17g}",
                    "baseline_probability": f"{base:.17g}",
                    "raw_multiplier": f"{raw_multiplier:.17g}",
                    "clipped_multiplier": f"{clipped:.17g}",
                    "final_probability": f"{final_probability:.17g}",
                })
    return prior_path


def run_panel_analyzer(run_dir: Path) -> int:
    exe = example_binary_path("diag_phase_b_panel")
    if exe.exists():
        cmd = [str(exe), str(run_dir)]
    else:
        cmd = [
            "cargo", "run", "--release", "--example", "diag_phase_b_panel",
            "--manifest-path", str(REPO_ROOT / "instnct-core" / "Cargo.toml"),
            "--", str(run_dir),
        ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    (run_dir / "panel_stdout.txt").write_text(proc.stdout)
    (run_dir / "panel_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  !! panel analyzer exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
    return proc.returncode


def run_phase_b_cell(
    fixture: str,
    phase: str,
    arm: str,
    h: int,
    seed: int,
    base_steps: int,
    corpus: str,
    packed: str,
    out_dir: Path,
    panel_interval: int | None,
    cfg: dict | None = None,
) -> tuple[dict | None, int, float]:
    cfg = cfg or phase_b_arm_config(arm, base_steps)
    run_id = f"phase_{phase.lower()}_{fixture}_{arm}_H{h}_seed{seed}"
    if phase in {"D2", "D3", "D3F", "D4", "D7", "D8", "D8A", "D8B"}:
        run_dir = out_dir / f"H_{h}" / arm / f"seed_{seed}"
    else:
        run_dir = out_dir / arm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    candidate_log = run_dir / "candidates.csv"
    checkpoint = run_dir / "final.ckpt"
    panel_timeseries = run_dir / "panel_timeseries.csv"
    operator_policy_timeseries = run_dir / "operator_policy_timeseries.csv"
    d8_state_log = run_dir / "accepted_state_log.csv"
    archive_parent_log = run_dir / "archive_parent_choice_log.csv"

    cmd = cargo_example_cmd(f"evolve_{fixture}", corpus, packed, prebuilt=True) + [
        "--steps", str(cfg["steps"]),
        "--seed", str(seed),
        "--H", str(h),
        "--jackpot", str(cfg["jackpot"]),
        "--phase", phase,
        "--arm", arm,
        "--run-id", run_id,
        "--seed-list", cfg.get("seed_list", str(seed)),
        "--candidate-log", str(candidate_log),
        "--checkpoint-at-end", str(checkpoint),
    ]
    if panel_interval is not None:
        cmd += [
            "--panel-interval", str(panel_interval),
            "--panel-log", str(panel_timeseries),
        ]
    if cfg["ticks"] is not None:
        cmd += ["--ticks", str(cfg["ticks"])]
    if cfg["accept_ties"] is not None:
        cmd += ["--accept-ties", "true" if cfg["accept_ties"] else "false"]
    if cfg.get("accept_policy") is not None:
        cmd += ["--accept-policy", str(cfg["accept_policy"])]
    if cfg.get("neutral_p") is not None:
        cmd += ["--neutral-p", str(cfg["neutral_p"])]
    if cfg.get("accept_epsilon") is not None:
        cmd += ["--accept-epsilon", str(cfg["accept_epsilon"])]
    if cfg.get("operator_policy") is not None:
        cmd += ["--operator-policy", str(cfg["operator_policy"])]
    if cfg.get("operator_prior") is not None:
        cmd += ["--operator-prior", str(cfg["operator_prior"])]
    if cfg.get("operator_epsilon_random") is not None:
        cmd += ["--operator-epsilon-random", str(cfg["operator_epsilon_random"])]
    if cfg.get("operator_weight_floor") is not None:
        cmd += ["--operator-weight-floor", str(cfg["operator_weight_floor"])]
    if cfg.get("operator_weight_cap") is not None:
        cmd += ["--operator-weight-cap", str(cfg["operator_weight_cap"])]
    if cfg.get("operator_ewma_alpha") is not None:
        cmd += ["--operator-ewma-alpha", str(cfg["operator_ewma_alpha"])]
    if cfg.get("operator_policy") is not None:
        cmd += ["--operator-policy-log", str(operator_policy_timeseries)]
    if cfg.get("d8_state_log"):
        if panel_interval is None:
            raise ValueError("D8 state logging requires --panel-interval")
        cmd += ["--d8-state-log", str(d8_state_log)]
    if cfg.get("archive_parent_log"):
        if panel_interval is None:
            raise ValueError("archive parent logging requires --panel-interval")
        cmd += [
            "--archive-parent-policy", str(cfg["archive_parent_policy"]),
            "--archive-parent-log", str(archive_parent_log),
            "--archive-max-size", str(cfg.get("archive_max_size", 64)),
            "--archive-switch-interval-panels", str(cfg.get("archive_switch_interval_panels", 1)),
        ]
        if cfg.get("archive_min_cell_confidence") is not None:
            cmd += ["--archive-min-cell-confidence", str(cfg["archive_min_cell_confidence"])]
        if cfg.get("archive_p2_model"):
            cmd += ["--archive-p2-model", str(cfg["archive_p2_model"])]
    if cfg["input_scatter"]:
        cmd += ["--input-scatter"]

    (run_dir / "run_cmd.json").write_text(json.dumps({"cmd": cmd}, indent=2))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  !! cargo exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return None, proc.returncode, wall
    summary = parse_summary_line(proc.stdout)
    if summary is None:
        return None, 2, wall
    if not checkpoint.exists() or not (run_dir / "run_meta.json").exists():
        print("  !! missing checkpoint or run_meta.json", file=sys.stderr)
        return None, 3, wall
    panel_rc = run_panel_analyzer(run_dir)
    if panel_rc != 0 or not (run_dir / "panel_summary.json").exists():
        return None, panel_rc or 4, wall
    if panel_interval is not None and not panel_timeseries.exists():
        print("  !! missing panel_timeseries.csv", file=sys.stderr)
        return None, 5, wall
    if cfg.get("operator_policy") is not None and not operator_policy_timeseries.exists():
        print("  !! missing operator_policy_timeseries.csv", file=sys.stderr)
        return None, 6, wall
    if cfg.get("d8_state_log") and not d8_state_log.exists():
        print("  !! missing accepted_state_log.csv", file=sys.stderr)
        return None, 7, wall
    if cfg.get("archive_parent_log") and not archive_parent_log.exists():
        print("  !! missing archive_parent_choice_log.csv", file=sys.stderr)
        return None, 8, wall

    summary.update({
        "phase": phase,
        "arm": arm,
        "run_id": run_id,
        "seed_list": cfg.get("seed_list", str(seed)),
        "configured_steps": cfg["steps"],
        "horizon_steps": cfg["steps"],
        "jackpot": cfg["jackpot"],
        "ticks": cfg["ticks"] or 6,
        "accept_ties": cfg["accept_ties"] if cfg["accept_ties"] is not None else summary.get("accept_ties", ""),
        "accept_policy": cfg.get("accept_policy") or summary.get("accept_policy", ""),
        "neutral_p": cfg.get("neutral_p") if cfg.get("neutral_p") is not None else summary.get("neutral_p", ""),
        "accept_epsilon": cfg.get("accept_epsilon") if cfg.get("accept_epsilon") is not None else summary.get("accept_epsilon", ""),
        "operator_policy": cfg.get("operator_policy") or summary.get("operator_policy", ""),
        "operator_prior": cfg.get("operator_prior") or "",
        "operator_epsilon_random": cfg.get("operator_epsilon_random", ""),
        "operator_weight_floor": cfg.get("operator_weight_floor", ""),
        "operator_weight_cap": cfg.get("operator_weight_cap", ""),
        "operator_ewma_alpha": cfg.get("operator_ewma_alpha", ""),
        "input_scatter": cfg["input_scatter"],
        "run_dir": str(run_dir),
        "candidate_log": str(candidate_log),
        "checkpoint": str(checkpoint),
        "panel_summary": str(run_dir / "panel_summary.json"),
        "panel_timeseries": str(panel_timeseries) if panel_interval is not None else "",
        "operator_policy_timeseries": str(operator_policy_timeseries) if cfg.get("operator_policy") is not None else "",
        "instrumentation_schema_version": cfg.get("instrumentation_schema_version", ""),
        "d8_state_log": str(d8_state_log) if cfg.get("d8_state_log") else "",
        "archive_parent_policy": cfg.get("archive_parent_policy", ""),
        "archive_parent_log": str(archive_parent_log) if cfg.get("archive_parent_log") else "",
        "archive_max_size": cfg.get("archive_max_size", ""),
        "archive_switch_interval_panels": cfg.get("archive_switch_interval_panels", ""),
        "archive_min_cell_confidence": cfg.get("archive_min_cell_confidence", ""),
        "archive_p2_model": cfg.get("archive_p2_model", ""),
        "panel_window_size": panel_interval or "",
        "expected_candidate_rows": cfg["steps"] * cfg["jackpot"],
    })
    return summary, 0, wall


def write_artifacts(out_dir: Path, results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda r: (
            r.get("fixture", ""),
            r.get("arm", ""),
            int(r.get("H", 0)),
            int(r.get("seed", 0)),
        ),
    )
    (out_dir / "results.json").write_text(json.dumps({"results": results}, indent=2))
    if not results:
        return
    csv_path = out_dir / "results.csv"
    fieldnames = sorted({k for r in results for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def print_aggregate(results: list[dict]) -> None:
    by_cell: dict[tuple[str, int, str], list[dict]] = {}
    for r in results:
        by_cell.setdefault((r["fixture"], int(r["H"]), r.get("arm", "")), []).append(r)
    print("\n" + "=" * 80)
    print("AGGREGATE (mean +- std, n = # seeds)")
    print("=" * 80)
    for fixture, h, arm in sorted(by_cell):
        rows = by_cell[(fixture, h, arm)]
        peak = [r["peak_acc"] * 100 for r in rows]
        final = [r["final_acc"] * 100 for r in rows]
        acc = [r["accept_rate_pct"] for r in rows]
        alive = [r["alive_frac_mean"] for r in rows]
        wall = [r["wall_clock_s"] for r in rows]
        pm, ps = mean_std(peak)
        fm, fs = mean_std(final)
        am, as_ = mean_std(acc)
        lm, ls = mean_std(alive)
        wm, _ = mean_std(wall)
        label = f"{fixture} H={h}" + (f" arm={arm}" if arm else "")
        print(f"  {label:<36} peak={pm:>6.2f} +- {ps:>5.2f}  final={fm:>6.2f} +- {fs:>5.2f}  "
              f"accept={am:>6.2f} +- {as_:>4.2f}  alive={lm:>5.3f} +- {ls:>5.3f}  "
              f"wall={wm:>7.1f}s  n={len(rows)}")


def run_constructability_analysis(out_dir: Path) -> int:
    script = REPO_ROOT / "tools" / "diag_constructability_analysis.py"
    cmd = [sys.executable, str(script), "--root", str(out_dir)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    (out_dir / "constructability_stdout.txt").write_text(proc.stdout)
    (out_dir / "constructability_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  !! constructability analysis exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
    else:
        print(proc.stdout)
    return proc.returncode


def active_process_cpu_range() -> dict:
    """Best-effort cumulative CPU-second range for active Rust example workers."""
    if not sys.platform.startswith("win"):
        return {"available": False, "reason": "implemented for Windows process names only"}
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-Process evolve_mutual_inhibition -ErrorAction SilentlyContinue | ForEach-Object { $_.CPU }",
    ]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    vals = []
    for line in proc.stdout.splitlines():
        try:
            vals.append(float(line.strip()))
        except ValueError:
            pass
    if not vals:
        return {"available": True, "process_count": 0, "cpu_seconds_min": None, "cpu_seconds_max": None}
    return {
        "available": True,
        "process_count": len(vals),
        "cpu_seconds_min": min(vals),
        "cpu_seconds_max": max(vals),
    }


def partial_arm_means(results: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in results:
        grouped.setdefault((int(row.get("H", 0)), str(row.get("arm", ""))), []).append(row)
    out = []
    for (h, arm), rows in sorted(grouped.items()):
        peak = [float(r["peak_acc"]) * 100.0 for r in rows if "peak_acc" in r]
        final = [float(r["final_acc"]) * 100.0 for r in rows if "final_acc" in r]
        accept = [float(r["accept_rate_pct"]) for r in rows if "accept_rate_pct" in r]
        pm, ps = mean_std(peak)
        fm, _ = mean_std(final)
        am, _ = mean_std(accept)
        out.append({
            "H": h,
            "arm": arm,
            "n": len(rows),
            "peak_mean_pct": pm,
            "peak_std_pct": ps,
            "final_mean_pct": fm,
            "accept_mean_pct": am,
        })
    return out


def write_heartbeat(
    out_dir: Path,
    *,
    phase: str,
    total: int,
    results: list[dict],
    pending_meta: list[dict],
    latest_completed: dict | None,
    t_sweep: float,
    status: str,
) -> None:
    elapsed = time.time() - t_sweep
    completed = len(results)
    eta_s = (elapsed / completed) * (total - completed) if completed else None
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        "status": status,
        "elapsed_s": elapsed,
        "completed": completed,
        "total": total,
        "running_count": len(pending_meta),
        "eta_s": eta_s,
        "latest_completed": latest_completed,
        "partial_means": partial_arm_means(results),
        "active_cells": pending_meta,
        "active_process_cpu_range": active_process_cpu_range(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "heartbeat.json").write_text(json.dumps(payload, indent=2))
    with (out_dir / "heartbeat.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def load_resume(out_dir: Path, phase_b: bool) -> tuple[list[dict], set[tuple]]:
    rj = out_dir / "results.json"
    if not rj.exists():
        return [], set()
    results = json.loads(rj.read_text()).get("results", [])
    if phase_b:
        done = {(r["fixture"], r["arm"], int(r["H"]), int(r["seed"])) for r in results}
    else:
        done = {(r["fixture"], int(r["H"]), int(r["seed"])) for r in results}
    return results, done


def main_phase_b_like(args: argparse.Namespace, phase: str, config_fn, default_arms: list[str]) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    if fixtures != ["mutual_inhibition"]:
        raise SystemExit(f"--phase-{phase.lower()} currently supports only --fixtures mutual_inhibition")
    h_values = [int(x) for x in args.H_values.split(",")]
    if phase in {"B1", "D1", "D2", "D3", "D3F"} and args.arms == "B0,B1,B2,B3,B4":
        arms = default_arms
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=True)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = [
        (fx, arm, h, seed_from_idx(i))
        for fx in fixtures
        for arm in arms
        for h in h_values
        for i in range(args.seeds)
    ]
    todo = [cell for cell in cells if cell not in done]
    print(f"  Phase {phase} plan: {len(cells)} total cells, {len(todo)} to run")
    print(f"  fixtures: {fixtures}")
    print(f"  arms:     {arms}")
    print(f"  H values: {h_values}")
    print(f"  seeds:    {args.seeds} per arm -> seed pattern 42 + i*1000")
    print(f"  base steps: {args.steps}")
    if args.panel_interval is not None:
        print(f"  panel interval: {args.panel_interval}")
    print(f"  jobs:     {args.jobs}")
    print(f"  out:      {out_dir}")

    if args.dry_run:
        for fx, arm, h, seed in todo:
            cfg = config_fn(arm, args.steps)
            print(f"  DRY-RUN fixture={fx} arm={arm} H={h} seed={seed} "
                  f"steps={cfg['steps']} jackpot={cfg['jackpot']} ticks={cfg['ticks'] or 6} "
                  f"accept_ties={cfg['accept_ties']} accept_policy={cfg.get('accept_policy')} "
                  f"neutral_p={cfg.get('neutral_p')} input_scatter={cfg['input_scatter']} "
                  f"panel_interval={args.panel_interval}")
        return 0

    prebuild_rc = prebuild_phase_examples(fixtures)
    if prebuild_rc != 0:
        return prebuild_rc

    t_sweep = time.time()
    jobs = max(1, args.jobs)
    if jobs == 1:
        for idx, (fx, arm, h, seed) in enumerate(todo, 1):
            elapsed = time.time() - t_sweep
            print(f"\n[{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed}", flush=True)
            summary, rc, wall = run_phase_b_cell(
                fx,
                phase,
                arm,
                h,
                seed,
                args.steps,
                args.corpus,
                args.packed,
                out_dir,
                args.panel_interval,
                config_fn(arm, args.steps),
            )
            if summary is None:
                print(f"  FAILED (rc={rc}, wall={wall:.1f}s)", file=sys.stderr)
                write_artifacts(out_dir, results)
                return rc or 1
            summary.setdefault("wall_clock_s", wall)
            results.append(summary)
            write_artifacts(out_dir, results)
            print(f"  done: peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                  f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                  f"wall={summary['wall_clock_s']:.1f}s", flush=True)
    else:
        print(f"\nRunning Phase {phase} with {jobs} parallel jobs", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {}
            for idx, (fx, arm, h, seed) in enumerate(todo, 1):
                print(f"  queue [{idx}/{len(todo)}] fixture={fx} arm={arm} H={h} seed={seed}", flush=True)
                future = executor.submit(
                    run_phase_b_cell,
                    fx,
                    phase,
                    arm,
                    h,
                    seed,
                    args.steps,
                    args.corpus,
                    args.packed,
                    out_dir,
                    args.panel_interval,
                    config_fn(arm, args.steps),
                )
                futures[future] = (idx, fx, arm, h, seed)

            first_failure = 0
            for future in as_completed(futures):
                idx, fx, arm, h, seed = futures[future]
                elapsed = time.time() - t_sweep
                try:
                    summary, rc, wall = future.result()
                except Exception as exc:
                    summary, rc, wall = None, 1, 0.0
                    print(
                        f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                        f"fixture={fx} arm={arm} H={h} seed={seed}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                if summary is None:
                    first_failure = first_failure or (rc or 1)
                    print(
                        f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                        f"fixture={fx} arm={arm} H={h} seed={seed} rc={rc} wall={wall:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    write_artifacts(out_dir, results)
                    continue

                summary.setdefault("wall_clock_s", wall)
                results.append(summary)
                write_artifacts(out_dir, results)
                print(
                    f"  done [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                    f"fixture={fx} arm={arm} H={h} seed={seed}: "
                    f"peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                    f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                    f"wall={summary['wall_clock_s']:.1f}s",
                    flush=True,
                )
            if first_failure:
                return first_failure

    print_aggregate(results)
    analysis_rc = run_constructability_analysis(out_dir)
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return analysis_rc


def main_phase_b(args: argparse.Namespace) -> int:
    return main_phase_b_like(args, "B", phase_b_arm_config, ["B0", "B1", "B2", "B3", "B4"])


def main_phase_b1(args: argparse.Namespace) -> int:
    return main_phase_b_like(
        args,
        "B1",
        phase_b1_arm_config,
        [
            "B1_S20_STRICT",
            "B1_S20_TIES",
            "B1_S40_STRICT",
            "B1_S40_TIES",
            "B1_S80_STRICT",
            "B1_S80_TIES",
        ],
    )


def main_phase_d1(args: argparse.Namespace) -> int:
    return main_phase_b_like(
        args,
        "D1",
        phase_d1_arm_config,
        [
            "D1_K1_STRICT",
            "D1_K1_ZERO_P03",
            "D1_K1_ZERO_P10",
            "D1_K3_STRICT",
            "D1_K3_ZERO_P03",
            "D1_K3_ZERO_P10",
            "D1_K9_STRICT",
            "D1_K9_ZERO_P03",
            "D1_K9_ZERO_P10",
        ],
    )


def main_phase_d2(args: argparse.Namespace) -> int:
    return main_phase_b_like(
        args,
        "D2",
        phase_d2_arm_config,
        [
            "D2_K1_STRICT",
            "D2_K1_ZERO_P10",
            "D2_K3_STRICT",
            "D2_K3_ZERO_P10",
            "D2_K9_STRICT",
            "D2_K9_ZERO_P10",
        ],
    )


def main_phase_d3_klock(args: argparse.Namespace) -> int:
    return main_phase_b_like(
        args,
        "D3",
        phase_d3_klock_arm_config,
        [
            "D3_K5_STRICT",
            "D3_K13_STRICT",
            "D3_K18_STRICT",
        ],
    )


def main_phase_d3_fine_k(args: argparse.Namespace) -> int:
    return main_phase_b_like(
        args,
        "D3F",
        phase_d3_fine_k_arm_config,
        [
            "D3F_K15_STRICT",
            "D3F_K18_STRICT",
            "D3F_K21_STRICT",
            "D3F_K24_STRICT",
        ],
    )


def main_phase_d4_softness(args: argparse.Namespace) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    if fixtures != ["mutual_inhibition"]:
        raise SystemExit("--phase-d4-softness currently supports only --fixtures mutual_inhibition")
    requested_h = {int(x) for x in args.H_values.split(",") if x.strip()}
    valid_h = {128, 256, 384}
    invalid_h = sorted(requested_h - valid_h)
    if invalid_h:
        raise SystemExit(f"--phase-d4-softness supports only H in {sorted(valid_h)}, got {invalid_h}")

    if args.arms == "B0,B1,B2,B3,B4":
        arms = [arm for arm in D4_SOFTNESS_ARMS if phase_d4_softness_arm_config(arm, args.steps)["H"] in requested_h]
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        for arm in arms:
            cfg_h = phase_d4_softness_arm_config(arm, args.steps)["H"]
            if cfg_h not in requested_h:
                raise SystemExit(f"arm {arm} is fixed to H={cfg_h}, which is not in --H-values")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=True)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = []
    for fx in fixtures:
        for arm in arms:
            cfg = phase_d4_softness_arm_config(arm, args.steps)
            h = int(cfg["H"])
            for i in range(args.seeds):
                cells.append((fx, arm, h, seed_from_idx(i)))
    todo = [cell for cell in cells if cell not in done]
    print(f"  Phase D4 plan: {len(cells)} total cells, {len(todo)} to run")
    print(f"  fixtures: {fixtures}")
    print(f"  H/K arms: {arms}")
    print(f"  H values: {sorted(requested_h)}")
    print(f"  seeds:    {args.seeds} per arm -> seed pattern 42 + i*1000")
    print(f"  steps:    {args.steps}")
    print(f"  panel interval: {args.panel_interval}")
    print(f"  heartbeat interval: {args.heartbeat_interval}")
    print(f"  jobs:     {args.jobs}")
    print(f"  out:      {out_dir}")

    if args.dry_run:
        for fx, arm, h, seed in todo:
            cfg = phase_d4_softness_arm_config(arm, args.steps)
            print(f"  DRY-RUN fixture={fx} arm={arm} H={h} seed={seed} "
                  f"steps={cfg['steps']} jackpot={cfg['jackpot']} accept_policy={cfg.get('accept_policy')} "
                  f"neutral_p={cfg.get('neutral_p')} expected_rows={cfg['steps'] * cfg['jackpot']}")
        return 0

    prebuild_rc = prebuild_phase_examples(fixtures)
    if prebuild_rc != 0:
        return prebuild_rc

    t_sweep = time.time()
    jobs = max(1, args.jobs)
    latest_completed: dict | None = None
    first_failure = 0
    print(f"\nRunning Phase D4 with {jobs} parallel jobs", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {}
        for idx, (fx, arm, h, seed) in enumerate(todo, 1):
            cfg = phase_d4_softness_arm_config(arm, args.steps)
            print(f"  queue [{idx}/{len(todo)}] fixture={fx} arm={arm} H={h} K={cfg['jackpot']} seed={seed}", flush=True)
            future = executor.submit(
                run_phase_b_cell,
                fx,
                "D4",
                arm,
                h,
                seed,
                args.steps,
                args.corpus,
                args.packed,
                out_dir,
                args.panel_interval,
                cfg,
            )
            pending[future] = {"idx": idx, "fixture": fx, "arm": arm, "H": h, "seed": seed, "jackpot": cfg["jackpot"]}

        if args.heartbeat_interval:
            write_heartbeat(
                out_dir,
                phase="D4",
                total=len(cells),
                results=results,
                pending_meta=list(pending.values()),
                latest_completed=latest_completed,
                t_sweep=t_sweep,
                status="running",
            )

        while pending:
            timeout = args.heartbeat_interval if args.heartbeat_interval else None
            done_futures, _ = wait(set(pending.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
            if not done_futures:
                write_heartbeat(
                    out_dir,
                    phase="D4",
                    total=len(cells),
                    results=results,
                    pending_meta=list(pending.values()),
                    latest_completed=latest_completed,
                    t_sweep=t_sweep,
                    status="running",
                )
                print(
                    f"  heartbeat: elapsed={(time.time() - t_sweep)/60:.1f}m "
                    f"completed={len(results)}/{len(cells)} running={len(pending)}",
                    flush=True,
                )
                continue

            for future in done_futures:
                meta = pending.pop(future)
                elapsed = time.time() - t_sweep
                try:
                    summary, rc, wall = future.result()
                except Exception as exc:
                    summary, rc, wall = None, 1, 0.0
                    print(
                        f"  FAILED [{meta['idx']}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                        f"fixture={meta['fixture']} arm={meta['arm']} H={meta['H']} seed={meta['seed']}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                if summary is None:
                    first_failure = first_failure or (rc or 1)
                    print(
                        f"  FAILED [{meta['idx']}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                        f"fixture={meta['fixture']} arm={meta['arm']} H={meta['H']} seed={meta['seed']} "
                        f"rc={rc} wall={wall:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    write_artifacts(out_dir, results)
                    continue

                summary.setdefault("wall_clock_s", wall)
                results.append(summary)
                write_artifacts(out_dir, results)
                latest_completed = {
                    "idx": meta["idx"],
                    "fixture": meta["fixture"],
                    "arm": meta["arm"],
                    "H": meta["H"],
                    "seed": meta["seed"],
                    "peak_acc_pct": summary["peak_acc"] * 100.0,
                    "final_acc_pct": summary["final_acc"] * 100.0,
                    "accept_rate_pct": summary["accept_rate_pct"],
                    "wall_clock_s": summary["wall_clock_s"],
                }
                print(
                    f"  done [{meta['idx']}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                    f"fixture={meta['fixture']} arm={meta['arm']} H={meta['H']} seed={meta['seed']}: "
                    f"peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                    f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                    f"wall={summary['wall_clock_s']:.1f}s",
                    flush=True,
                )
            if args.heartbeat_interval:
                write_heartbeat(
                    out_dir,
                    phase="D4",
                    total=len(cells),
                    results=results,
                    pending_meta=list(pending.values()),
                    latest_completed=latest_completed,
                    t_sweep=t_sweep,
                    status="running",
                )

    if first_failure:
        if args.heartbeat_interval:
            write_heartbeat(
                out_dir,
                phase="D4",
                total=len(cells),
                results=results,
                pending_meta=[],
                latest_completed=latest_completed,
                t_sweep=t_sweep,
                status="failed",
            )
        return first_failure

    print_aggregate(results)
    analysis_rc = run_constructability_analysis(out_dir)
    if args.heartbeat_interval:
        write_heartbeat(
            out_dir,
            phase="D4",
            total=len(cells),
            results=results,
            pending_meta=[],
            latest_completed=latest_completed,
            t_sweep=t_sweep,
            status="complete" if analysis_rc == 0 else "analysis_failed",
        )
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return analysis_rc


def main_phase_d7_bandit(args: argparse.Namespace) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    if fixtures != ["mutual_inhibition"]:
        raise SystemExit("--phase-d7-bandit currently supports only --fixtures mutual_inhibition")
    requested_h = {int(x) for x in args.H_values.split(",") if x.strip()}
    valid_h = {128, 256, 384}
    invalid_h = sorted(requested_h - valid_h)
    if invalid_h:
        raise SystemExit(f"--phase-d7-bandit supports only H in {sorted(valid_h)}, got {invalid_h}")

    arms = D7_BANDIT_ARMS if args.arms == "B0,B1,B2,B3,B4" else [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in D7_BANDIT_ARMS:
            raise SystemExit(f"unknown D7 arm: {arm}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_path = generate_operator_prior(out_dir)
    seed_list = [seed_from_idx(i) for i in range(args.seeds)]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    (out_dir / "d7_run_manifest.json").write_text(json.dumps({
        "phase": "D7",
        "git_commit": git_commit,
        "fixtures": fixtures,
        "H_values": sorted(requested_h),
        "seeds": seed_list,
        "K_lock": {"128": 9, "256": 18, "384": 9},
        "arms": arms,
        "operator_prior": str(prior_path),
        "operator_policy": {
            "epsilon_random": 0.15,
            "epsilon_definition": "final_probs = (1 - epsilon) * weighted_policy_probs + epsilon * baseline_probs",
            "weight_floor": 0.25,
            "weight_cap": 4.0,
            "ewma_alpha": 0.05,
            "ewma_reward": "per-candidate max(delta_U, 0), updated from live local candidate outcomes only",
            "ewma_update_granularity": "per candidate",
            "entropy_collapse_threshold": 0.60,
        },
    }, indent=2))

    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=True)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = []
    for fx in fixtures:
        for h in sorted(requested_h):
            for arm in arms:
                for seed in seed_list:
                    cells.append((fx, arm, h, seed))
    todo = [cell for cell in cells if cell not in done]
    print(f"  Phase D7 plan: {len(cells)} total cells, {len(todo)} to run")
    print(f"  fixtures: {fixtures}")
    print(f"  arms:     {arms}")
    print(f"  H values: {sorted(requested_h)}")
    print(f"  seeds:    {seed_list}")
    print(f"  steps:    {args.steps}")
    print(f"  panel interval: {args.panel_interval}")
    print(f"  jobs:     {args.jobs}")
    print(f"  prior:    {prior_path}")
    print(f"  out:      {out_dir}")

    if args.dry_run:
        for fx, arm, h, seed in todo:
            cfg = phase_d7_bandit_arm_config(arm, args.steps, h, prior_path)
            cfg["seed_list"] = ",".join(str(s) for s in seed_list)
            print(f"  DRY-RUN fixture={fx} arm={arm} H={h} seed={seed} "
                  f"steps={cfg['steps']} jackpot={cfg['jackpot']} operator_policy={cfg['operator_policy']} "
                  f"expected_rows={cfg['steps'] * cfg['jackpot']}")
        return 0

    prebuild_rc = prebuild_phase_examples(fixtures)
    if prebuild_rc != 0:
        return prebuild_rc

    t_sweep = time.time()
    jobs = max(1, args.jobs)
    first_failure = 0
    latest_completed: dict | None = None
    print(f"\nRunning Phase D7 with {jobs} parallel jobs", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {}
        for idx, (fx, arm, h, seed) in enumerate(todo, 1):
            cfg = phase_d7_bandit_arm_config(arm, args.steps, h, prior_path)
            cfg["seed_list"] = ",".join(str(s) for s in seed_list)
            print(f"  queue [{idx}/{len(todo)}] fixture={fx} arm={arm} H={h} K={cfg['jackpot']} seed={seed}", flush=True)
            future = executor.submit(
                run_phase_b_cell,
                fx,
                "D7",
                arm,
                h,
                seed,
                args.steps,
                args.corpus,
                args.packed,
                out_dir,
                args.panel_interval,
                cfg,
            )
            pending[future] = {"idx": idx, "fixture": fx, "arm": arm, "H": h, "seed": seed, "jackpot": cfg["jackpot"]}

        if args.heartbeat_interval:
            write_heartbeat(out_dir, phase="D7", total=len(cells), results=results,
                            pending_meta=list(pending.values()), latest_completed=latest_completed,
                            t_sweep=t_sweep, status="running")

        while pending:
            timeout = args.heartbeat_interval if args.heartbeat_interval else None
            done_futures, _ = wait(set(pending.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
            if not done_futures:
                write_heartbeat(out_dir, phase="D7", total=len(cells), results=results,
                                pending_meta=list(pending.values()), latest_completed=latest_completed,
                                t_sweep=t_sweep, status="running")
                print(f"  heartbeat: elapsed={(time.time() - t_sweep)/60:.1f}m completed={len(results)}/{len(cells)} running={len(pending)}", flush=True)
                continue
            for future in done_futures:
                meta = pending.pop(future)
                elapsed = time.time() - t_sweep
                try:
                    summary, rc, wall = future.result()
                except Exception as exc:
                    summary, rc, wall = None, 1, 0.0
                    print(f"  FAILED [{meta['idx']}/{len(todo)}] elapsed={elapsed/60:.1f}m {meta}: {exc}", file=sys.stderr, flush=True)
                if summary is None:
                    first_failure = first_failure or (rc or 1)
                    print(f"  FAILED [{meta['idx']}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                          f"fixture={meta['fixture']} arm={meta['arm']} H={meta['H']} seed={meta['seed']} "
                          f"rc={rc} wall={wall:.1f}s", file=sys.stderr, flush=True)
                    write_artifacts(out_dir, results)
                    continue
                summary.setdefault("wall_clock_s", wall)
                results.append(summary)
                write_artifacts(out_dir, results)
                latest_completed = {
                    "idx": meta["idx"],
                    "fixture": meta["fixture"],
                    "arm": meta["arm"],
                    "H": meta["H"],
                    "seed": meta["seed"],
                    "peak_acc_pct": summary["peak_acc"] * 100.0,
                    "final_acc_pct": summary["final_acc"] * 100.0,
                    "accept_rate_pct": summary["accept_rate_pct"],
                    "wall_clock_s": summary["wall_clock_s"],
                }
                print(f"  done [{meta['idx']}/{len(todo)}] elapsed={elapsed/60:.1f}m "
                      f"fixture={meta['fixture']} arm={meta['arm']} H={meta['H']} seed={meta['seed']}: "
                      f"peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                      f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                      f"wall={summary['wall_clock_s']:.1f}s", flush=True)
            if args.heartbeat_interval:
                write_heartbeat(out_dir, phase="D7", total=len(cells), results=results,
                                pending_meta=list(pending.values()), latest_completed=latest_completed,
                                t_sweep=t_sweep, status="running")

    if first_failure:
        if args.heartbeat_interval:
            write_heartbeat(out_dir, phase="D7", total=len(cells), results=results,
                            pending_meta=[], latest_completed=latest_completed,
                            t_sweep=t_sweep, status="failed")
        return first_failure

    print_aggregate(results)
    rc = run_constructability_analysis(out_dir)
    analyzer = REPO_ROOT / "tools" / "analyze_phase_d7_operator_bandit.py"
    if rc == 0 and analyzer.exists():
        proc = subprocess.run([sys.executable, str(analyzer), "--root", str(out_dir)], cwd=REPO_ROOT, capture_output=True, text=True)
        (out_dir / "d7_analysis_stdout.txt").write_text(proc.stdout)
        (out_dir / "d7_analysis_stderr.txt").write_text(proc.stderr)
        print(proc.stdout)
        rc = proc.returncode
    if args.heartbeat_interval:
        write_heartbeat(out_dir, phase="D7", total=len(cells), results=results,
                        pending_meta=[], latest_completed=latest_completed, t_sweep=t_sweep,
                        status="complete" if rc == 0 else "analysis_failed")
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return rc


def main_phase_d8_instrumentation(args: argparse.Namespace) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    if fixtures != ["mutual_inhibition"]:
        raise SystemExit("--phase-d8-instrumentation currently supports only --fixtures mutual_inhibition")
    requested_h = {int(x) for x in args.H_values.split(",") if x.strip()}
    valid_h = {128, 256, 384}
    invalid_h = sorted(requested_h - valid_h)
    if invalid_h:
        raise SystemExit(f"--phase-d8-instrumentation supports only H in {sorted(valid_h)}, got {invalid_h}")
    arms = D8_INSTRUMENTATION_ARMS if args.arms == "B0,B1,B2,B3,B4" else [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in D8_INSTRUMENTATION_ARMS:
            raise SystemExit(f"unknown Phase D8 instrumentation arm: {arm}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=True)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = []
    for fx in fixtures:
        for arm in arms:
            for h in sorted(requested_h):
                for i in range(args.seeds):
                    cells.append((fx, arm, h, seed_from_idx(i)))
    todo = [cell for cell in cells if cell not in done]
    print(f"  Phase D8.3 instrumentation plan: {len(cells)} total cells, {len(todo)} to run")
    print(f"  fixtures: {fixtures}")
    print(f"  arms:     {arms}")
    print(f"  H values: {sorted(requested_h)}")
    print(f"  seeds:    {args.seeds} per arm -> seed pattern 42 + i*1000")
    print(f"  steps:    {args.steps}")
    print(f"  panel interval: {args.panel_interval}")
    print(f"  jobs:     {args.jobs}")
    print(f"  out:      {out_dir}")
    if args.panel_interval is None:
        raise SystemExit("--phase-d8-instrumentation requires --panel-interval")

    if args.dry_run:
        for fx, arm, h, seed in todo:
            cfg = phase_d8_instrumentation_arm_config(arm, args.steps, h)
            print(f"  DRY-RUN fixture={fx} arm={arm} H={h} seed={seed} "
                  f"steps={cfg['steps']} jackpot={cfg['jackpot']} "
                  f"schema={cfg['instrumentation_schema_version']} expected_rows={cfg['steps'] * cfg['jackpot']}")
        return 0

    prebuild_rc = prebuild_phase_examples(fixtures)
    if prebuild_rc != 0:
        return prebuild_rc

    t_sweep = time.time()
    jobs = max(1, args.jobs)
    first_failure = 0
    print(f"\nRunning Phase D8.3 instrumentation with {jobs} parallel jobs", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {}
        for idx, (fx, arm, h, seed) in enumerate(todo, 1):
            cfg = phase_d8_instrumentation_arm_config(arm, args.steps, h)
            print(f"  queue [{idx}/{len(todo)}] fixture={fx} arm={arm} H={h} K={cfg['jackpot']} seed={seed}", flush=True)
            future = executor.submit(
                run_phase_b_cell,
                fx,
                "D8",
                arm,
                h,
                seed,
                args.steps,
                args.corpus,
                args.packed,
                out_dir,
                args.panel_interval,
                cfg,
            )
            futures[future] = (idx, fx, arm, h, seed)
        for future in as_completed(futures):
            idx, fx, arm, h, seed = futures[future]
            elapsed = time.time() - t_sweep
            try:
                summary, rc, wall = future.result()
            except Exception as exc:
                summary, rc, wall = None, 1, 0.0
                print(f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed}: {exc}", file=sys.stderr, flush=True)
            if summary is None:
                first_failure = first_failure or (rc or 1)
                print(f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed} rc={rc} wall={wall:.1f}s", file=sys.stderr, flush=True)
                write_artifacts(out_dir, results)
                continue
            summary.setdefault("wall_clock_s", wall)
            results.append(summary)
            write_artifacts(out_dir, results)
            print(f"  done [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed}: "
                  f"peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                  f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                  f"wall={summary['wall_clock_s']:.1f}s", flush=True)
    if first_failure:
        return first_failure
    print_aggregate(results)
    rc = run_constructability_analysis(out_dir)
    analyzer = REPO_ROOT / "tools" / "analyze_phase_d8_instrumentation.py"
    if rc == 0 and analyzer.exists():
        proc = subprocess.run([sys.executable, str(analyzer), "--root", str(out_dir)], cwd=REPO_ROOT, capture_output=True, text=True)
        (out_dir / "d8_instrumentation_stdout.txt").write_text(proc.stdout)
        (out_dir / "d8_instrumentation_stderr.txt").write_text(proc.stderr)
        print(proc.stdout)
        rc = proc.returncode
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return rc


def main_phase_d8_archive_microprobe(args: argparse.Namespace) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    if fixtures != ["mutual_inhibition"]:
        raise SystemExit("--phase-d8-archive-microprobe currently supports only --fixtures mutual_inhibition")
    requested_h = {int(x) for x in args.H_values.split(",") if x.strip()}
    valid_h = {128, 256, 384}
    invalid_h = sorted(requested_h - valid_h)
    if invalid_h:
        raise SystemExit(f"--phase-d8-archive-microprobe supports only H in {sorted(valid_h)}, got {invalid_h}")
    arms = D8_ARCHIVE_MICROPROBE_ARMS if args.arms == "B0,B1,B2,B3,B4" else [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in D8_ARCHIVE_MICROPROBE_ARMS:
            raise SystemExit(f"unknown Phase D8 archive microprobe arm: {arm}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=True)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = []
    for fx in fixtures:
        for arm in arms:
            for h in sorted(requested_h):
                for i in range(args.seeds):
                    cells.append((fx, arm, h, seed_from_idx(i)))
    todo = [cell for cell in cells if cell not in done]
    print(f"  Phase D8.4a archive-parent microprobe plan: {len(cells)} total cells, {len(todo)} to run")
    print(f"  fixtures: {fixtures}")
    print(f"  arms:     {arms}")
    print(f"  H values: {sorted(requested_h)}")
    print(f"  seeds:    {args.seeds} per arm -> seed pattern 42 + i*1000")
    print(f"  steps:    {args.steps}")
    print(f"  panel interval: {args.panel_interval}")
    print(f"  jobs:     {args.jobs}")
    print(f"  out:      {out_dir}")
    if args.panel_interval is None:
        raise SystemExit("--phase-d8-archive-microprobe requires --panel-interval")

    if args.dry_run:
        for fx, arm, h, seed in todo:
            cfg = phase_d8_archive_microprobe_arm_config(arm, args.steps, h)
            print(f"  DRY-RUN fixture={fx} arm={arm} H={h} seed={seed} "
                  f"steps={cfg['steps']} jackpot={cfg['jackpot']} "
                  f"archive_policy={cfg['archive_parent_policy']} expected_rows={cfg['steps'] * cfg['jackpot']}")
        return 0

    prebuild_rc = prebuild_phase_examples(fixtures)
    if prebuild_rc != 0:
        return prebuild_rc

    t_sweep = time.time()
    jobs = max(1, args.jobs)
    first_failure = 0
    print(f"\nRunning Phase D8.4a archive-parent microprobe with {jobs} parallel jobs", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {}
        for idx, (fx, arm, h, seed) in enumerate(todo, 1):
            cfg = phase_d8_archive_microprobe_arm_config(arm, args.steps, h)
            print(f"  queue [{idx}/{len(todo)}] fixture={fx} arm={arm} H={h} K={cfg['jackpot']} policy={cfg['archive_parent_policy']} seed={seed}", flush=True)
            future = executor.submit(
                run_phase_b_cell,
                fx,
                "D8A",
                arm,
                h,
                seed,
                args.steps,
                args.corpus,
                args.packed,
                out_dir,
                args.panel_interval,
                cfg,
            )
            futures[future] = (idx, fx, arm, h, seed)
        for future in as_completed(futures):
            idx, fx, arm, h, seed = futures[future]
            elapsed = time.time() - t_sweep
            try:
                summary, rc, wall = future.result()
            except Exception as exc:
                summary, rc, wall = None, 1, 0.0
                print(f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed}: {exc}", file=sys.stderr, flush=True)
            if summary is None:
                first_failure = first_failure or (rc or 1)
                print(f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed} rc={rc} wall={wall:.1f}s", file=sys.stderr, flush=True)
                write_artifacts(out_dir, results)
                continue
            summary.setdefault("wall_clock_s", wall)
            results.append(summary)
            write_artifacts(out_dir, results)
            print(f"  done [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed}: "
                  f"peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                  f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                  f"wall={summary['wall_clock_s']:.1f}s", flush=True)
    if first_failure:
        return first_failure
    print_aggregate(results)
    rc = run_constructability_analysis(out_dir)
    analyzer = REPO_ROOT / "tools" / "analyze_phase_d8_archive_parent.py"
    if rc == 0 and analyzer.exists():
        proc = subprocess.run([sys.executable, str(analyzer), "--root", str(out_dir)], cwd=REPO_ROOT, capture_output=True, text=True)
        (out_dir / "d8_archive_parent_stdout.txt").write_text(proc.stdout)
        (out_dir / "d8_archive_parent_stderr.txt").write_text(proc.stderr)
        print(proc.stdout)
        rc = proc.returncode
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return rc


def main_phase_d8_p2_microprobe(args: argparse.Namespace) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    if fixtures != ["mutual_inhibition"]:
        raise SystemExit("--phase-d8-p2-microprobe currently supports only --fixtures mutual_inhibition")
    requested_h = {int(x) for x in args.H_values.split(",") if x.strip()}
    valid_h = {128, 256, 384}
    invalid_h = sorted(requested_h - valid_h)
    if invalid_h:
        raise SystemExit(f"--phase-d8-p2-microprobe supports only H in {sorted(valid_h)}, got {invalid_h}")
    arms = D8_P2_MICROPROBE_ARMS if args.arms == "B0,B1,B2,B3,B4" else [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in D8_P2_MICROPROBE_ARMS:
            raise SystemExit(f"unknown Phase D8 P2 microprobe arm: {arm}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = generate_d8_p2_model(out_dir)
    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=True)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = []
    for fx in fixtures:
        for arm in arms:
            for h in sorted(requested_h):
                for i in range(args.seeds):
                    cells.append((fx, arm, h, seed_from_idx(i)))
    todo = [cell for cell in cells if cell not in done]
    print(f"  Phase D8.4b P2_PSI_CONF microprobe plan: {len(cells)} total cells, {len(todo)} to run")
    print(f"  fixtures: {fixtures}")
    print(f"  arms:     {arms}")
    print(f"  H values: {sorted(requested_h)}")
    print(f"  seeds:    {args.seeds} per arm -> seed pattern 42 + i*1000")
    print(f"  steps:    {args.steps}")
    print(f"  panel interval: {args.panel_interval}")
    print(f"  p2 model: {model_path}")
    print(f"  jobs:     {args.jobs}")
    print(f"  out:      {out_dir}")
    if args.panel_interval is None:
        raise SystemExit("--phase-d8-p2-microprobe requires --panel-interval")

    if args.dry_run:
        for fx, arm, h, seed in todo:
            cfg = phase_d8_p2_microprobe_arm_config(arm, args.steps, h, model_path)
            print(f"  DRY-RUN fixture={fx} arm={arm} H={h} seed={seed} "
                  f"steps={cfg['steps']} jackpot={cfg['jackpot']} "
                  f"archive_policy={cfg['archive_parent_policy']} interval={cfg['archive_switch_interval_panels']} "
                  f"min_conf={cfg['archive_min_cell_confidence']} expected_rows={cfg['steps'] * cfg['jackpot']}")
        return 0

    prebuild_rc = prebuild_phase_examples(fixtures)
    if prebuild_rc != 0:
        return prebuild_rc

    t_sweep = time.time()
    jobs = max(1, args.jobs)
    first_failure = 0
    print(f"\nRunning Phase D8.4b P2 microprobe with {jobs} parallel jobs", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {}
        for idx, (fx, arm, h, seed) in enumerate(todo, 1):
            cfg = phase_d8_p2_microprobe_arm_config(arm, args.steps, h, model_path)
            print(f"  queue [{idx}/{len(todo)}] fixture={fx} arm={arm} H={h} K={cfg['jackpot']} policy={cfg['archive_parent_policy']} seed={seed}", flush=True)
            future = executor.submit(
                run_phase_b_cell,
                fx,
                "D8B",
                arm,
                h,
                seed,
                args.steps,
                args.corpus,
                args.packed,
                out_dir,
                args.panel_interval,
                cfg,
            )
            futures[future] = (idx, fx, arm, h, seed)
        for future in as_completed(futures):
            idx, fx, arm, h, seed = futures[future]
            elapsed = time.time() - t_sweep
            try:
                summary, rc, wall = future.result()
            except Exception as exc:
                summary, rc, wall = None, 1, 0.0
                print(f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed}: {exc}", file=sys.stderr, flush=True)
            if summary is None:
                first_failure = first_failure or (rc or 1)
                print(f"  FAILED [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed} rc={rc} wall={wall:.1f}s", file=sys.stderr, flush=True)
                write_artifacts(out_dir, results)
                continue
            summary.setdefault("wall_clock_s", wall)
            results.append(summary)
            write_artifacts(out_dir, results)
            print(f"  done [{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} arm={arm} H={h} seed={seed}: "
                  f"peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
                  f"accept={summary['accept_rate_pct']:.2f}% rows={summary['expected_candidate_rows']} "
                  f"wall={summary['wall_clock_s']:.1f}s", flush=True)
    if first_failure:
        return first_failure
    print_aggregate(results)
    rc = run_constructability_analysis(out_dir)
    analyzer = REPO_ROOT / "tools" / "analyze_phase_d8_archive_parent.py"
    if rc == 0 and analyzer.exists():
        proc = subprocess.run([sys.executable, str(analyzer), "--root", str(out_dir)], cwd=REPO_ROOT, capture_output=True, text=True)
        (out_dir / "d8_p2_archive_parent_stdout.txt").write_text(proc.stdout)
        (out_dir / "d8_p2_archive_parent_stderr.txt").write_text(proc.stderr)
        print(proc.stdout)
        rc = proc.returncode
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return rc


def main_default(args: argparse.Namespace) -> int:
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    h_values = [int(x) for x in args.H_values.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    done: set[tuple] = set()
    if args.resume:
        results, done = load_resume(out_dir, phase_b=False)
        print(f"  resume: loaded {len(results)} previous results, skipping {len(done)} cells")

    cells = [(fx, h, seed_from_idx(i)) for fx in fixtures for h in h_values for i in range(args.seeds)]
    todo = [c for c in cells if c not in done]
    print(f"  plan: {len(cells)} total cells, {len(todo)} to run, {len(done)} already done")
    print(f"  fixtures: {fixtures}")
    print(f"  H values: {h_values}")
    print(f"  seeds:    {args.seeds} per cell -> seed pattern 42 + i*1000")
    print(f"  steps:    {args.steps}")
    print(f"  out:      {out_dir}")

    if args.dry_run:
        for c in todo:
            print(f"  DRY-RUN fixture={c[0]:22s} H={c[1]:>4} seed={c[2]}")
        return 0

    t_sweep = time.time()
    for idx, (fx, h, seed) in enumerate(todo, 1):
        elapsed = time.time() - t_sweep
        print(f"\n[{idx}/{len(todo)}] elapsed={elapsed/60:.1f}m fixture={fx} H={h} seed={seed}", flush=True)
        summary, rc, wall = run_cell(fx, h, seed, args.steps, args.corpus, args.packed)
        if summary is None:
            print(f"  skipped (rc={rc}, wall={wall:.1f}s)", file=sys.stderr)
            continue
        summary.setdefault("wall_clock_s", wall)
        results.append(summary)
        write_artifacts(out_dir, results)
        print(f"  done: peak={summary['peak_acc']*100:.2f}% final={summary['final_acc']*100:.2f}% "
              f"accept={summary['accept_rate_pct']:.2f}% alive={summary['alive_frac_mean']:.3f} "
              f"edges={summary['edges']} wall={summary['wall_clock_s']:.1f}s", flush=True)

    print_aggregate(results)
    print(f"\nSweep total wall clock: {(time.time() - t_sweep) / 60:.1f} min")
    print(f"Artifacts: {out_dir / 'results.json'}  {out_dir / 'results.csv'}")
    return 0


def main() -> int:
    args = parse_args()
    if sum([args.phase_b, args.phase_b1, args.phase_d1, args.phase_d2, args.phase_d3_klock, args.phase_d3_fine_k, args.phase_d4_softness, args.phase_d7_bandit, args.phase_d8_instrumentation, args.phase_d8_archive_microprobe, args.phase_d8_p2_microprobe]) > 1:
        raise SystemExit("--phase-b, --phase-b1, --phase-d1, --phase-d2, --phase-d3-klock, --phase-d3-fine-k, --phase-d4-softness, --phase-d7-bandit, --phase-d8-instrumentation, --phase-d8-archive-microprobe, and --phase-d8-p2-microprobe are mutually exclusive")
    if args.phase_b:
        return main_phase_b(args)
    if args.phase_b1:
        return main_phase_b1(args)
    if args.phase_d1:
        return main_phase_d1(args)
    if args.phase_d2:
        return main_phase_d2(args)
    if args.phase_d3_klock:
        return main_phase_d3_klock(args)
    if args.phase_d3_fine_k:
        return main_phase_d3_fine_k(args)
    if args.phase_d4_softness:
        return main_phase_d4_softness(args)
    if args.phase_d7_bandit:
        return main_phase_d7_bandit(args)
    if args.phase_d8_instrumentation:
        return main_phase_d8_instrumentation(args)
    if args.phase_d8_archive_microprobe:
        return main_phase_d8_archive_microprobe(args)
    if args.phase_d8_p2_microprobe:
        return main_phase_d8_p2_microprobe(args)
    return main_default(args)


if __name__ == "__main__":
    raise SystemExit(main())
