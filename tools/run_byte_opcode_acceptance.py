from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "instnct-core"
EXAMPLE = "byte_opcode_grower"
DEFAULT_GOLDEN = ROOT / "instnct-core" / "tests" / "fixtures" / "byte_opcode_golden.json"


def run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def compare(a: object, b: object, prefix: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a:
                errors.append(f"missing current key: {path}")
                continue
            if key not in b:
                errors.append(f"unexpected current key: {path}")
                continue
            errors.extend(compare(a[key], b[key], path))
        return errors
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{prefix}: expected list len {len(b)}, got {len(a)}"]
        for idx, (av, bv) in enumerate(zip(a, b)):
            errors.extend(compare(av, bv, f"{prefix}[{idx}]"))
        return errors
    if a != b:
        errors.append(f"{prefix}: expected {b!r}, got {a!r}")
    return errors


def render_summary(train: dict, reload_metrics: dict, golden_errors: list[str], reload_errors: list[str]) -> str:
    lines = [
        "# Byte Opcode Acceptance Summary",
        "",
        "## Aggregate",
        "",
        f"- direct_byte_acc: {train['direct_byte_acc']}",
        f"- translator_byte_acc: {train['translator_byte_acc']}",
        f"- distinct_keys: {train['distinct_keys']}",
        f"- conflicting_keys: {train['conflicting_keys']}",
        f"- key_bits: {train['key_bits']}",
        f"- total_neurons: {train['total_neurons']}",
        f"- max_depth: {train['max_depth']}",
        "",
        "## Per Opcode",
        "",
        f"- direct COPY/NOT/INC/DEC: {train['direct_per_op']['copy']} / {train['direct_per_op']['not']} / {train['direct_per_op']['inc']} / {train['direct_per_op']['dec']}",
        f"- translator COPY/NOT/INC/DEC: {train['translator_per_op']['copy']} / {train['translator_per_op']['not']} / {train['translator_per_op']['inc']} / {train['translator_per_op']['dec']}",
        "",
        "## Reload Check",
        "",
    ]
    if reload_errors:
        lines.append("FAILED")
        lines.extend([f"- {err}" for err in reload_errors])
    else:
        lines.append("PASS")
    lines.extend(["", "## Golden Check", ""])
    if golden_errors:
        lines.append("FAILED")
        lines.extend([f"- {err}" for err in golden_errors])
    else:
        lines.append("PASS")
    lines.extend(["", "## Adversarial", ""])
    for row in train["adversarial"]:
        lines.append(
            f"- {row['op']}({row['input']}): direct={row['direct_pred']} "
            f"({'OK' if row['direct_ok'] else 'MISS'}), lut={row['translator_pred']} "
            f"({'OK' if row['translator_ok'] else 'MISS'}), target={row['target']}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical byte+opcode acceptance and emit an append-only evidence bundle.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Output directory. Defaults to target/byte-opcode-acceptance/<timestamp>.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--write-golden", type=Path, default=None, help="Write the current train metrics as the golden file.")
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--max-neurons", type=int, default=8)
    parser.add_argument("--stall", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=120)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = args.report_dir or (ROOT / "target" / "byte-opcode-acceptance" / timestamp)
    report_dir.mkdir(parents=True, exist_ok=True)

    stack_json = report_dir / "stack.json"
    train_metrics_json = report_dir / "train_metrics.json"
    reload_metrics_json = report_dir / "reload_metrics.json"

    build_cmd = ["cargo", "build", "--example", EXAMPLE, "--release"]
    train_cmd = [
        "cargo", "run", "--example", EXAMPLE, "--release", "--",
        "--search-seed", str(args.search_seed),
        "--max-neurons", str(args.max_neurons),
        "--stall", str(args.stall),
        "--probe-epochs", str(args.probe_epochs),
        "--export-json", str(stack_json),
        "--metrics-json", str(train_metrics_json),
    ]
    reload_cmd = [
        "cargo", "run", "--example", EXAMPLE, "--release", "--",
        "--reload-json", str(stack_json),
        "--metrics-json", str(reload_metrics_json),
    ]

    run(build_cmd, CORE)
    train_stdout = run(train_cmd, CORE)
    reload_stdout = run(reload_cmd, CORE)

    train_metrics = json.loads(train_metrics_json.read_text(encoding="utf-8"))
    reload_metrics = json.loads(reload_metrics_json.read_text(encoding="utf-8"))

    reload_errors = compare(reload_metrics, train_metrics)
    golden_errors: list[str] = []
    if args.golden.exists():
        golden_metrics = json.loads(args.golden.read_text(encoding="utf-8"))
        golden_errors = compare(train_metrics, golden_metrics)

    env = {
        "cwd": str(ROOT),
        "platform": platform.platform(),
        "python": sys.version,
        "cargo_version": run(["cargo", "--version"], ROOT).strip(),
        "rustc_version": run(["rustc", "--version"], ROOT).strip(),
        "search_seed": args.search_seed,
        "max_neurons": args.max_neurons,
        "stall": args.stall,
        "probe_epochs": args.probe_epochs,
    }

    (report_dir / "train.stdout.txt").write_text(train_stdout, encoding="utf-8")
    (report_dir / "reload.stdout.txt").write_text(reload_stdout, encoding="utf-8")
    (report_dir / "run_cmd.txt").write_text("\n".join([" ".join(build_cmd), " ".join(train_cmd), " ".join(reload_cmd)]) + "\n", encoding="utf-8")
    (report_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (report_dir / "summary.md").write_text(render_summary(train_metrics, reload_metrics, golden_errors, reload_errors), encoding="utf-8")
    (report_dir / "golden_check.json").write_text(json.dumps({"errors": golden_errors}, indent=2) + "\n", encoding="utf-8")
    (report_dir / "reload_check.json").write_text(json.dumps({"errors": reload_errors}, indent=2) + "\n", encoding="utf-8")

    if args.write_golden:
        args.write_golden.parent.mkdir(parents=True, exist_ok=True)
        args.write_golden.write_text(json.dumps(train_metrics, indent=2) + "\n", encoding="utf-8")

    if reload_errors or golden_errors:
        print(render_summary(train_metrics, reload_metrics, golden_errors, reload_errors))
        return 1

    print(render_summary(train_metrics, reload_metrics, golden_errors, reload_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
