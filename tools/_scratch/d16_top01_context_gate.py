#!/usr/bin/env python3
"""
D16 H384 top_01 context gate.

The D13 release package proves that top_01 is artifact-safe, but chain diagnosis
showed a production blocker: recurrent state does not yet carry useful
sequential context. This script turns that diagnosis into a repeatable gate and
report so the next training/search step has a concrete target.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


DEFAULT_CHECKPOINT = "output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt"
DEFAULT_PACKED = "output/block_c_bytepair_champion/packed.bin"
DEFAULT_CHAIN_EXE = "target/release/examples/chain_diagnosis.exe"


def run_chain(chain_exe: Path, checkpoint: Path, packed: Path) -> str:
    if not chain_exe.exists():
        raise FileNotFoundError(f"missing chain_diagnosis executable: {chain_exe}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    if not packed.exists():
        raise FileNotFoundError(f"missing packed table: {packed}")
    result = subprocess.run(
        [str(chain_exe), str(checkpoint), str(packed)],
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def first_match(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else default


def parse_chain(text: str) -> dict[str, object]:
    checkpoint = first_match(r"Checkpoint: step=(\d+),", text)
    accuracy = first_match(r"acc=([0-9.]+)%", text)
    h = first_match(r"H=(\d+)", text)
    input_avg_diff = first_match(r"Avg pairwise diff: ([0-9.]+)/32 dims", text)
    input_reach = first_match(r"(\d+)/32 neurons reach output", text)
    output_avg_diff = first_match(r"Avg pairwise output diff: ([0-9.]+)/", text)
    unique_predictions = first_match(r"Unique predictions: (\d+)/8", text)
    context_line = re.search(r"Context-dependent predictions: (\d+)/(\d+) differ", text)
    context_diffs = int(context_line.group(1)) if context_line else 0
    context_total = int(context_line.group(2)) if context_line else 0

    if context_total == 0:
        verdict = "D16_CONTEXT_GATE_PARSE_FAIL"
    elif context_diffs == 0:
        verdict = "D16_CONTEXT_BLOCKED"
    elif context_diffs < context_total:
        verdict = "D16_PARTIAL_CONTEXT_SIGNAL"
    else:
        verdict = "D16_CONTEXT_READY"

    return {
        "verdict": verdict,
        "checkpoint_step": int(checkpoint) if checkpoint else None,
        "checkpoint_accuracy_pct": float(accuracy) if accuracy else None,
        "h": int(h) if h else None,
        "input_avg_pairwise_diff_dims": float(input_avg_diff) if input_avg_diff else None,
        "input_neurons_reaching_output": int(input_reach) if input_reach else None,
        "output_avg_pairwise_diff_dims": float(output_avg_diff) if output_avg_diff else None,
        "unique_predictions": int(unique_predictions) if unique_predictions else None,
        "context_dependent_predictions": context_diffs,
        "context_sequence_len": context_total,
    }


def write_report(path: Path, summary: dict[str, object], checkpoint: Path, packed: Path) -> None:
    verdict = summary["verdict"]
    context = f"{summary['context_dependent_predictions']}/{summary['context_sequence_len']}"
    lines = [
        "# D16 Top-01 Context Gate",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- packed_table: `{packed}`",
        f"- verdict: `{verdict}`",
        "",
        "## Measured Facts",
        "",
        f"- H: `{summary['h']}`",
        f"- checkpoint accuracy: `{summary['checkpoint_accuracy_pct']}%`",
        f"- input pairwise differentiation: `{summary['input_avg_pairwise_diff_dims']}/32 dims`",
        f"- input neurons reaching output: `{summary['input_neurons_reaching_output']}/32`",
        f"- output pairwise differentiation: `{summary['output_avg_pairwise_diff_dims']}` dims",
        f"- unique predictions on probe pairs: `{summary['unique_predictions']}/8`",
        f"- context-dependent predictions: `{context}`",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "D16_CONTEXT_BLOCKED":
        lines.extend(
            [
                "The top_01 checkpoint is structurally active, but it does not yet use recurrent state in this probe.",
                "This blocks language-like capability: a next-token model needs the same token to behave differently depending on previous context.",
                "",
                "Next useful work is explicit context-carrying training/search, not larger-H brute force.",
            ]
        )
    elif verdict == "D16_PARTIAL_CONTEXT_SIGNAL":
        lines.extend(
            [
                "The checkpoint has a partial recurrent context signal.",
                "Next work should optimize this signal while preserving the D10r-v8 artifact/state gate.",
            ]
        )
    elif verdict == "D16_CONTEXT_READY":
        lines.extend(
            [
                "The checkpoint changes predictions under sequential context in this probe.",
                "Next work can package a small interactive continuation demo and run broader context eval.",
            ]
        )
    else:
        lines.append("The chain diagnosis output could not be parsed; inspect raw_chain_diagnosis.txt.")

    lines.extend(
        [
            "",
            "## Progress Map",
            "",
            "```text",
            "Release-ready AI",
            "[=======___] ~74-76%",
            "",
            "[1] artifact-safe H384 checkpoint",
            "    DONE: top_01",
            "",
            "[2] high-H brute force",
            "    BLOCKED: projection/selectivity controls",
            "",
            "[3] context-carrying capability",
            f"    CURRENT: {verdict}",
            "",
            "[4] next unlock",
            "    context objective + edge/threshold search around top_01",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--packed", default=DEFAULT_PACKED)
    parser.add_argument("--chain-exe", default=DEFAULT_CHAIN_EXE)
    parser.add_argument("--out", default="output/phase_d16_top01_context_gate_20260502")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    packed = Path(args.packed)
    text = run_chain(Path(args.chain_exe), checkpoint, packed)
    (out / "raw_chain_diagnosis.txt").write_text(text, encoding="utf-8")
    summary = parse_chain(text)
    (out / "context_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out / "D16_TOP01_CONTEXT_GATE_REPORT.md", summary, checkpoint, packed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
