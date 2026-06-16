#!/usr/bin/env python3
"""E136D OutputTextField binary matrix smoke.

This probe checks the homogeneous output-side field proposed after E136C:

rendered text -> Agency commit -> OutputTextField[N][8] -> text egress

Boundary: this is a representation/commit smoke, not a text-generation or
next-token training proof.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136D_OUTPUT_TEXT_FIELD_BINARY_MATRIX_SMOKE"
DECISION_CONFIRMED = "e136d_output_text_field_binary_matrix_confirmed"
DECISION_REJECTED = "e136d_output_text_field_binary_matrix_rejected"
NEXT = "E136E_OUTPUT_TEXT_FIELD_ROUTE_RENDER_INTEGRATION_CONFIRM"

DEFAULT_OUT = Path("target/pilot_wave/e136d_output_text_field_binary_matrix_smoke")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136d_output_text_field_binary_matrix_smoke")

ARTIFACT_FILES = (
    "run_manifest.json",
    "case_results.jsonl",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


@dataclass(frozen=True)
class OutputTextCase:
    case_id: str
    text: str
    capacity: int
    expected_action: str
    direct_write_attempt: bool = False
    tamper_after_commit: bool = False


class OutputTextField:
    def __init__(self, byte_capacity: int) -> None:
        self.byte_capacity = byte_capacity
        self.bits_per_row = 8
        self.rows = [[0 for _ in range(self.bits_per_row)] for _ in range(byte_capacity)]
        self.committed_byte_len = 0
        self.checksum = checksum32(b"")

    @property
    def shape(self) -> list[int]:
        return [self.byte_capacity, self.bits_per_row]

    def commit_from_agency(self, text: str, *, direct_write_attempt: bool = False) -> dict[str, Any]:
        if direct_write_attempt:
            return {"action": "reject", "reason": "direct_output_text_field_write_rejected"}
        data = text.encode("utf-8")
        if len(data) > self.byte_capacity:
            return {"action": "reject", "reason": "capacity_overflow"}
        if b"\x00" in data:
            return {"action": "reject", "reason": "nul_byte_rejected"}
        self.rows = [[0 for _ in range(self.bits_per_row)] for _ in range(self.byte_capacity)]
        for index, byte in enumerate(data):
            self.rows[index] = byte_to_bits(byte)
        self.committed_byte_len = len(data)
        self.checksum = checksum32(data)
        return {"action": "commit", "reason": "agency_output_text_field_commit"}

    def as_bytes(self) -> bytes:
        return bytes(bits_to_byte(row) for row in self.rows[: self.committed_byte_len])

    def as_text(self) -> str:
        return self.as_bytes().decode("utf-8")

    def verify_checksum(self) -> bool:
        try:
            return checksum32(self.as_bytes()) == self.checksum
        except ValueError:
            return False

    def zero_fill_after_commit(self) -> bool:
        return all(all(bit == 0 for bit in row) for row in self.rows[self.committed_byte_len :])

    def active_bit_count(self) -> int:
        return sum(1 for row in self.rows for bit in row if bit)

    def flip_bit_for_test(self, row: int, bit: int) -> None:
        self.rows[row][bit] ^= 1


CASES = (
    OutputTextCase("ascii_greeting", "Szia!", 16, "commit"),
    OutputTextCase("age_answer_ascii", "2251-ben leszel 250 eves.", 64, "commit"),
    OutputTextCase("age_answer_utf8", "2251-ben leszel 250 \u00e9ves.", 64, "commit"),
    OutputTextCase("json_status", '{"summary":"ok","next_steps":["verify"]}', 96, "commit"),
    OutputTextCase("multiline_code", "def sorted_copy(items):\n    return sorted(items)", 96, "commit"),
    OutputTextCase("empty_output", "", 8, "commit"),
    OutputTextCase("overflow_reject", "0123456789", 8, "reject"),
    OutputTextCase("direct_write_reject", "direct", 16, "reject", direct_write_attempt=True),
    OutputTextCase("nul_reject", "bad\x00text", 16, "reject"),
    OutputTextCase("tamper_detect", "ok", 16, "commit", tamper_after_commit=True),
)


def byte_to_bits(byte: int) -> list[int]:
    return [(byte >> shift) & 1 for shift in range(7, -1, -1)]


def bits_to_byte(bits: list[int]) -> int:
    if len(bits) != 8 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("invalid bit row")
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def checksum32(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()


def evaluate_case(case: OutputTextCase) -> dict[str, Any]:
    field = OutputTextField(case.capacity)
    verdict = field.commit_from_agency(case.text, direct_write_attempt=case.direct_write_attempt)
    action = verdict["action"]
    rendered = None
    roundtrip_pass = False
    utf8_valid = True
    if action == "commit":
        try:
            rendered = field.as_text()
            roundtrip_pass = rendered == case.text
        except UnicodeDecodeError:
            utf8_valid = False
        if case.tamper_after_commit:
            field.flip_bit_for_test(0, 0)
    checksum_valid_after_optional_tamper = field.verify_checksum()
    expected_tamper_detected = case.tamper_after_commit and not checksum_valid_after_optional_tamper
    pass_gate = (
        action == case.expected_action
        and field.shape == [case.capacity, 8]
        and field.zero_fill_after_commit()
        and (
            (action == "commit" and roundtrip_pass and utf8_valid)
            or (action == "reject" and field.committed_byte_len == 0)
        )
        and ((not case.tamper_after_commit and checksum_valid_after_optional_tamper) or expected_tamper_detected)
    )
    encoded = case.text.encode("utf-8")
    return {
        "case_id": case.case_id,
        "capacity_bytes": case.capacity,
        "matrix_shape": field.shape,
        "bits_per_row": field.bits_per_row,
        "input_char_len": len(case.text),
        "input_byte_len": len(encoded),
        "expected_action": case.expected_action,
        "actual_action": action,
        "reason": verdict["reason"],
        "committed_byte_len": field.committed_byte_len,
        "active_bit_count": field.active_bit_count(),
        "roundtrip_pass": roundtrip_pass,
        "rendered_text": rendered,
        "utf8_valid": utf8_valid,
        "zero_fill_after_commit": field.zero_fill_after_commit(),
        "checksum_valid_after_optional_tamper": checksum_valid_after_optional_tamper,
        "tamper_after_commit": case.tamper_after_commit,
        "tamper_detected": expected_tamper_detected,
        "first_four_rows": field.rows[:4],
        "pass_gate": pass_gate,
    }


def aggregate(rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    total = len(rows)
    commits = [row for row in rows if row["actual_action"] == "commit"]
    rejects = [row for row in rows if row["actual_action"] == "reject"]
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": total,
        "pass_count": sum(1 for row in rows if row["pass_gate"]),
        "fail_count": sum(1 for row in rows if not row["pass_gate"]),
        "commit_case_count": len(commits),
        "reject_case_count": len(rejects),
        "matrix_shape_pass_count": sum(1 for row in rows if row["matrix_shape"][1] == 8),
        "roundtrip_pass_count": sum(1 for row in commits if row["roundtrip_pass"]),
        "utf8_valid_count": sum(1 for row in commits if row["utf8_valid"]),
        "zero_fill_pass_count": sum(1 for row in rows if row["zero_fill_after_commit"]),
        "overflow_reject_count": sum(1 for row in rows if row["reason"] == "capacity_overflow"),
        "direct_write_reject_count": sum(1 for row in rows if row["reason"] == "direct_output_text_field_write_rejected"),
        "nul_reject_count": sum(1 for row in rows if row["reason"] == "nul_byte_rejected"),
        "tamper_detect_count": sum(1 for row in rows if row["tamper_detected"]),
        "seconds": round(seconds, 3),
    }


def write_report(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E136D OutputTextField Binary Matrix Smoke",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next     = {summary['next']}",
        "```",
        "",
        "## Metrics",
        "",
        "```text",
        f"case_count = {summary['case_count']}",
        f"pass_count = {summary['pass_count']}",
        f"fail_count = {summary['fail_count']}",
        f"commit_case_count = {summary['commit_case_count']}",
        f"reject_case_count = {summary['reject_case_count']}",
        f"matrix_shape_pass_count = {summary['matrix_shape_pass_count']}",
        f"roundtrip_pass_count = {summary['roundtrip_pass_count']}",
        f"zero_fill_pass_count = {summary['zero_fill_pass_count']}",
        f"overflow_reject_count = {summary['overflow_reject_count']}",
        f"direct_write_reject_count = {summary['direct_write_reject_count']}",
        f"nul_reject_count = {summary['nul_reject_count']}",
        f"tamper_detect_count = {summary['tamper_detect_count']}",
        "```",
        "",
        "## Interpretation",
        "",
        "OutputTextField is the canonical name for the output-side text matrix.",
        "The field is byte-shaped, not tokenizer-shaped: N rows by 8 bit cells.",
        "For ASCII, N bytes equals N characters. UTF-8 text may use more rows than",
        "visible characters.",
        "",
        "## Boundary",
        "",
        "This confirms binary output-field representation and Agency-gated commit",
        "semantics. It does not claim text generation, next-token prediction,",
        "assistant quality, or open-domain chat.",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        lines.extend([
            f"### {row['case_id']}",
            "",
            "```text",
            f"shape = {row['matrix_shape'][0]} x {row['matrix_shape'][1]}",
            f"action = {row['actual_action']}",
            f"reason = {row['reason']}",
            f"input_byte_len = {row['input_byte_len']}",
            f"committed_byte_len = {row['committed_byte_len']}",
            f"roundtrip_pass = {row['roundtrip_pass']}",
            f"zero_fill_after_commit = {row['zero_fill_after_commit']}",
            f"checksum_valid_after_optional_tamper = {row['checksum_valid_after_optional_tamper']}",
            f"pass_gate = {row['pass_gate']}",
            "```",
            "",
        ])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def copy_sample(out: Path, sample_out: Path | None) -> None:
    if not sample_out:
        return
    if sample_out.exists():
        shutil.rmtree(sample_out)
    sample_out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        src = out / name
        if src.exists():
            shutil.copy2(src, sample_out / name)


def run(out: Path, sample_out: Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    prepare_output_dir(out)
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "field_name": "OutputTextField",
        "matrix_shape": "N x 8 bits",
        "boundary": "binary output-field representation smoke; not text generation",
        "case_count": len(CASES),
    })
    rows = [evaluate_case(case) for case in CASES]
    metrics = aggregate(rows, time.perf_counter() - started)
    pass_gate = (
        metrics["fail_count"] == 0
        and metrics["case_count"] == 10
        and metrics["matrix_shape_pass_count"] == 10
        and metrics["roundtrip_pass_count"] == 7
        and metrics["zero_fill_pass_count"] == 10
        and metrics["overflow_reject_count"] == 1
        and metrics["direct_write_reject_count"] == 1
        and metrics["nul_reject_count"] == 1
        and metrics["tamper_detect_count"] == 1
    )
    decision = DECISION_CONFIRMED if pass_gate else DECISION_REJECTED
    summary = {
        **metrics,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
        "field_name": "OutputTextField",
        "boundary": "N x 8 binary output text field smoke only; not generation",
    }
    write_jsonl(out / "case_results.jsonl", rows)
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
    })
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": metrics["fail_count"],
        "failures": [row["case_id"] for row in rows if not row["pass_gate"]],
    })
    write_report(out, summary, rows)
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    sample_out = Path(args.sample_out) if args.sample_out else None
    summary = run(Path(args.out), sample_out)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "case_count": summary["case_count"],
        "pass_count": summary["pass_count"],
        "fail_count": summary["fail_count"],
        "pass_gate": summary["pass_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
