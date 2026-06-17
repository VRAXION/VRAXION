#!/usr/bin/env python3
"""E136J shadow variant apply and residual prune confirm.

This long-running probe consumes the E136I supersession ledger and repeatedly
shadow-applies selected variants against the E132/E136A corpora. It is designed
to avoid premature success exits: a passing gate keeps collecting evidence until
the configured wall-clock deadline or minimum runtime has elapsed.

Boundary: shadow evidence only. This script does not mutate the committed
runtime operator library and does not destructively prune operators.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e136h_existing_operator_refinement_mutation_prune_night_cycle import (  # noqa: E402
    PROFILES,
    current_match,
    row_text,
    semantic_hits,
)


ARTIFACT_CONTRACT = "E136J_SHADOW_VARIANT_APPLY_AND_RESIDUAL_PRUNE_CONFIRM"
DECISION_CONFIRMED = "e136j_shadow_variant_apply_and_residual_prune_confirmed"
DECISION_REJECTED = "e136j_shadow_variant_apply_and_residual_prune_rejected"
DECISION_INTERRUPTED = "e136j_shadow_variant_apply_interrupted_before_deadline"
NEXT = "E136K_OPERATOR_REPLACEMENT_APPLY_PLAN_OR_FLOW_SCALE_TRANSFER"

DEFAULT_E136I = Path("docs/research/artifact_samples/e136i_operator_supersession_and_output_ledger_planning")
DEFAULT_E132_DATASET = Path(
    "target/datasets/e132_external_math_text_seed_pack/normalized/e132_external_math_text_skill_seed.jsonl"
)
DEFAULT_E136A_DATASET = Path(
    "target/datasets/e136_assistant_text_seed_pack/normalized/e136_assistant_text_skill_seed.jsonl"
)
DEFAULT_OUT = Path("target/daytime/e136j_shadow_variant_apply_and_residual_prune_confirm")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136j_shadow_variant_apply_and_residual_prune_confirm")

ARTIFACT_FILES = (
    "run_manifest.json",
    "progress.jsonl",
    "checkpoint.json",
    "partial_summary.json",
    "replacement_shadow_ledger.json",
    "abstract_lineage_profile.json",
    "residual_prune_ledger.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


@dataclass
class JsonlCycler:
    path: Path
    handle: Any | None = None
    line_no: int = 0
    wraps: int = 0

    def __post_init__(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"dataset not found: {self.path}")
        self.handle = self.path.open("r", encoding="utf-8")

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None

    def next_rows(self, limit: int) -> list[dict[str, Any]]:
        assert self.handle is not None
        rows: list[dict[str, Any]] = []
        while len(rows) < limit:
            line = self.handle.readline()
            if not line:
                self.handle.seek(0)
                self.line_no = 0
                self.wraps += 1
                continue
            self.line_no += 1
            if line.strip():
                rows.append(json.loads(line))
        return rows


def now_ms() -> int:
    return int(time.time() * 1000)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_output_dir(out: Path, resume: bool) -> None:
    resolved = out.resolve()
    target_root = (REPO_ROOT / "target").resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"--out must resolve under {target_root}") from exc
    out.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()


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


def parse_deadline(run_until_local: str | None) -> float | None:
    if not run_until_local:
        return None
    pieces = run_until_local.strip().split(":")
    if len(pieces) not in {2, 3}:
        raise ValueError("--run-until-local must be HH:MM or HH:MM:SS")
    hour = int(pieces[0])
    minute = int(pieces[1])
    second = int(pieces[2]) if len(pieces) == 3 else 0
    now = datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.timestamp()


def load_ledger(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("decision") != "e136i_operator_supersession_and_output_ledger_confirmed":
        raise ValueError("E136I input is not confirmed")
    rows = json.loads((input_dir / "supersession_ledger.json").read_text(encoding="utf-8"))["rows"]
    return rows, {row["operator_id"]: row for row in rows}


def empty_metric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": row["operator_id"],
        "display_name": row["display_name"],
        "source": row["source"],
        "selected_variant_id": row["selected_variant_id"],
        "selected_variant_type": row["selected_variant_type"],
        "supersession_tier": row["supersession_tier"],
        "readiness": row["readiness"],
        "next_gate": row["next_gate"],
        "replacement_ready": bool(row["replacement_ready"]),
        "direct_runtime_candidate": bool(row["direct_runtime_candidate"]),
        "lineage_required": bool(row["lineage_required"]),
        "rows_seen": 0,
        "current_activation": 0,
        "selected_activation": 0,
        "shadow_pruned_activation": 0,
        "strict_activation": 0,
        "tag_only_activation": 0,
        "term_or_semantic_activation": 0,
        "strict_recall_miss": 0,
        "wrong_scope_proxy": 0,
        "hard_negative": 0,
        "unsupported_answer": 0,
        "direct_flow_write": 0,
        "top_tags": Counter(),
        "top_hits": Counter(),
        "top_family": Counter(),
        "examples": [],
    }


def selected_active_for(variant_type: str, active: bool, strict: bool) -> bool:
    if not active:
        return False
    if variant_type in {"semantic_verified_pruned", "semantic_tightened_trigger"}:
        return strict
    if variant_type == "abstract_kernel_shadow":
        return active
    return False


def tag_hint_active(profile: Any, tags: set[str]) -> bool:
    return any(tag in tags for tag in profile.tag_hints)


def clean_head(text: str, limit: int = 220) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def process_rows(
    rows: Iterable[dict[str, Any]],
    profiles: list[Any],
    ledger_by_id: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> int:
    row_count = 0
    for row in rows:
        row_count += 1
        tags = {str(tag) for tag in row.get("skill_tags", [])}
        text = row_text(row)
        lowered = text.lower()
        source = str(row.get("source") or "unknown")
        family = str(row.get("family") or "unknown")
        for profile in profiles:
            ledger_row = ledger_by_id[profile.operator_id]
            metric = metrics[profile.operator_id]
            metric["rows_seen"] += 1
            active = current_match(profile, row, tags, lowered)
            if not active:
                continue
            hits = semantic_hits(profile, text, tags, source)
            strict = bool(hits)
            selected_active = selected_active_for(ledger_row["selected_variant_type"], active, strict)
            metric["current_activation"] += 1
            metric["selected_activation"] += int(selected_active)
            metric["shadow_pruned_activation"] += int(not selected_active)
            metric["strict_activation"] += int(strict)
            metric["tag_only_activation"] += int(not strict and tag_hint_active(profile, tags))
            metric["term_or_semantic_activation"] += int(strict)
            metric["strict_recall_miss"] += int(strict and not selected_active)
            metric["wrong_scope_proxy"] += int(selected_active and not strict and ledger_row["selected_variant_type"] != "abstract_kernel_shadow")
            for tag in tags:
                metric["top_tags"][tag] += 1
            for hit in hits:
                metric["top_hits"][hit] += 1
            metric["top_family"][family] += 1
            if len(metric["examples"]) < 5:
                metric["examples"].append({
                    "record_id": row.get("record_id"),
                    "source": source,
                    "family": family,
                    "tags": sorted(tags),
                    "semantic_hits": hits,
                    "selected_active": selected_active,
                    "text_head": clean_head(text),
                })
    return row_count


def serializable_metric(metric: dict[str, Any]) -> dict[str, Any]:
    out = dict(metric)
    out["top_tags"] = metric["top_tags"].most_common(12)
    out["top_hits"] = metric["top_hits"].most_common(12)
    out["top_family"] = metric["top_family"].most_common(12)
    if out["current_activation"]:
        out["shadow_prune_ratio"] = round(out["shadow_pruned_activation"] / out["current_activation"], 6)
        out["strict_ratio"] = round(out["strict_activation"] / out["current_activation"], 6)
    else:
        out["shadow_prune_ratio"] = 0.0
        out["strict_ratio"] = 0.0
    return out


def summarize(
    metrics: dict[str, dict[str, Any]],
    cycles_completed: int,
    rows_processed: int,
    started_at: float,
    deadline_ts: float | None,
    stop_reason: str,
    pass_gate: bool | None = None,
) -> dict[str, Any]:
    serialized = [serializable_metric(metric) for metric in metrics.values()]
    replacement = [row for row in serialized if row["replacement_ready"]]
    direct = [row for row in serialized if row["direct_runtime_candidate"]]
    tightened = [row for row in serialized if row["supersession_tier"] == "T2_TIGHTENED_TRIGGER_REPLACEMENT"]
    abstract = [row for row in serialized if row["lineage_required"]]
    elapsed = time.time() - started_at
    if pass_gate is None:
        pass_gate = (
            len(serialized) == 34
            and len(replacement) == 27
            and len(direct) == 16
            and len(tightened) == 11
            and len(abstract) == 7
            and sum(row["strict_recall_miss"] for row in replacement) == 0
            and sum(row["wrong_scope_proxy"] for row in replacement) == 0
            and sum(row["hard_negative"] for row in serialized) == 0
            and sum(row["unsupported_answer"] for row in serialized) == 0
            and sum(row["direct_flow_write"] for row in serialized) == 0
            and rows_processed > 0
            and cycles_completed > 0
        )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "stop_reason": stop_reason,
        "cycles_completed": cycles_completed,
        "rows_processed": rows_processed,
        "elapsed_seconds": round(elapsed, 3),
        "deadline_epoch": deadline_ts,
        "operator_count": len(serialized),
        "replacement_ready_count": len(replacement),
        "direct_runtime_candidate_count": len(direct),
        "tightened_challenger_required_count": len(tightened),
        "abstract_lineage_required_count": len(abstract),
        "current_activation_total": sum(row["current_activation"] for row in serialized),
        "selected_activation_total": sum(row["selected_activation"] for row in serialized),
        "shadow_pruned_activation_total": sum(row["shadow_pruned_activation"] for row in serialized),
        "strict_recall_miss_total": sum(row["strict_recall_miss"] for row in serialized),
        "wrong_scope_proxy_total": sum(row["wrong_scope_proxy"] for row in serialized),
        "hard_negative_total": sum(row["hard_negative"] for row in serialized),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in serialized),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in serialized),
    }


def write_artifacts(
    out: Path,
    metrics: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    checker_failures: list[str],
) -> None:
    rows = [serializable_metric(metric) for metric in metrics.values()]
    replacement_rows = [row for row in rows if row["replacement_ready"]]
    abstract_rows = [row for row in rows if row["lineage_required"]]
    residual_rows = [
        {
            "operator_id": row["operator_id"],
            "supersession_tier": row["supersession_tier"],
            "current_activation": row["current_activation"],
            "selected_activation": row["selected_activation"],
            "shadow_pruned_activation": row["shadow_pruned_activation"],
            "shadow_prune_ratio": row["shadow_prune_ratio"],
            "strict_recall_miss": row["strict_recall_miss"],
            "wrong_scope_proxy": row["wrong_scope_proxy"],
        }
        for row in rows
    ]
    write_json(out / "replacement_shadow_ledger.json", {"rows": replacement_rows})
    write_json(out / "abstract_lineage_profile.json", {"rows": abstract_rows})
    write_json(out / "residual_prune_ledger.json", {"rows": residual_rows})
    write_json(out / "aggregate_metrics.json", {key: value for key, value in summary.items() if key not in {"decision", "pass_gate", "next"}})
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": NEXT,
        "pass_gate": summary["pass_gate"],
    })
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(checker_failures),
        "failures": checker_failures,
    })
    lines = [
        "# E136J Shadow Variant Apply And Residual Prune Confirm",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next     = {summary['next']}",
        "```",
        "",
        "## Metrics",
        "",
        "```text",
        f"stop_reason = {summary['stop_reason']}",
        f"cycles_completed = {summary['cycles_completed']}",
        f"rows_processed = {summary['rows_processed']}",
        f"elapsed_seconds = {summary['elapsed_seconds']}",
        f"replacement_ready_count = {summary['replacement_ready_count']}",
        f"direct_runtime_candidate_count = {summary['direct_runtime_candidate_count']}",
        f"tightened_challenger_required_count = {summary['tightened_challenger_required_count']}",
        f"abstract_lineage_required_count = {summary['abstract_lineage_required_count']}",
        f"current_activation_total = {summary['current_activation_total']}",
        f"selected_activation_total = {summary['selected_activation_total']}",
        f"shadow_pruned_activation_total = {summary['shadow_pruned_activation_total']}",
        f"strict_recall_miss_total = {summary['strict_recall_miss_total']}",
        f"wrong_scope_proxy_total = {summary['wrong_scope_proxy_total']}",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        "```",
        "",
        "## Boundary",
        "",
        "This is a shadow-apply confirmation artifact only. No runtime operator is",
        "destructively replaced or pruned by this run.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def should_stop(
    *,
    started_at: float,
    deadline_ts: float | None,
    min_wall_seconds: float,
    cycles_completed: int,
    min_cycles: int,
    max_cycles: int | None,
    stop_file: Path | None,
) -> tuple[bool, str]:
    if stop_file and stop_file.exists():
        return True, "stop_file"
    if max_cycles is not None and cycles_completed >= max_cycles:
        return True, "max_cycles"
    elapsed = time.time() - started_at
    deadline_reached = deadline_ts is not None and time.time() >= deadline_ts
    min_runtime_reached = elapsed >= min_wall_seconds
    min_cycles_reached = cycles_completed >= min_cycles
    if deadline_reached and min_runtime_reached and min_cycles_reached:
        return True, "deadline"
    if deadline_ts is None and min_wall_seconds > 0 and min_runtime_reached and min_cycles_reached:
        return True, "min_runtime"
    return False, "running"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_out = Path(args.sample_out) if args.sample_out else None
    stop_file = Path(args.stop_file) if args.stop_file else None
    prepare_output_dir(out, args.resume)
    deadline_ts = parse_deadline(args.run_until_local)
    ledger_rows, ledger_by_id = load_ledger(Path(args.e136i_artifact))
    profile_by_source = defaultdict(list)
    for profile in PROFILES:
        if profile.operator_id in ledger_by_id:
            profile_by_source[profile.source].append(profile)
    metrics = {row["operator_id"]: empty_metric(row) for row in ledger_rows}
    started_at = time.time()
    last_heartbeat = 0.0
    cycles_completed = 0
    rows_processed = 0
    e132 = JsonlCycler(Path(args.e132_dataset))
    e136a = JsonlCycler(Path(args.e136a_dataset))

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "shadow apply only; no destructive runtime replacement or prune",
        "e136i_artifact": str(args.e136i_artifact),
        "e132_dataset": str(args.e132_dataset),
        "e136a_dataset": str(args.e136a_dataset),
        "e132_batch_rows": args.e132_batch_rows,
        "e136a_batch_rows": args.e136a_batch_rows,
        "run_until_local": args.run_until_local,
        "deadline_epoch": deadline_ts,
        "min_wall_seconds": args.min_wall_seconds,
        "min_cycles": args.min_cycles,
        "max_cycles": args.max_cycles,
    })
    append_jsonl(out / "progress.jsonl", {
        "event": "start",
        "timestamp_ms": now_ms(),
        "deadline_epoch": deadline_ts,
        "min_wall_seconds": args.min_wall_seconds,
        "min_cycles": args.min_cycles,
    })

    stop_reason = "running"
    interrupted = False
    try:
        while True:
            stop, stop_reason = should_stop(
                started_at=started_at,
                deadline_ts=deadline_ts,
                min_wall_seconds=args.min_wall_seconds,
                cycles_completed=cycles_completed,
                min_cycles=args.min_cycles,
                max_cycles=args.max_cycles,
                stop_file=stop_file,
            )
            if stop:
                break

            e132_rows = e132.next_rows(args.e132_batch_rows)
            e136a_rows = e136a.next_rows(args.e136a_batch_rows)
            rows_processed += process_rows(e132_rows, profile_by_source["E132"], ledger_by_id, metrics)
            rows_processed += process_rows(e136a_rows, profile_by_source["E136A"], ledger_by_id, metrics)
            cycles_completed += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            now = time.time()
            if now - last_heartbeat >= args.heartbeat_seconds or cycles_completed % args.checkpoint_cycles == 0:
                partial = summarize(metrics, cycles_completed, rows_processed, started_at, deadline_ts, "running")
                write_json(out / "partial_summary.json", partial)
                write_json(out / "checkpoint.json", {
                    "artifact_contract": ARTIFACT_CONTRACT,
                    "timestamp_ms": now_ms(),
                    "cycles_completed": cycles_completed,
                    "rows_processed": rows_processed,
                    "e132_line_no": e132.line_no,
                    "e132_wraps": e132.wraps,
                    "e136a_line_no": e136a.line_no,
                    "e136a_wraps": e136a.wraps,
                    "elapsed_seconds": round(now - started_at, 3),
                    "deadline_seconds_remaining": None if deadline_ts is None else round(deadline_ts - now, 3),
                })
                append_jsonl(out / "progress.jsonl", {
                    "event": "heartbeat",
                    "timestamp_ms": now_ms(),
                    "cycles_completed": cycles_completed,
                    "rows_processed": rows_processed,
                    "pass_gate_so_far": partial["pass_gate"],
                    "current_activation_total": partial["current_activation_total"],
                    "shadow_pruned_activation_total": partial["shadow_pruned_activation_total"],
                    "strict_recall_miss_total": partial["strict_recall_miss_total"],
                    "deadline_seconds_remaining": None if deadline_ts is None else round(deadline_ts - now, 3),
                })
                last_heartbeat = now
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "keyboard_interrupt"
    finally:
        e132.close()
        e136a.close()

    summary = summarize(metrics, cycles_completed, rows_processed, started_at, deadline_ts, stop_reason)
    if interrupted:
        summary["decision"] = DECISION_INTERRUPTED
        summary["pass_gate"] = False
    checker_failures = []
    if not summary["pass_gate"]:
        checker_failures.append("e136j_pass_gate_failed")
    if stop_reason not in {"deadline", "min_runtime", "max_cycles"} and not interrupted:
        checker_failures.append(f"unexpected_stop_reason:{stop_reason}")
    write_artifacts(out, metrics, summary, checker_failures)
    append_jsonl(out / "progress.jsonl", {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "decision": summary["decision"],
        "pass_gate": summary["pass_gate"],
        "stop_reason": stop_reason,
        "cycles_completed": cycles_completed,
        "rows_processed": rows_processed,
    })
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136i-artifact", default=str(DEFAULT_E136I))
    parser.add_argument("--e132-dataset", default=str(DEFAULT_E132_DATASET))
    parser.add_argument("--e136a-dataset", default=str(DEFAULT_E136A_DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default="")
    parser.add_argument("--run-until-local", default=None, help="Local wall-clock stop time, HH:MM or HH:MM:SS")
    parser.add_argument("--min-wall-seconds", type=float, default=0.0)
    parser.add_argument("--min-cycles", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--e132-batch-rows", type=int, default=1024)
    parser.add_argument("--e136a-batch-rows", type=int, default=1024)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--checkpoint-cycles", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
        "stop_reason": summary["stop_reason"],
        "cycles_completed": summary["cycles_completed"],
        "rows_processed": summary["rows_processed"],
        "replacement_ready_count": summary["replacement_ready_count"],
        "direct_runtime_candidate_count": summary["direct_runtime_candidate_count"],
        "tightened_challenger_required_count": summary["tightened_challenger_required_count"],
        "abstract_lineage_required_count": summary["abstract_lineage_required_count"],
        "shadow_pruned_activation_total": summary["shadow_pruned_activation_total"],
        "strict_recall_miss_total": summary["strict_recall_miss_total"],
        "wrong_scope_proxy_total": summary["wrong_scope_proxy_total"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
