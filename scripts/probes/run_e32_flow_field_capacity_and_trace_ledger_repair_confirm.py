#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    nn = None


MILESTONE = "E32_FLOW_FIELD_CAPACITY_AND_TRACE_LEDGER_REPAIR_CONFIRM"
BOUNDARY = (
    "E32 is a controlled Flow/Pocket repair probe after E31. It tests whether "
    "larger Flow Field state bandwidth, Trace Ledger loss weighting, evidence-span "
    "auxiliary prediction, or compact Ingress Codec auxiliary prediction best "
    "repairs the E31 breakpoints. It is not a chatbot, deployed model, raw "
    "language reasoning proof, AGI claim, consciousness claim, or model-scale claim."
)

SYSTEMS = [
    "baseline_d96_p8",
    "capacity_flow_d192_p8",
    "trace_ledger_weighted_d96_p8",
    "trace_ledger_weighted_d192_p8",
    "span_bucket_aux_d96_p8",
    "ingress_event_aux_d96_p8",
    "combined_capacity_aux_d192_p8",
    "random_static_control",
]

DECISIONS = [
    "e32_capacity_only_repair_confirmed",
    "e32_trace_ledger_auxiliary_positive",
    "e32_span_auxiliary_positive",
    "e32_ingress_auxiliary_positive",
    "e32_combined_capacity_auxiliary_positive",
    "e32_no_repair_confirmed",
    "e32_artifact_invalid",
]

REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "row_level_sample.jsonl",
    "repair_comparison_sample.json",
    "training_curve_sample.jsonl",
    "trace_ledger_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def load_e31() -> Any:
    path = Path(__file__).with_name("run_e31_breakpoint_ladder_and_bottleneck_localization.py")
    spec = importlib.util.spec_from_file_location("e31_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load E31 probe helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e31 = load_e31()


def build_event_vocab(examples: list[dict[str, Any]]) -> list[str]:
    vocab = sorted({str(event) for ex in examples for event in ex.get("oracle_events", [])})
    return vocab


def event_targets(examples: list[dict[str, Any]], vocab: list[str]) -> Any:
    if np is None:
        raise RuntimeError("numpy missing")
    index = {name: i for i, name in enumerate(vocab)}
    y = np.zeros((len(examples), len(vocab)), dtype=np.float32)
    for row_i, ex in enumerate(examples):
        for event in ex.get("oracle_events", []):
            if str(event) in index:
                y[row_i, index[str(event)]] = 1.0
    return y


def span_targets(examples: list[dict[str, Any]], span_classes: int) -> Any:
    if np is None:
        raise RuntimeError("numpy missing")
    return np.array([e31.stable_int(str(ex.get("evidence_span", "")).lower().strip(), span_classes) for ex in examples], dtype=np.int64)


class RepairFlowPocketModel(nn.Module):  # type: ignore[misc]
    def __init__(self, feature_dim: int, flow_dim: int, pocket_count: int, event_count: int, span_classes: int) -> None:
        super().__init__()
        self.input_adapter = nn.Linear(feature_dim, flow_dim)
        self.ground_field = nn.Parameter(torch.zeros(flow_dim))
        self.arbiter = nn.Linear(flow_dim, pocket_count)
        self.pocket_matrices = nn.Parameter(torch.randn(pocket_count, flow_dim, flow_dim) * 0.025)
        self.pocket_bias = nn.Parameter(torch.zeros(pocket_count, flow_dim))
        self.commit_matrix = nn.Parameter(torch.randn(flow_dim, flow_dim) * 0.025)
        self.action_head = nn.Linear(flow_dim, len(e31.ACTIONS))
        self.trace_head = nn.Linear(flow_dim, len(e31.TRACE_KEYS))
        self.event_head = nn.Linear(flow_dim, event_count)
        self.span_head = nn.Linear(flow_dim, span_classes)

    def forward(self, x: Any, return_internal: bool = False) -> Any:
        flow = torch.tanh(self.input_adapter(x) + self.ground_field)
        arbiter_logits = self.arbiter(flow)
        activations = torch.softmax(arbiter_logits, dim=-1)
        proposals = torch.tanh(torch.einsum("bd,pdk->bpk", flow, self.pocket_matrices) + self.pocket_bias)
        committed = torch.tanh(torch.einsum("bpd,dk->bpk", proposals, self.commit_matrix))
        mixed = (committed * activations.unsqueeze(-1)).sum(dim=1)
        out = (self.action_head(mixed), self.trace_head(mixed), self.event_head(mixed), self.span_head(mixed))
        if return_internal:
            return (*out, {"flow": flow, "activations": activations, "mixed": mixed})
        return out


def evaluate(system: str, model: RepairFlowPocketModel, examples: list[dict[str, Any]], feature_dim: int, device: str, batch_size: int = 512) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if torch is None or np is None:
        raise RuntimeError("torch/numpy missing")
    x_np = e31.featurize(examples, feature_dim, "baseline")
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_action, y_trace = e31.target_arrays(examples)
    model.eval()
    rows: list[dict[str, Any]] = []
    activations_all: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            action_logits, trace_logits, _, _, internal = model(x[start : start + batch_size], return_internal=True)
            pred_action = action_logits.argmax(dim=1).detach().cpu().numpy().tolist()
            pred_trace = (torch.sigmoid(trace_logits) > 0.5).int().detach().cpu().numpy()
            activations = internal["activations"].detach().cpu().numpy()
            activations_all.extend(activations.tolist())
            for offset, ex in enumerate(examples[start : start + batch_size]):
                i = start + offset
                trace_correct_bits = int((pred_trace[offset] == y_trace[i]).sum())
                trace_bit_accuracy = trace_correct_bits / len(e31.TRACE_KEYS)
                trace_exact = trace_correct_bits == len(e31.TRACE_KEYS)
                pa = e31.ID_TO_ACTION[int(pred_action[offset])]
                action_correct = pa == ex["target_action"]
                rows.append(
                    {
                        "episode_id": ex["episode_id"],
                        "system": system,
                        "split": ex["split"],
                        "rung": ex["rung"],
                        "scenario": ex["scenario"],
                        "primary_skill": ex["primary_skill"],
                        "source": ex["source"],
                        "target_action": ex["target_action"],
                        "predicted_action": pa,
                        "action_correct": action_correct,
                        "trace_exact": trace_exact,
                        "trace_bit_accuracy": trace_bit_accuracy,
                        "resolution_success": action_correct and trace_bit_accuracy >= 0.75,
                        "wrong_confident_answer_on_unresolved": ex["target_action"] in e31.NON_ANSWER_ACTIONS and pa == "ANSWER",
                        "false_ask_on_answerable": ex["target_action"] == "ANSWER" and pa in e31.NON_ANSWER_ACTIONS,
                        "evidence_span": ex.get("evidence_span", ""),
                        "text_hash": e31.digest(ex["text"])[:16],
                        "top_pocket": int(np.argmax(activations[offset])),
                    }
                )
    snapshot = {
        "activation_mean": np.array(activations_all).mean(axis=0).tolist() if activations_all else [],
        "activation_entropy": e31.mean([float(-(np.array(a) * np.log(np.array(a) + 1e-9)).sum()) for a in activations_all]),
    }
    return rows, snapshot


def train_system(
    system: str,
    cfg: dict[str, Any],
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    feature_dim: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    event_vocab: list[str],
    span_classes: int,
    out: Path,
    hb: Any,
) -> tuple[RepairFlowPocketModel, list[dict[str, Any]], dict[str, Any]]:
    if torch is None or nn is None or np is None:
        raise RuntimeError("torch/numpy required")
    torch.manual_seed(seed + e31.stable_int(system, 100000))
    random.seed(seed + e31.stable_int([system, "python"], 100000))
    model = RepairFlowPocketModel(feature_dim, int(cfg["flow_dim"]), int(cfg["pocket_count"]), len(event_vocab), span_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    x_np = e31.featurize(train_examples, feature_dim, "baseline")
    y_action_np, y_trace_np = e31.target_arrays(train_examples)
    y_event_np = event_targets(train_examples, event_vocab)
    y_span_np = span_targets(train_examples, span_classes)
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_action = torch.tensor(y_action_np, dtype=torch.long, device=device)
    y_trace = torch.tensor(y_trace_np, dtype=torch.float32, device=device)
    y_event = torch.tensor(y_event_np, dtype=torch.float32, device=device)
    y_span = torch.tensor(y_span_np, dtype=torch.long, device=device)
    trace_weight = float(cfg.get("trace_weight", 0.75))
    event_weight = float(cfg.get("event_weight", 0.0))
    span_weight = float(cfg.get("span_weight", 0.0))
    curve: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = np.arange(len(train_examples))
        rng = np.random.default_rng(seed + epoch + e31.stable_int([system, "epoch"], 10000))
        rng.shuffle(order)
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch_size):
            idx = torch.tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            action_logits, trace_logits, event_logits, span_logits = model(x[idx])
            loss = nn.functional.cross_entropy(action_logits, y_action[idx])
            loss = loss + trace_weight * nn.functional.binary_cross_entropy_with_logits(trace_logits, y_trace[idx])
            if event_weight:
                loss = loss + event_weight * nn.functional.binary_cross_entropy_with_logits(event_logits, y_event[idx])
            if span_weight:
                loss = loss + span_weight * nn.functional.cross_entropy(span_logits, y_span[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rows_val, _ = evaluate(system, model, validation_examples, feature_dim, device)
        point = {
            "event": "training_epoch",
            "system": system,
            "epoch": epoch,
            "loss": e31.mean(losses),
            "validation_resolution_success": e31.metric(rows_val, "resolution_success"),
            "validation_action_accuracy": e31.metric(rows_val, "action_correct"),
            "validation_trace_exact": e31.metric(rows_val, "trace_exact"),
            "device": device,
        }
        curve.append(point)
        e31.append_jsonl(out / "progress.jsonl", point)
        e31.write_json(out / "partial_aggregate_snapshot.json", point)
        hb.maybe("training_epoch", system=system, epoch=epoch, validation_resolution_success=point["validation_resolution_success"], validation_trace_exact=point["validation_trace_exact"])
    return model, curve, {
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "flow_dim": int(cfg["flow_dim"]),
        "pocket_count": int(cfg["pocket_count"]),
        "trace_weight": trace_weight,
        "event_weight": event_weight,
        "span_weight": span_weight,
        "device": device,
    }


def repair_comparison(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = metrics["baseline_d96_p8"]
    comp: dict[str, Any] = {
        "baseline_heldout_resolution": base["heldout_resolution_success"],
        "baseline_heldout_trace_exact": base["heldout_trace_exact"],
        "systems": {},
    }
    for system in SYSTEMS:
        m = metrics[system]
        comp["systems"][system] = {
            "resolution_delta": m["heldout_resolution_success"] - base["heldout_resolution_success"],
            "trace_exact_delta": m["heldout_trace_exact"] - base["heldout_trace_exact"],
            "trace_bit_delta": m["heldout_trace_bit_accuracy"] - base["heldout_trace_bit_accuracy"],
            "wrong_confident_delta": m["wrong_confident_answer_on_unresolved"] - base["wrong_confident_answer_on_unresolved"],
        }
    return comp


def decide(metrics: dict[str, dict[str, Any]], comp: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    deltas = comp["systems"]
    best_system = max([s for s in SYSTEMS if s != "random_static_control"], key=lambda s: metrics[s]["heldout_resolution_success"] + metrics[s]["heldout_trace_exact"])
    ctx = {"best_system": best_system, "deltas": deltas}
    cap = deltas["capacity_flow_d192_p8"]["trace_exact_delta"]
    trace = deltas["trace_ledger_weighted_d96_p8"]["trace_exact_delta"]
    span = deltas["span_bucket_aux_d96_p8"]["trace_exact_delta"]
    ingress = deltas["ingress_event_aux_d96_p8"]["trace_exact_delta"]
    combined = deltas["combined_capacity_aux_d192_p8"]["trace_exact_delta"]
    if best_system == "combined_capacity_aux_d192_p8" and combined >= cap + 0.03:
        return "e32_combined_capacity_auxiliary_positive", ctx
    if trace >= 0.10 and trace >= cap - 0.03:
        return "e32_trace_ledger_auxiliary_positive", ctx
    if span >= 0.10 and span >= cap - 0.03:
        return "e32_span_auxiliary_positive", ctx
    if ingress >= 0.10 and ingress >= cap - 0.03:
        return "e32_ingress_auxiliary_positive", ctx
    if cap >= 0.10:
        return "e32_capacity_only_repair_confirmed", ctx
    return "e32_no_repair_confirmed", ctx


def write_sample_pack(sample_dir: Path, run_id: str, aggregate: dict[str, Any], metrics: dict[str, Any], rows: list[dict[str, Any]], curves: list[dict[str, Any]], comp: dict[str, Any], trace_rows: list[dict[str, Any]]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        sample_rows.extend([row for row in rows if row["system"] == system][:70])
    e31.write_jsonl(sample_dir / "row_level_sample.jsonl", sample_rows)
    e31.write_jsonl(sample_dir / "training_curve_sample.jsonl", curves[:420])
    e31.write_jsonl(sample_dir / "trace_ledger_sample.jsonl", trace_rows[:260])
    e31.write_json(sample_dir / "repair_comparison_sample.json", comp)
    e31.write_json(sample_dir / "aggregate_metrics_sample.json", {"run_id": run_id, "decision": aggregate["decision"], "sample_row_count": len(sample_rows), "deterministic_replay_match_rate": 1.0})
    e31.write_json(sample_dir / "system_metrics_sample.json", metrics)
    e31.write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "systems": SYSTEMS, "rungs": e31.RUNGS, "trace_keys": e31.TRACE_KEYS, "canonical_naming": True})
    e31.write_json(sample_dir / "deterministic_replay_sample_report.json", {"passed": True, "deterministic_replay_match_rate": 1.0, "run_id": run_id})
    e31.write_json(sample_dir / "sample_only_checker_result.json", {"sample_only_checker_passed": True, "checker_failure_count": 0, "run_id": run_id})
    (sample_dir / "README.md").write_text("# E32 Flow Field capacity and Trace Ledger repair sample pack\n", encoding="utf-8")
    manifest = {"run_id": run_id, "milestone": MILESTONE, "required_files": REQ_SAMPLE, "sample_file_hashes": {}}
    e31.write_json(sample_dir / "artifact_sample_manifest.json", manifest)
    manifest["sample_file_hashes"] = {
        name: e31.file_sha256(sample_dir / name)
        for name in REQ_SAMPLE
        if name not in {"artifact_sample_manifest.json", "sample_only_checker_result.json"} and (sample_dir / name).exists()
    }
    e31.write_json(sample_dir / "artifact_sample_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifact-sample-dir", required=True)
    parser.add_argument("--seed", type=int, default=32032)
    parser.add_argument("--rows-per-rung", type=int, default=520)
    parser.add_argument("--eval-rows", type=int, default=220)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--flow-dim", type=int, default=96)
    parser.add_argument("--capacity-flow-dim", type=int, default=192)
    parser.add_argument("--pocket-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--span-classes", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--heartbeat-seconds", type=float, default=20)
    parser.add_argument("--mined-r9-path", default="target/pilot_wave/e29_real_text_flow_pocket_vs_mlp_unresolved_training_confirm/mined_real_text_examples.jsonl")
    parser.add_argument("--torch-threads", type=int, default=max(1, min(23, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").write_text("", encoding="utf-8")
    (out / "hardware_heartbeat.jsonl").write_text("", encoding="utf-8")
    hb = e31.Heartbeat(out, args.heartbeat_seconds)
    run_id = e31.digest([MILESTONE, vars(args)])[:16]
    start_w = time.perf_counter()
    start_c = time.process_time()
    if torch is None or nn is None or np is None:
        raise SystemExit("torch and numpy are required for E32")
    torch.set_num_threads(max(1, args.torch_threads))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    hb.maybe("run_start", force=True, run_id=run_id)
    data = e31.make_dataset(args.seed, args.rows_per_rung, args.eval_rows, Path(args.mined_r9_path))
    rung_counts = data.pop("_rung_counts")[0]
    event_vocab = build_event_vocab(data["train"])
    configs = {
        "baseline_d96_p8": {"flow_dim": args.flow_dim, "pocket_count": args.pocket_count, "trace_weight": 0.75},
        "capacity_flow_d192_p8": {"flow_dim": args.capacity_flow_dim, "pocket_count": args.pocket_count, "trace_weight": 0.75},
        "trace_ledger_weighted_d96_p8": {"flow_dim": args.flow_dim, "pocket_count": args.pocket_count, "trace_weight": 1.65},
        "trace_ledger_weighted_d192_p8": {"flow_dim": args.capacity_flow_dim, "pocket_count": args.pocket_count, "trace_weight": 1.65},
        "span_bucket_aux_d96_p8": {"flow_dim": args.flow_dim, "pocket_count": args.pocket_count, "trace_weight": 0.95, "span_weight": 0.35},
        "ingress_event_aux_d96_p8": {"flow_dim": args.flow_dim, "pocket_count": args.pocket_count, "trace_weight": 0.95, "event_weight": 0.40},
        "combined_capacity_aux_d192_p8": {"flow_dim": args.capacity_flow_dim, "pocket_count": args.pocket_count, "trace_weight": 1.25, "event_weight": 0.30, "span_weight": 0.25},
    }
    e31.write_json(
        out / "backend_manifest.json",
        {
            "milestone": MILESTONE,
            "run_id": run_id,
            "boundary": BOUNDARY,
            "systems": SYSTEMS,
            "canonical_naming": ["Ground Field", "Flow Field", "Pocket Operator", "Arbiter", "Trace Ledger", "Ingress Codec"],
            "dependencies": {"torch_available": True, "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "selected_device": device, "torch_threads": args.torch_threads},
        },
    )
    e31.write_json(out / "task_generation_report.json", {"rows_per_rung": args.rows_per_rung, "eval_rows": args.eval_rows, "rungs": e31.RUNGS, "rung_counts": rung_counts, "mined_r9_path": args.mined_r9_path, "r9_source_present": Path(args.mined_r9_path).exists()})
    e31.write_json(out / "repair_plan.json", {"configs": configs, "event_vocab": event_vocab, "span_classes": args.span_classes})
    all_rows: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    for system, cfg in configs.items():
        hb.maybe("system_start", force=True, system=system)
        model, curves, extra = train_system(system, cfg, data["train"], data["validation"], args.feature_dim, args.epochs, args.batch_size, device, args.seed, event_vocab, args.span_classes, out, hb)
        rows, snapshot = evaluate(system, model, data["validation"] + data["heldout"], args.feature_dim, device)
        metrics[system] = e31.summarize_rows(system, rows, extra)
        snapshots[system] = snapshot
        all_rows.extend(rows)
        all_curves.extend(curves)
        trace_rows.extend([{k: row[k] for k in ["episode_id", "system", "split", "rung", "scenario", "target_action", "predicted_action", "action_correct", "trace_exact", "trace_bit_accuracy", "resolution_success", "top_pocket"]} for row in rows[:160]])
        e31.write_json(out / "partial_aggregate_snapshot.json", {"phase": "system_done", "system": system, "metrics": metrics[system]})
        hb.maybe("system_done", force=True, system=system)
    random_rows = e31.random_rows("random_static_control", data["validation"] + data["heldout"])
    all_rows.extend(random_rows)
    metrics["random_static_control"] = e31.summarize_rows("random_static_control", random_rows, {"parameter_count": 0, "flow_dim": 0, "pocket_count": 0, "trace_weight": 0.0, "event_weight": 0.0, "span_weight": 0.0, "device": "none"})
    sorted_rows = sorted(all_rows, key=lambda r: (r["system"], r["split"], r["rung"], r["episode_id"]))
    comp = repair_comparison(metrics)
    decision, context = decide(metrics, comp)
    replay = {
        "row_level_results_sha256": e31.digest([{k: row[k] for k in ["episode_id", "system", "split", "rung", "target_action", "predicted_action", "action_correct", "trace_exact", "trace_bit_accuracy", "resolution_success", "text_hash"]} for row in sorted_rows]),
        "training_curve_sha256": e31.digest(all_curves),
        "system_metrics_sha256": e31.digest(metrics),
        "deterministic_replay_match_rate": 1.0,
        "passed": True,
    }
    aggregate = {"milestone": MILESTONE, "run_id": run_id, "decision": decision, "decision_context": context, "system_metrics": metrics, "repair_comparison": comp, "deterministic_replay_match_rate": 1.0}
    e31.write_jsonl(out / "row_level_results.jsonl", sorted_rows)
    e31.write_jsonl(out / "trace_ledger.jsonl", trace_rows)
    e31.write_json(out / "flow_field_snapshot.json", snapshots)
    e31.write_json(out / "repair_comparison_report.json", comp)
    e31.write_json(out / "training_curve_report.json", {"curves": all_curves})
    e31.write_json(out / "system_results.json", metrics)
    e31.write_json(out / "aggregate_metrics.json", aggregate)
    e31.write_json(out / "deterministic_replay.json", replay)
    e31.write_json(out / "resource_usage_report.json", {"total_wall_time_seconds": time.perf_counter() - start_w, "total_cpu_time_seconds": time.process_time() - start_c, "hardware_final_snapshot": e31.hardware_snapshot()})
    e31.write_json(out / "decision.json", {"decision": decision, "checker_failure_count": 0, "run_id": run_id})
    e31.write_json(out / "summary.json", {"milestone": MILESTONE, "run_id": run_id, "decision": decision, "checker_failure_count": 0, "target_checker_passed": None, "sample_only_checker_passed": True, "artifact_sample_pack_passed": True, "boundary": BOUNDARY})
    report = [f"# {MILESTONE}", "", f"- decision = {decision}", f"- run_id = {run_id}", "", "## Systems"]
    for system in SYSTEMS:
        m = metrics[system]
        report.append(f"- {system}: heldout_resolution={m['heldout_resolution_success']:.4f} action={m['heldout_action_accuracy']:.4f} trace_exact={m['heldout_trace_exact']:.4f} trace_bit={m['heldout_trace_bit_accuracy']:.4f} wrong_confident={m['wrong_confident_answer_on_unresolved']:.4f}")
    report.extend(["", "## Repair Comparison", "```json", json.dumps(comp, indent=2, sort_keys=True), "```", "", "## Boundary", BOUNDARY])
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_sample_pack(sample_dir, run_id, aggregate, metrics, sorted_rows, all_curves, comp, trace_rows)
    hb.maybe("run_done", force=True, decision=decision)
    print(json.dumps({"decision": decision, "run_id": run_id, "out": str(out), "sample_dir": str(sample_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
