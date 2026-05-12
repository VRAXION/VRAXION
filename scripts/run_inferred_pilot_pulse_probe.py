#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_pilot_pulse_commit_reframe_probe as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "inferred-pilot-pulse"
REPORT_PATH = ROOT / "docs" / "research" / "INFERRED_PILOT_PULSE_PROBE.md"
SUMMARY_NAME = "inferred_pilot_pulse_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferred Pilot Pulse probe without explicit phase input.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden = 32
        args.epochs = 50
    return args


def swap_evidence_order(examples: list[base.PulseExample]) -> list[base.PulseExample]:
    return [replace(ex, cue1=ex.cue2, cue2=ex.cue1) for ex in examples]


class InferredPulseModel(base.PulseModel):
    """Pulse model with phase input removed for the main inferred-pulse arm."""

    def forward(
        self,
        examples: list[base.PulseExample],
        *,
        carry_mode: str,
        force_mode: str | None = None,
        swap_evidence: bool = False,
    ) -> dict[str, torch.Tensor]:
        input_examples = swap_evidence_order(examples) if swap_evidence else examples
        x1 = base.encode_evidence(input_examples, step=1, ablate_phase=True)
        step1 = self.evidence(x1)

        if carry_mode == "free_run":
            state1 = F.one_hot(step1["state"].argmax(dim=-1), num_classes=len(base.SELF_STATES)).float()
        elif carry_mode in {"teacher_forced_state", "oracle"}:
            state1 = F.one_hot(base.labels(examples)["state1"], num_classes=len(base.SELF_STATES)).float()
        elif carry_mode == "no_commit":
            state1 = None
        else:
            raise ValueError(carry_mode)

        x2 = base.encode_evidence(
            input_examples,
            step=2,
            previous_state=state1,
            ablate_phase=True,
            force_mode=force_mode,
        )
        step2 = self.evidence(x2)

        if carry_mode == "free_run":
            assert state1 is not None
            state2 = self.pulse_gated_state(step2["pulse"], step2["state"], state1)
        elif carry_mode in {"teacher_forced_state", "oracle"}:
            state2 = F.one_hot(base.labels(examples)["state2"], num_classes=len(base.SELF_STATES)).float()
        else:
            state2 = None

        action = self.action_trunk(base.encode_action(state2, len(examples)))
        return {
            "semantic1": step1["semantic"],
            "mode1": step1["mode"],
            "authority1": step1["authority"],
            "pulse1": step1["pulse"],
            "state1": step1["state"],
            "semantic2": step2["semantic"],
            "mode2": step2["mode"],
            "authority2": step2["authority"],
            "pulse2": step2["pulse"],
            "state2": step2["state"],
            "committed_state2": state2 if state2 is not None else torch.zeros(len(examples), len(base.SELF_STATES)),
            "action": action,
        }


class StaticNoPhaseModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * base.input_dim(), hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.semantic = nn.Linear(hidden, 2)
        self.mode = nn.Linear(hidden, len(base.MODES))
        self.authority = nn.Linear(hidden, 3)
        self.pulse = nn.Linear(hidden, len(base.PULSE_COMMANDS))
        self.state = nn.Linear(hidden, len(base.SELF_STATES))
        self.action = nn.Linear(hidden, len(base.CONTROLLER_ACTIONS))

    def forward(self, examples: list[base.PulseExample]) -> dict[str, torch.Tensor]:
        x1 = base.encode_evidence(examples, step=1, ablate_phase=True)
        x2 = base.encode_evidence(examples, step=2, ablate_phase=True)
        h = self.net(torch.cat([x1, x2], dim=-1))
        return {
            "semantic2": self.semantic(h),
            "mode2": self.mode(h),
            "authority2": self.authority(h),
            "pulse2": self.pulse(h),
            "state2": self.state(h),
            "action": self.action(h),
        }


def inferred_loss(model: InferredPulseModel, examples: list[base.PulseExample], *, carry_mode: str) -> torch.Tensor:
    pred = model(examples, carry_mode=carry_mode)
    tgt = base.labels(examples)
    oracle_action = model.action_trunk(
        base.encode_action(F.one_hot(tgt["state2"], num_classes=len(base.SELF_STATES)).float(), len(examples))
    )
    return (
        F.binary_cross_entropy_with_logits(pred["semantic1"], tgt["semantic"])
        + F.cross_entropy(pred["mode1"], tgt["mode1"])
        + F.binary_cross_entropy_with_logits(pred["authority1"], tgt["authority1"])
        + F.cross_entropy(pred["pulse1"], tgt["pulse1"])
        + F.cross_entropy(pred["state1"], tgt["state1"])
        + F.binary_cross_entropy_with_logits(pred["semantic2"], tgt["semantic"])
        + F.cross_entropy(pred["mode2"], tgt["mode2"])
        + F.binary_cross_entropy_with_logits(pred["authority2"], tgt["authority2"])
        + F.cross_entropy(pred["pulse2"], tgt["pulse2"])
        + F.cross_entropy(pred["state2"], tgt["state2"])
        + F.cross_entropy(pred["action"], tgt["action"])
        + F.cross_entropy(oracle_action, tgt["action"])
    )


def static_loss(model: StaticNoPhaseModel, examples: list[base.PulseExample]) -> torch.Tensor:
    pred = model(examples)
    tgt = base.labels(examples)
    return (
        F.binary_cross_entropy_with_logits(pred["semantic2"], tgt["semantic"])
        + F.cross_entropy(pred["mode2"], tgt["mode2"])
        + F.binary_cross_entropy_with_logits(pred["authority2"], tgt["authority2"])
        + F.cross_entropy(pred["pulse2"], tgt["pulse2"])
        + F.cross_entropy(pred["state2"], tgt["state2"])
        + F.cross_entropy(pred["action"], tgt["action"])
    )


def train_inferred_model(
    seed: int,
    split: dict[str, list[base.PulseExample]],
    *,
    args: argparse.Namespace,
    carry_mode: str,
) -> InferredPulseModel:
    base.set_seed(seed)
    model = InferredPulseModel(args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = inferred_loss(model, split["train"], carry_mode=carry_mode)
        loss.backward()
        opt.step()
    return model


def train_static_model(seed: int, split: dict[str, list[base.PulseExample]], *, args: argparse.Namespace) -> StaticNoPhaseModel:
    base.set_seed(seed)
    model = StaticNoPhaseModel(args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = static_loss(model, split["train"])
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def metrics_from_predictions(
    examples: list[base.PulseExample],
    *,
    semantic_logits: torch.Tensor,
    mode_logits: torch.Tensor,
    authority_logits: torch.Tensor,
    pulse_logits: torch.Tensor,
    state_logits: torch.Tensor,
    action_logits: torch.Tensor,
    authority1_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    out = base.metrics_from_predictions(
        examples,
        semantic_logits=semantic_logits,
        mode_logits=mode_logits,
        authority_logits=authority_logits,
        pulse_logits=pulse_logits,
        state_logits=state_logits,
        action_logits=action_logits,
        authority1_logits=authority1_logits,
    )
    tgt = base.labels(examples)
    pulse_pred = pulse_logits.argmax(dim=-1)
    delayed_confirmation = base.family_indices(examples, {"delayed_confirmation"})
    reframe_cases = base.family_indices(examples, {"delayed_correction", "reframe_after_commit"})
    recovery_cases = base.family_indices(examples, {"recovery_update"})
    out["pulse_command_accuracy"] = out["commit_timing_accuracy"]
    out["commit_after_confirmation_accuracy"] = base.acc_at(pulse_pred, tgt["pulse2"], delayed_confirmation)
    out["reframe_after_correction_accuracy"] = base.acc_at(pulse_pred, tgt["pulse2"], reframe_cases)
    out["recovery_update_accuracy"] = base.acc_at(pulse_pred, tgt["pulse2"], recovery_cases)
    return out


@torch.no_grad()
def evaluate_inferred(
    model: InferredPulseModel,
    examples: list[base.PulseExample],
    *,
    carry_mode: str,
    force_mode: str | None = None,
    swap_evidence: bool = False,
) -> dict[str, float]:
    pred = model(examples, carry_mode=carry_mode, force_mode=force_mode, swap_evidence=swap_evidence)
    return metrics_from_predictions(
        examples,
        semantic_logits=pred["semantic2"],
        mode_logits=pred["mode2"],
        authority_logits=pred["authority2"],
        pulse_logits=pred["pulse2"],
        state_logits=pred["committed_state2"] * 12.0 - 6.0,
        action_logits=pred["action"],
        authority1_logits=pred["authority1"],
    )


@torch.no_grad()
def evaluate_explicit_reference(model: base.PulseModel, examples: list[base.PulseExample]) -> dict[str, float]:
    pred = model(examples, carry_mode="recursive")
    return metrics_from_predictions(
        examples,
        semantic_logits=pred["semantic2"],
        mode_logits=pred["mode2"],
        authority_logits=pred["authority2"],
        pulse_logits=pred["pulse2"],
        state_logits=pred["committed_state2"] * 12.0 - 6.0,
        action_logits=pred["action"],
        authority1_logits=pred["authority1"],
    )


@torch.no_grad()
def evaluate_no_pulse(model: base.PulseModel, examples: list[base.PulseExample]) -> dict[str, float]:
    pred = model(examples, carry_mode="no_commit", ablate_phase=True)
    return metrics_from_predictions(
        examples,
        semantic_logits=pred["semantic2"],
        mode_logits=pred["mode2"],
        authority_logits=pred["authority2"],
        pulse_logits=pred["pulse2"],
        state_logits=pred["committed_state2"] * 12.0 - 6.0,
        action_logits=pred["action"],
        authority1_logits=pred["authority1"],
    )


@torch.no_grad()
def evaluate_static(model: StaticNoPhaseModel, examples: list[base.PulseExample]) -> dict[str, float]:
    pred = model(examples)
    return metrics_from_predictions(
        examples,
        semantic_logits=pred["semantic2"],
        mode_logits=pred["mode2"],
        authority_logits=pred["authority2"],
        pulse_logits=pred["pulse2"],
        state_logits=pred["state2"],
        action_logits=pred["action"],
    )


@torch.no_grad()
def evaluate_deterministic(examples: list[base.PulseExample], arm: str) -> dict[str, float]:
    pred = base.deterministic_predictions(examples, arm)
    return metrics_from_predictions(
        examples,
        semantic_logits=pred["semantic"],
        mode_logits=pred["mode"],
        authority_logits=pred["authority"],
        pulse_logits=pred["pulse"],
        state_logits=pred["state"],
        action_logits=pred["action"],
    )


def run_arm(
    args: argparse.Namespace,
    split: dict[str, list[base.PulseExample]],
    seed: int,
    arm: str,
) -> tuple[dict[str, Any], InferredPulseModel | None]:
    if arm == "inferred_pulse_recursive_model":
        model = train_inferred_model(seed, split, args=args, carry_mode="free_run")
        free_run = evaluate_inferred(model, split["final_test"], carry_mode="free_run")
        teacher = evaluate_inferred(model, split["final_test"], carry_mode="teacher_forced_state")
        return {"arm": arm, "seed": seed, "final_test": free_run, "teacher_forced_state": teacher}, model
    if arm == "explicit_phase_reference":
        model = base.train_pulse_model(seed, split, args=args, carry_mode="recursive")
        return {"arm": arm, "seed": seed, "final_test": evaluate_explicit_reference(model, split["final_test"])}, None
    if arm == "no_pulse_baseline":
        model = base.train_pulse_model(seed, split, args=args, carry_mode="no_commit", ablate_phase=True)
        return {"arm": arm, "seed": seed, "final_test": evaluate_no_pulse(model, split["final_test"])}, None
    if arm == "oracle_state_upper_bound":
        model = train_inferred_model(seed, split, args=args, carry_mode="teacher_forced_state")
        return {"arm": arm, "seed": seed, "final_test": evaluate_inferred(model, split["final_test"], carry_mode="teacher_forced_state")}, None
    if arm == "static_full_context_baseline":
        model = train_static_model(seed, split, args=args)
        return {"arm": arm, "seed": seed, "final_test": evaluate_static(model, split["final_test"])}, None
    if arm in {"always_commit_baseline", "never_reframe_baseline"}:
        return {"arm": arm, "seed": seed, "final_test": evaluate_deterministic(split["final_test"], arm)}, None
    raise ValueError(arm)


def summarize(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in records}):
        rows = [row for row in records if row["arm"] == arm]
        keys = sorted({key for row in rows for key in row["final_test"]})
        item: dict[str, Any] = {key: summarize([float(row["final_test"][key]) for row in rows]) for key in keys}
        control_keys = sorted({key for row in rows for key in row.get("controls", {})})
        if control_keys:
            item["controls"] = {key: summarize([float(row["controls"][key]) for row in rows if "controls" in row]) for key in control_keys}
        out[arm] = item

    base_rows = [row for row in records if row["arm"] == "inferred_pulse_recursive_model"]
    for other, key, direction in [
        ("explicit_phase_reference", "gap_vs_explicit_phase_reference", "other_minus_base"),
        ("no_pulse_baseline", "gap_vs_no_pulse", "base_minus_other"),
        ("always_commit_baseline", "gap_vs_always_commit", "base_minus_other"),
        ("never_reframe_baseline", "gap_vs_never_reframe", "base_minus_other"),
        ("static_full_context_baseline", "gap_vs_static", "base_minus_other"),
    ]:
        other_rows = [row for row in records if row["arm"] == other]
        if base_rows and other_rows:
            values = []
            for b, o in zip(base_rows, other_rows):
                b_acc = float(b["final_test"]["hard_counterfactual_action_accuracy"])
                o_acc = float(o["final_test"]["hard_counterfactual_action_accuracy"])
                values.append(o_acc - b_acc if direction == "other_minus_base" else b_acc - o_acc)
            out["inferred_pulse_recursive_model"][key] = summarize(values)
    return out


def mean_at(agg: dict[str, Any], arm: str, key: str) -> float:
    value = agg.get(arm, {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def control_mean(agg: dict[str, Any], key: str) -> float:
    value = agg.get("inferred_pulse_recursive_model", {}).get("controls", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def gap_mean(agg: dict[str, Any], key: str) -> float:
    value = agg.get("inferred_pulse_recursive_model", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    arm = "inferred_pulse_recursive_model"
    supports_wait = mean_at(agg, arm, "wait_when_ambiguous_accuracy") >= 0.90
    supports_commit = mean_at(agg, arm, "commit_after_confirmation_accuracy") >= 0.90 and mean_at(agg, arm, "missed_commit_rate") <= 0.10
    supports_reframe = mean_at(agg, arm, "reframe_after_correction_accuracy") >= 0.85 and mean_at(agg, "never_reframe_baseline", "reframe_case_action_accuracy") <= 0.60
    supports_update = mean_at(agg, arm, "recovery_update_accuracy") >= 0.85 and mean_at(agg, arm, "recovery_case_action_accuracy") >= 0.85
    supports_free = (
        mean_at(agg, arm, "action_accuracy") >= 0.95
        and mean_at(agg, arm, "committed_state_accuracy") >= 0.95
        and control_mean(agg, "free_run_vs_teacher_forced_gap") <= 0.10
    )
    no_phase_needed = gap_mean(agg, "gap_vs_explicit_phase_reference") <= 0.10
    return {
        "supports_inferred_pilot_pulse": (
            supports_wait
            and supports_commit
            and supports_reframe
            and supports_update
            and supports_free
            and mean_at(agg, arm, "pulse_command_accuracy") >= 0.90
            and mean_at(agg, arm, "false_commit_rate") <= 0.10
            and mean_at(agg, arm, "nonreality_action_leakage") <= 0.10
            and no_phase_needed
            and gap_mean(agg, "gap_vs_no_pulse") >= 0.25
        ),
        "supports_wait_inference": supports_wait,
        "supports_commit_inference": supports_commit,
        "supports_reframe_inference": supports_reframe,
        "supports_update_inference": supports_update,
        "supports_free_run_committed_state_control": supports_free,
        "explicit_phase_no_longer_required": no_phase_needed,
        "static_baseline_shortcut_detected": mean_at(agg, "static_full_context_baseline", "action_accuracy") >= 0.95 and abs(gap_mean(agg, "gap_vs_static")) <= 0.05,
        "full_pilot_v1_strengthened": (
            supports_wait
            and supports_commit
            and supports_reframe
            and supports_update
            and supports_free
            and no_phase_needed
            and control_mean(agg, "shuffled_evidence_drop") >= 0.10
            and control_mean(agg, "wrong_mode_drop") >= 0.50
        ),
    }


def fmt(item: dict[str, Any], key: str) -> str:
    value = item.get(key, {}).get("mean")
    return "null" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Inferred Pilot Pulse Probe",
        "",
        "## Goal",
        "",
        "Test whether the Pilot Pulse can be inferred from delayed evidence without explicit pulse-phase input.",
        "",
        "The primary arm receives event and cue evidence, predicts wait/commit/reframe/update/hold, carries a hard committed state, and selects the final action from that committed state only.",
        "",
        "## Main Results",
        "",
        "| Arm | Semantic | Mode | Pulse | State | Action | Wait | Commit Confirm | Reframe | Update | False Commit | Leakage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in [
        "inferred_pulse_recursive_model",
        "explicit_phase_reference",
        "no_pulse_baseline",
        "always_commit_baseline",
        "never_reframe_baseline",
        "static_full_context_baseline",
        "oracle_state_upper_bound",
        "shuffled_evidence_order_control",
        "wrong_grounding_control",
    ]:
        item = agg[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item, 'semantic_accuracy')}` | `{fmt(item, 'grounding_mode_accuracy')}` "
            f"| `{fmt(item, 'pulse_command_accuracy')}` | `{fmt(item, 'committed_state_accuracy')}` "
            f"| `{fmt(item, 'action_accuracy')}` | `{fmt(item, 'wait_when_ambiguous_accuracy')}` "
            f"| `{fmt(item, 'commit_after_confirmation_accuracy')}` | `{fmt(item, 'reframe_after_correction_accuracy')}` "
            f"| `{fmt(item, 'recovery_update_accuracy')}` | `{fmt(item, 'false_commit_rate')}` "
            f"| `{fmt(item, 'nonreality_action_leakage')}` |"
        )

    pilot = agg["inferred_pulse_recursive_model"]
    lines.extend([
        "",
        "## Gaps And Controls",
        "",
        f"- gap_vs_explicit_phase_reference: `{pilot['gap_vs_explicit_phase_reference']['mean']:.6f}`",
        f"- gap_vs_no_pulse: `{pilot['gap_vs_no_pulse']['mean']:.6f}`",
        f"- gap_vs_always_commit: `{pilot['gap_vs_always_commit']['mean']:.6f}`",
        f"- gap_vs_never_reframe: `{pilot['gap_vs_never_reframe']['mean']:.6f}`",
        f"- gap_vs_static: `{pilot['gap_vs_static']['mean']:.6f}`",
        "",
        "| Control | Mean | Std |",
        "|---|---:|---:|",
    ])
    for key, item in pilot.get("controls", {}).items():
        lines.append(f"| `{key}` | `{item['mean']:.6f}` | `{item['std']:.6f}` |")

    lines.extend([
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "A positive result means the pulse phase no longer needs to be externally supplied: the model infers wait, commit, reframe, and update from delayed grounding evidence, then acts from the hard committed state.",
        "",
        "The static full-context baseline is a label shortcut baseline. It is not counted as a Pilot mechanism unless it also matches committed-state and pulse-control behavior.",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. No consciousness, biology, quantum behavior, natural-language-understanding, full VRAXION, production, or deployment claim.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split = base.make_split()
    records: list[dict[str, Any]] = []
    base_arms = [
        "inferred_pulse_recursive_model",
        "explicit_phase_reference",
        "no_pulse_baseline",
        "always_commit_baseline",
        "never_reframe_baseline",
        "static_full_context_baseline",
        "oracle_state_upper_bound",
    ]

    for seed in range(args.seeds):
        print(f"[inferred-pilot-pulse] seed={seed}", flush=True)
        inferred_model: InferredPulseModel | None = None
        inferred_row: dict[str, Any] | None = None
        for arm_i, arm in enumerate(base_arms):
            row, model = run_arm(args, split, seed + 10_000 * arm_i, arm)
            records.append(row)
            if arm == "inferred_pulse_recursive_model":
                inferred_model = model
                inferred_row = row

        if inferred_model is not None and inferred_row is not None:
            free_run = inferred_row["final_test"]
            teacher = inferred_row["teacher_forced_state"]
            shuffled = evaluate_inferred(inferred_model, split["final_test"], carry_mode="free_run", swap_evidence=True)
            wrong = evaluate_inferred(inferred_model, split["final_test"], carry_mode="free_run", force_mode="tv")
            records.append({"arm": "shuffled_evidence_order_control", "seed": seed, "final_test": shuffled})
            records.append({"arm": "wrong_grounding_control", "seed": seed, "final_test": wrong})
            inferred_row["controls"] = {
                "free_run_vs_teacher_forced_gap": teacher["hard_counterfactual_action_accuracy"] - free_run["hard_counterfactual_action_accuracy"],
                "shuffled_evidence_drop": free_run["hard_counterfactual_action_accuracy"] - shuffled["hard_counterfactual_action_accuracy"],
                "wrong_mode_drop": free_run["reality_action_authority"] - wrong["reality_action_authority"],
                "teacher_forced_action_accuracy": teacher["hard_counterfactual_action_accuracy"],
                "teacher_forced_state_accuracy": teacher["committed_state_accuracy"],
            }

    summary = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "records": records, "aggregate": aggregate(records)}
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
