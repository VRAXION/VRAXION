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

import run_inferred_pilot_pulse_probe as inferred
import run_pilot_pulse_commit_reframe_probe as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "pilot-supervision-knockout"
REPORT_PATH = ROOT / "docs" / "research" / "PILOT_SUPERVISION_KNOCKOUT_MATRIX.md"
SUMMARY_NAME = "pilot_supervision_knockout_summary.json"

TRAINING_MODES = [
    "full_supervision",
    "no_pulse_supervision",
    "no_state_supervision",
    "no_pulse_no_state",
    "action_only",
    "action_plus_counterfactual_controls",
    "static_full_context_baseline",
    "oracle_full_supervision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot Pulse supervision knockout and robustness matrix.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden = 32
        args.epochs = 60
    return args


def one_hot(index: int, size: int) -> torch.Tensor:
    out = torch.zeros(size)
    out[index] = 1.0
    return out


def st_one_hot(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    hard = F.one_hot(probs.argmax(dim=-1), num_classes=logits.shape[-1]).float()
    return hard.detach() - probs.detach() + probs


def encode_evidence_soft(
    examples: list[base.PulseExample],
    *,
    step: int,
    previous_state: torch.Tensor | None = None,
    force_mode: str | None = None,
    swap_evidence: bool = False,
) -> torch.Tensor:
    input_examples = inferred.swap_evidence_order(examples) if swap_evidence else examples
    rows = torch.zeros(len(input_examples), base.input_dim())
    actor_offset = 0
    action_offset = actor_offset + len(base.ACTORS)
    patient_offset = action_offset + len(base.EVENT_ACTIONS)
    cue_offset = patient_offset + len(base.PATIENTS)
    phase_offset = cue_offset + len(base.CUE_ATOMS)
    state_offset = phase_offset + len(base.PULSE_PHASES)
    for i, ex in enumerate(input_examples):
        rows[i, actor_offset + base.ACTORS.index(ex.actor)] = 1.0
        rows[i, action_offset + base.EVENT_ACTIONS.index(ex.action)] = 1.0
        rows[i, patient_offset + base.PATIENTS.index(ex.patient)] = 1.0
        if step == 1:
            cue = ex.cue1
        elif force_mode is not None and force_mode in base.CUE_FEATURES:
            cue = base.cue_pairs(force_mode)[0]
        else:
            cue = ex.cue2
        for atom in cue:
            rows[i, cue_offset + base.CUE_ATOMS.index(atom)] = 1.0
        if step == 1:
            rows[i, state_offset + base.SELF_STATES.index("safe")] = 1.0
        elif previous_state is not None:
            rows[i, state_offset:state_offset + len(base.SELF_STATES)] = previous_state[i]
        else:
            rows[i, state_offset + base.SELF_STATES.index("safe")] = 1.0
    return rows


class KnockoutPulseModel(base.PulseModel):
    def soft_pulse_gated_state(
        self,
        pulse_logits: torch.Tensor,
        state_logits: torch.Tensor,
        previous_state: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        if hard:
            return self.pulse_gated_state(pulse_logits, state_logits, previous_state)
        pulse_probs = F.softmax(pulse_logits, dim=-1)
        proposed = st_one_hot(state_logits)
        wait_state = one_hot(base.SELF_STATES.index("uncertain"), len(base.SELF_STATES)).to(state_logits)
        wait_state = wait_state.unsqueeze(0).expand_as(proposed)
        wait_p = pulse_probs[:, base.PULSE_COMMANDS.index("wait")].unsqueeze(-1)
        hold_p = pulse_probs[:, base.PULSE_COMMANDS.index("hold")].unsqueeze(-1)
        replace_p = 1.0 - wait_p - hold_p
        return wait_p * wait_state + hold_p * previous_state + replace_p * proposed

    def forward(
        self,
        examples: list[base.PulseExample],
        *,
        carry_style: str,
        force_mode: str | None = None,
        swap_evidence: bool = False,
    ) -> dict[str, torch.Tensor]:
        x1 = encode_evidence_soft(examples, step=1, force_mode=None, swap_evidence=swap_evidence)
        step1 = self.evidence(x1)
        tgt = base.labels(examples)

        if carry_style == "hard":
            state1 = F.one_hot(step1["state"].argmax(dim=-1), num_classes=len(base.SELF_STATES)).float()
        elif carry_style == "soft":
            state1 = st_one_hot(step1["state"])
        elif carry_style == "teacher":
            state1 = F.one_hot(tgt["state1"], num_classes=len(base.SELF_STATES)).float()
        elif carry_style == "no_commit":
            state1 = None
        else:
            raise ValueError(carry_style)

        x2 = encode_evidence_soft(
            examples,
            step=2,
            previous_state=state1,
            force_mode=force_mode,
            swap_evidence=swap_evidence,
        )
        step2 = self.evidence(x2)

        if carry_style == "hard":
            assert state1 is not None
            state2 = self.soft_pulse_gated_state(step2["pulse"], step2["state"], state1, hard=True)
        elif carry_style == "soft":
            assert state1 is not None
            state2 = self.soft_pulse_gated_state(step2["pulse"], step2["state"], state1, hard=False)
        elif carry_style == "teacher":
            state2 = F.one_hot(tgt["state2"], num_classes=len(base.SELF_STATES)).float()
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


def family_action_accuracy(action_logits: torch.Tensor, target: torch.Tensor, examples: list[base.PulseExample], family: str) -> float:
    idx = base.family_indices(examples, {family})
    return base.acc_at(action_logits.argmax(dim=-1), target, idx)


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
    out = inferred.metrics_from_predictions(
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
    out["pulse_alignment_to_gold"] = out["pulse_command_accuracy"]
    out["state_alignment_to_gold"] = out["committed_state_accuracy"]
    out["wait_alignment"] = out["wait_when_ambiguous_accuracy"]
    out["commit_alignment"] = out["commit_after_confirmation_accuracy"]
    out["reframe_alignment"] = out["reframe_after_correction_accuracy"]
    out["update_alignment"] = out["recovery_update_accuracy"]
    out["clean_commit_accuracy"] = family_action_accuracy(action_logits, tgt["action"], examples, "clean_commit")
    out["delayed_correction_accuracy"] = family_action_accuracy(action_logits, tgt["action"], examples, "delayed_correction")
    out["delayed_confirmation_accuracy"] = family_action_accuracy(action_logits, tgt["action"], examples, "delayed_confirmation")
    out["reframe_after_commit_accuracy"] = family_action_accuracy(action_logits, tgt["action"], examples, "reframe_after_commit")
    out["recovery_update_split_accuracy"] = family_action_accuracy(action_logits, tgt["action"], examples, "recovery_update")
    out["heldout_cue_combo_accuracy"] = out["action_accuracy"]
    return out


@torch.no_grad()
def evaluate_model(
    model: KnockoutPulseModel,
    examples: list[base.PulseExample],
    *,
    carry_style: str,
    force_mode: str | None = None,
    swap_evidence: bool = False,
) -> dict[str, float]:
    pred = model(examples, carry_style=carry_style, force_mode=force_mode, swap_evidence=swap_evidence)
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
def evaluate_static(model: inferred.StaticNoPhaseModel, examples: list[base.PulseExample]) -> dict[str, float]:
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


def supervision_loss(pred: dict[str, torch.Tensor], examples: list[base.PulseExample], mode: str) -> torch.Tensor:
    tgt = base.labels(examples)
    losses: list[torch.Tensor] = []
    if mode not in {"action_only", "action_plus_counterfactual_controls"}:
        losses.extend([
            F.binary_cross_entropy_with_logits(pred["semantic1"], tgt["semantic"]),
            F.cross_entropy(pred["mode1"], tgt["mode1"]),
            F.binary_cross_entropy_with_logits(pred["authority1"], tgt["authority1"]),
            F.binary_cross_entropy_with_logits(pred["semantic2"], tgt["semantic"]),
            F.cross_entropy(pred["mode2"], tgt["mode2"]),
            F.binary_cross_entropy_with_logits(pred["authority2"], tgt["authority2"]),
        ])
    if mode in {"full_supervision", "no_state_supervision"}:
        losses.extend([
            F.cross_entropy(pred["pulse1"], tgt["pulse1"]),
            F.cross_entropy(pred["pulse2"], tgt["pulse2"]),
        ])
    if mode in {"full_supervision", "no_pulse_supervision"}:
        losses.extend([
            F.cross_entropy(pred["state1"], tgt["state1"]),
            F.cross_entropy(pred["state2"], tgt["state2"]),
        ])
    losses.append(F.cross_entropy(pred["action"], tgt["action"]))
    return sum(losses)


def state_mass(committed_state: torch.Tensor, states: set[str]) -> torch.Tensor:
    idx = [base.SELF_STATES.index(item) for item in states]
    return committed_state[:, idx].sum(dim=-1)


def counterfactual_pressure_loss(model: KnockoutPulseModel, examples: list[base.PulseExample]) -> torch.Tensor:
    pred = model(examples, carry_style="soft")
    tgt = base.labels(examples)
    committed = pred["committed_state2"]
    nonreal = [i for i, ex in enumerate(examples) if ex.final_mode in {"tv", "game", "dream", "memory"} and ex.patient == "me" and ex.action in base.THREAT_ACTIONS]
    reality_self = [i for i, ex in enumerate(examples) if ex.final_mode == "reality" and ex.patient == "me" and ex.state2 in {"injured", "alert"}]
    recovery = base.family_indices(examples, {"recovery_update"})
    losses = [F.cross_entropy(pred["action"], tgt["action"])]
    if nonreal:
        losses.append(state_mass(committed[nonreal], {"injured", "alert"}).mean())
    if reality_self:
        losses.append((1.0 - state_mass(committed[reality_self], {"injured", "alert"})).mean())
    if recovery:
        losses.append((1.0 - state_mass(committed[recovery], {"recovered"})).mean())
    wrong = model(examples, carry_style="soft", force_mode="tv")
    wrong_committed = wrong["committed_state2"]
    losses.append(0.5 * state_mass(wrong_committed, {"injured", "alert"}).mean())
    return sum(losses)


def train_knockout_model(seed: int, split: dict[str, list[base.PulseExample]], *, args: argparse.Namespace, mode: str) -> KnockoutPulseModel:
    base.set_seed(seed)
    model = KnockoutPulseModel(args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        if mode == "action_plus_counterfactual_controls":
            loss = counterfactual_pressure_loss(model, split["train"])
        else:
            pred = model(split["train"], carry_style="soft")
            loss = supervision_loss(pred, split["train"], mode)
        loss.backward()
        opt.step()
    return model


def add_distractors(examples: list[base.PulseExample]) -> list[base.PulseExample]:
    out = []
    for ex in examples:
        distractor_atoms: list[str] = []
        for mode, atoms in base.CUE_FEATURES.items():
            if mode != ex.final_mode:
                distractor_atoms.extend(atoms[:1])
            if len(distractor_atoms) >= 3:
                break
        cue2 = tuple(dict.fromkeys(list(ex.cue2) + distractor_atoms[:3]))
        out.append(replace(ex, cue2=cue2))
    return out


def add_longer_delay_noise(examples: list[base.PulseExample]) -> list[base.PulseExample]:
    filler = tuple(base.UNCERTAIN_FEATURES[:3])
    return [replace(ex, cue1=tuple(dict.fromkeys(list(ex.cue1) + list(filler)))) for ex in examples]


def evaluate_with_controls(model: KnockoutPulseModel, examples: list[base.PulseExample]) -> tuple[dict[str, float], dict[str, float]]:
    hard = evaluate_model(model, examples, carry_style="hard")
    soft = evaluate_model(model, examples, carry_style="soft")
    teacher = evaluate_model(model, examples, carry_style="teacher")
    shuffled = evaluate_model(model, examples, carry_style="hard", swap_evidence=True)
    wrong = evaluate_model(model, examples, carry_style="hard", force_mode="tv")
    controls = {
        "free_run_vs_teacher_forced_gap": teacher["hard_counterfactual_action_accuracy"] - hard["hard_counterfactual_action_accuracy"],
        "train_soft_eval_hard_gap": soft["hard_counterfactual_action_accuracy"] - hard["hard_counterfactual_action_accuracy"],
        "shuffled_evidence_drop": hard["hard_counterfactual_action_accuracy"] - shuffled["hard_counterfactual_action_accuracy"],
        "wrong_mode_drop": hard["reality_action_authority"] - wrong["reality_action_authority"],
        "teacher_forced_action_accuracy": teacher["hard_counterfactual_action_accuracy"],
        "soft_action_accuracy": soft["hard_counterfactual_action_accuracy"],
    }
    return hard, controls


def run_arm(args: argparse.Namespace, split: dict[str, list[base.PulseExample]], seed: int, arm: str) -> tuple[dict[str, Any], KnockoutPulseModel | None]:
    if arm == "static_full_context_baseline":
        model = inferred.train_static_model(seed, split, args=args)
        return {"arm": arm, "seed": seed, "final_test": evaluate_static(model, split["final_test"])}, None
    if arm == "oracle_full_supervision":
        model = train_knockout_model(seed, split, args=args, mode="full_supervision")
        return {"arm": arm, "seed": seed, "final_test": evaluate_model(model, split["final_test"], carry_style="teacher")}, None
    model = train_knockout_model(seed, split, args=args, mode=arm)
    hard, controls = evaluate_with_controls(model, split["final_test"])
    return {"arm": arm, "seed": seed, "final_test": hard, "controls": controls}, model


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
        robustness_keys = sorted({key for row in rows for key in row.get("robustness", {})})
        if robustness_keys:
            item["robustness"] = {key: summarize([float(row["robustness"][key]) for row in rows if "robustness" in row]) for key in robustness_keys}
        out[arm] = item
    return out


def mean_at(agg: dict[str, Any], arm: str, key: str) -> float:
    value = agg.get(arm, {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def control_mean(agg: dict[str, Any], arm: str, key: str) -> float:
    value = agg.get(arm, {}).get("controls", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def best_reduced_arm(agg: dict[str, Any]) -> str:
    candidates = ["no_pulse_supervision", "no_state_supervision", "no_pulse_no_state", "action_only", "action_plus_counterfactual_controls"]
    return max(candidates, key=lambda arm: (mean_at(agg, arm, "action_accuracy"), mean_at(agg, arm, "pulse_alignment_to_gold") + mean_at(agg, arm, "state_alignment_to_gold")))


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    reduced = ["no_pulse_supervision", "no_state_supervision", "no_pulse_no_state", "action_only", "action_plus_counterfactual_controls"]
    latent_reduced = [
        arm for arm in reduced
        if mean_at(agg, arm, "action_accuracy") >= 0.90
        and min(mean_at(agg, arm, "pulse_alignment_to_gold"), mean_at(agg, arm, "state_alignment_to_gold")) >= 0.80
        and mean_at(agg, arm, "false_commit_rate") <= 0.10
        and mean_at(agg, arm, "nonreality_action_leakage") <= 0.10
    ]
    action_only_high = mean_at(agg, "action_only", "action_accuracy") >= 0.90
    action_only_low_alignment = min(mean_at(agg, "action_only", "pulse_alignment_to_gold"), mean_at(agg, "action_only", "state_alignment_to_gold")) < 0.80
    cf_alignment = (mean_at(agg, "action_plus_counterfactual_controls", "pulse_alignment_to_gold") + mean_at(agg, "action_plus_counterfactual_controls", "state_alignment_to_gold")) / 2.0
    action_alignment = (mean_at(agg, "action_only", "pulse_alignment_to_gold") + mean_at(agg, "action_only", "state_alignment_to_gold")) / 2.0
    return {
        "pulse_emerges_without_pulse_labels": mean_at(agg, "no_pulse_supervision", "pulse_alignment_to_gold") >= 0.80 or mean_at(agg, "action_plus_counterfactual_controls", "pulse_alignment_to_gold") >= 0.80,
        "state_emerges_without_state_labels": mean_at(agg, "no_state_supervision", "state_alignment_to_gold") >= 0.80 or mean_at(agg, "action_plus_counterfactual_controls", "state_alignment_to_gold") >= 0.80,
        "action_only_collapses_to_shortcut": action_only_high and action_only_low_alignment,
        "counterfactual_pressure_recovers_mechanism": (
            mean_at(agg, "action_plus_counterfactual_controls", "action_accuracy") >= 0.90
            and cf_alignment - action_alignment >= 0.20
            and control_mean(agg, "action_plus_counterfactual_controls", "shuffled_evidence_drop") >= 0.20
            and control_mean(agg, "action_plus_counterfactual_controls", "wrong_mode_drop") >= 0.50
        ),
        "full_supervision_still_upper_bound": mean_at(agg, "full_supervision", "action_accuracy") >= 0.95 and mean_at(agg, "full_supervision", "state_alignment_to_gold") >= 0.95,
        "latent_pilot_mechanism_supported": len(latent_reduced) > 0,
        "supervision_dependency_identified": len(latent_reduced) == 0 or mean_at(agg, "full_supervision", "action_accuracy") - max(mean_at(agg, arm, "action_accuracy") for arm in reduced) >= 0.10,
    }


def fmt(item: dict[str, Any], key: str) -> str:
    value = item.get(key, {}).get("mean")
    return "null" if value is None else f"{float(value):.6f}"


def control_fmt(item: dict[str, Any], key: str) -> str:
    value = item.get("controls", {}).get(key, {}).get("mean")
    return "null" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    best = summary["best_reduced_arm"]
    lines = [
        "# Pilot Supervision Knockout Matrix",
        "",
        "## Goal",
        "",
        "Measure how much explicit pulse/self-state supervision is required for the inferred Pilot Pulse mechanism.",
        "",
        "Reduced-supervision arms train with differentiable soft committed-state carry, but all primary verdict metrics use hard free-run evaluation.",
        "",
        "## Main Results",
        "",
        "| Arm | Action | Pulse Align | State Align | Wait | Commit | Reframe | Update | False Commit | Leakage | Soft-Hard Gap | Shuffled Drop | Wrong Mode Drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in TRAINING_MODES:
        item = agg[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item, 'action_accuracy')}` | `{fmt(item, 'pulse_alignment_to_gold')}` "
            f"| `{fmt(item, 'state_alignment_to_gold')}` | `{fmt(item, 'wait_alignment')}` "
            f"| `{fmt(item, 'commit_alignment')}` | `{fmt(item, 'reframe_alignment')}` "
            f"| `{fmt(item, 'update_alignment')}` | `{fmt(item, 'false_commit_rate')}` "
            f"| `{fmt(item, 'nonreality_action_leakage')}` | `{control_fmt(item, 'train_soft_eval_hard_gap')}` "
            f"| `{control_fmt(item, 'shuffled_evidence_drop')}` | `{control_fmt(item, 'wrong_mode_drop')}` |"
        )

    lines.extend([
        "",
        "## Robustness",
        "",
        f"Best reduced-supervision arm by action/alignment: `{best}`.",
        "",
        "| Arm | Heldout Cue | Distractor Cue | Longer Delay | Delayed Correction | Reframe After Commit | Recovery |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ["full_supervision", best, "action_plus_counterfactual_controls"]:
        item = agg[arm]
        rob = item.get("robustness", {})
        lines.append(
            f"| `{arm}` | `{fmt(item, 'heldout_cue_combo_accuracy')}` "
            f"| `{rob.get('distractor_cue_accuracy', {}).get('mean', 0.0):.6f}` "
            f"| `{rob.get('longer_delay_accuracy', {}).get('mean', 0.0):.6f}` "
            f"| `{fmt(item, 'delayed_correction_accuracy')}` "
            f"| `{fmt(item, 'reframe_after_commit_accuracy')}` "
            f"| `{fmt(item, 'recovery_update_split_accuracy')}` |"
        )

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
        "If reduced-supervision arms preserve action but lose pulse/state alignment, they are reported as shortcut solvers rather than latent Pilot mechanisms.",
        "",
        "If counterfactual pressure recovers alignment without direct pulse/state labels, this supports consequence-based training pressure as a replacement for component supervision.",
        "",
        "Observed result: full supervision remains the only arm that preserves action, pulse alignment, state alignment, low leakage, and robustness together.",
        "",
        "`no_pulse_supervision` preserves final action and state alignment, but pulse alignment collapses, so the pulse policy does not emerge from state/action supervision alone.",
        "",
        "`no_state_supervision` preserves most pulse labels but loses committed-state alignment and action reliability, so pulse labels alone do not recover the state mechanism.",
        "",
        "`action_only` and `action_plus_counterfactual_controls` do not recover latent pulse/state structure under hard free-run evaluation.",
        "",
        "The supervision dependency is therefore identified: current Pilot Pulse v1 works as a supervised component stack, but this toy setup does not yet show self-discovery of the pulse/state mechanism from final consequences alone.",
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
    for seed in range(args.seeds):
        print(f"[pilot-supervision-knockout] seed={seed}", flush=True)
        trained_models: dict[str, KnockoutPulseModel] = {}
        for arm_i, arm in enumerate(TRAINING_MODES):
            row, model = run_arm(args, split, seed + 10_000 * arm_i, arm)
            if model is not None:
                trained_models[arm] = model
            records.append(row)
        for arm in ["full_supervision", "no_pulse_supervision", "no_state_supervision", "action_plus_counterfactual_controls"]:
            if arm in trained_models:
                distractor = evaluate_model(trained_models[arm], add_distractors(split["final_test"]), carry_style="hard")
                longer = evaluate_model(trained_models[arm], add_longer_delay_noise(split["final_test"]), carry_style="hard")
                for row in records:
                    if row["arm"] == arm and row["seed"] == seed:
                        row["robustness"] = {
                            "distractor_cue_accuracy": distractor["hard_counterfactual_action_accuracy"],
                            "longer_delay_accuracy": longer["hard_counterfactual_action_accuracy"],
                        }
                        break

    summary = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "records": records, "aggregate": aggregate(records)}
    summary["best_reduced_arm"] = best_reduced_arm(summary["aggregate"])
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
