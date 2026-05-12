#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "pilot-pulse-commit-reframe"
REPORT_PATH = ROOT / "docs" / "research" / "PILOT_PULSE_COMMIT_REFRAME_PROBE.md"
SUMMARY_NAME = "pilot_pulse_commit_reframe_summary.json"

ACTORS = ["dog", "cat", "snake"]
EVENT_ACTIONS = ["bit", "chased", "scared", "barked"]
PATIENTS = ["me", "john", "child"]
MODES = ["reality", "tv", "game", "dream", "memory", "uncertain"]
SELF_STATES = ["safe", "uncertain", "alert", "injured", "other_help", "nonreal_observe", "recovered"]
CONTROLLER_ACTIONS = [
    "wait_for_more_info",
    "seek_help",
    "avoid",
    "help_other",
    "observe",
    "ignore_real_action",
    "continue",
    "monitor",
]
PULSE_COMMANDS = ["wait", "commit", "reframe", "update", "hold"]
PULSE_PHASES = ["ambiguity_phase", "commit_phase", "reframe_phase", "update_phase", "hold_phase"]
THREAT_ACTIONS = {"bit", "chased", "scared"}

CUE_FEATURES: dict[str, list[str]] = {
    "reality": ["now", "present", "physical", "in_room"],
    "tv": ["screen", "watched", "fictional", "episode"],
    "game": ["avatar", "level", "respawn", "controller"],
    "dream": ["asleep", "unreal", "nightmare", "remembered_after"],
    "memory": ["past", "remembered", "autobiographical", "yesterday"],
}
RECOVERY_FEATURES = ["cleaned_wound", "doctor_helped", "safe_now", "bandaged"]
UNCERTAIN_FEATURES = ["ambiguous", "unclear", "maybe_real", "missing_context"]
CUE_ATOMS = [atom for mode in CUE_FEATURES for atom in CUE_FEATURES[mode]] + RECOVERY_FEATURES + UNCERTAIN_FEATURES


@dataclass(frozen=True)
class PulseExample:
    family: str
    actor: str
    action: str
    patient: str
    initial_mode: str
    final_mode: str
    cue1: tuple[str, ...]
    cue2: tuple[str, ...]
    phase1: str
    phase2: str
    state1: str
    state2: str
    pulse1: str
    pulse2: str
    controller_action: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot Pulse commit/reframe stress test.")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cue_pairs(mode: str) -> list[tuple[str, str]]:
    return list(combinations(CUE_FEATURES[mode], 2))


def recovery_pairs() -> list[tuple[str, str]]:
    return list(combinations(RECOVERY_FEATURES, 2))


def split_bundles() -> dict[str, dict[str, list[tuple[str, ...]]]]:
    split: dict[str, dict[str, list[tuple[str, ...]]]] = {"train": {}, "validation": {}, "final_test": {}}
    for mode in CUE_FEATURES:
        pairs = cue_pairs(mode)
        split["train"][mode] = [tuple(item) for item in pairs[:4]]
        split["validation"][mode] = [tuple(pairs[4])]
        split["final_test"][mode] = [tuple(pairs[5])]
    rpairs = recovery_pairs()
    split["train"]["recovery"] = [tuple(item) for item in rpairs[:4]]
    split["validation"]["recovery"] = [tuple(rpairs[4])]
    split["final_test"]["recovery"] = [tuple(rpairs[5])]
    split["train"]["uncertain"] = [("ambiguous", "unclear"), ("maybe_real", "missing_context")]
    split["validation"]["uncertain"] = [("ambiguous", "maybe_real")]
    split["final_test"]["uncertain"] = [("unclear", "missing_context")]
    return split


def action_for_state(state: str) -> str:
    return {
        "safe": "continue",
        "uncertain": "wait_for_more_info",
        "alert": "avoid",
        "injured": "seek_help",
        "other_help": "help_other",
        "nonreal_observe": "observe",
        "recovered": "monitor",
    }[state]


def threat_state(action: str, patient: str) -> str:
    if patient == "me":
        if action == "bit":
            return "injured"
        if action in {"chased", "scared"}:
            return "alert"
    if action in THREAT_ACTIONS and patient != "me":
        return "other_help"
    return "safe"


def add_example(
    rows: list[PulseExample],
    *,
    family: str,
    actor: str,
    action: str,
    patient: str,
    initial_mode: str,
    final_mode: str,
    cue1: tuple[str, ...],
    cue2: tuple[str, ...],
    phase1: str,
    phase2: str,
    state1: str,
    state2: str,
    pulse1: str,
    pulse2: str,
    controller_action: str | None = None,
) -> None:
    rows.append(PulseExample(
        family=family,
        actor=actor,
        action=action,
        patient=patient,
        initial_mode=initial_mode,
        final_mode=final_mode,
        cue1=cue1,
        cue2=cue2,
        phase1=phase1,
        phase2=phase2,
        state1=state1,
        state2=state2,
        pulse1=pulse1,
        pulse2=pulse2,
        controller_action=controller_action or action_for_state(state2),
    ))


def make_split() -> dict[str, list[PulseExample]]:
    bundles = split_bundles()
    out: dict[str, list[PulseExample]] = {}
    for split_name, by_mode in bundles.items():
        rows: list[PulseExample] = []

        for actor, action, patient in [
            ("dog", "bit", "me"),
            ("snake", "scared", "me"),
            ("dog", "bit", "john"),
            ("cat", "chased", "child"),
            ("dog", "barked", "me"),
        ]:
            for reality in by_mode["reality"]:
                state = threat_state(action, patient)
                pulse = "commit" if state not in {"safe"} else "hold"
                add_example(
                    rows,
                    family="clean_commit",
                    actor=actor,
                    action=action,
                    patient=patient,
                    initial_mode="reality",
                    final_mode="reality",
                    cue1=reality,
                    cue2=reality,
                    phase1="commit_phase" if pulse == "commit" else "hold_phase",
                    phase2="hold_phase",
                    state1=state,
                    state2=state,
                    pulse1=pulse,
                    pulse2="hold",
                )

        for mode in ["tv", "game", "dream", "memory"]:
            for cue in by_mode[mode]:
                add_example(
                    rows,
                    family="clean_nonreal",
                    actor="dog",
                    action="bit",
                    patient="me",
                    initial_mode=mode,
                    final_mode=mode,
                    cue1=cue,
                    cue2=cue,
                    phase1="hold_phase",
                    phase2="hold_phase",
                    state1="nonreal_observe",
                    state2="nonreal_observe",
                    pulse1="hold",
                    pulse2="hold",
                )

        for mode in ["tv", "game", "dream", "memory"]:
            for uncertain in by_mode["uncertain"]:
                for cue in by_mode[mode]:
                    add_example(
                        rows,
                        family="delayed_correction",
                        actor="dog",
                        action="bit",
                        patient="me",
                        initial_mode="uncertain",
                        final_mode=mode,
                        cue1=uncertain,
                        cue2=cue,
                        phase1="ambiguity_phase",
                        phase2="reframe_phase",
                        state1="uncertain",
                        state2="nonreal_observe",
                        pulse1="wait",
                        pulse2="reframe",
                    )

        for actor, action, patient, state in [
            ("dog", "bit", "me", "injured"),
            ("snake", "scared", "me", "alert"),
        ]:
            for uncertain in by_mode["uncertain"]:
                for reality in by_mode["reality"]:
                    add_example(
                        rows,
                        family="delayed_confirmation",
                        actor=actor,
                        action=action,
                        patient=patient,
                        initial_mode="uncertain",
                        final_mode="reality",
                        cue1=uncertain,
                        cue2=reality,
                        phase1="ambiguity_phase",
                        phase2="commit_phase",
                        state1="uncertain",
                        state2=state,
                        pulse1="wait",
                        pulse2="commit",
                    )

        for mode in ["tv", "game", "dream", "memory"]:
            for reality in by_mode["reality"]:
                for cue in by_mode[mode]:
                    add_example(
                        rows,
                        family="reframe_after_commit",
                        actor="dog",
                        action="bit",
                        patient="me",
                        initial_mode="reality",
                        final_mode=mode,
                        cue1=reality,
                        cue2=cue,
                        phase1="commit_phase",
                        phase2="reframe_phase",
                        state1="injured",
                        state2="nonreal_observe",
                        pulse1="commit",
                        pulse2="reframe",
                    )

        for reality in by_mode["reality"]:
            for recovery in by_mode["recovery"]:
                add_example(
                    rows,
                    family="recovery_update",
                    actor="dog",
                    action="bit",
                    patient="me",
                    initial_mode="reality",
                    final_mode="reality",
                    cue1=reality,
                    cue2=recovery,
                    phase1="commit_phase",
                    phase2="update_phase",
                    state1="injured",
                    state2="recovered",
                    pulse1="commit",
                    pulse2="update",
                )

        out[split_name] = rows
    return out


def input_dim() -> int:
    return (
        len(ACTORS)
        + len(EVENT_ACTIONS)
        + len(PATIENTS)
        + len(CUE_ATOMS)
        + len(PULSE_PHASES)
        + len(SELF_STATES)
    )


def action_dim() -> int:
    return 1 + len(SELF_STATES)


def encode_evidence(
    examples: list[PulseExample],
    *,
    step: int,
    previous_state: torch.Tensor | None = None,
    ablate_phase: bool = False,
    force_phase: str | None = None,
    shuffled_phases: list[str] | None = None,
    force_mode: str | None = None,
) -> torch.Tensor:
    rows = torch.zeros(len(examples), input_dim())
    actor_offset = 0
    action_offset = actor_offset + len(ACTORS)
    patient_offset = action_offset + len(EVENT_ACTIONS)
    cue_offset = patient_offset + len(PATIENTS)
    phase_offset = cue_offset + len(CUE_ATOMS)
    state_offset = phase_offset + len(PULSE_PHASES)
    for i, ex in enumerate(examples):
        rows[i, actor_offset + ACTORS.index(ex.actor)] = 1.0
        rows[i, action_offset + EVENT_ACTIONS.index(ex.action)] = 1.0
        rows[i, patient_offset + PATIENTS.index(ex.patient)] = 1.0
        if step == 1:
            cue = ex.cue1
            phase = ex.phase1
        else:
            if force_mode is not None and force_mode in CUE_FEATURES:
                cue = cue_pairs(force_mode)[0]
            else:
                cue = ex.cue2
            phase = ex.phase2
        for atom in cue:
            rows[i, cue_offset + CUE_ATOMS.index(atom)] = 1.0
        if not ablate_phase:
            phase_value = force_phase or phase
            if shuffled_phases is not None and step == 2:
                phase_value = shuffled_phases[i]
            rows[i, phase_offset + PULSE_PHASES.index(phase_value)] = 1.0
        if step == 1:
            rows[i, state_offset + SELF_STATES.index("safe")] = 1.0
        elif previous_state is not None:
            rows[i, state_offset:state_offset + len(SELF_STATES)] = previous_state[i]
        else:
            rows[i, state_offset + SELF_STATES.index("safe")] = 1.0
    return rows


def encode_action(state_probs: torch.Tensor | None, n: int) -> torch.Tensor:
    rows = torch.zeros(n, action_dim())
    rows[:, 0] = 1.0
    if state_probs is not None:
        rows[:, 1:] = state_probs
    return rows


def labels(examples: list[PulseExample]) -> dict[str, torch.Tensor]:
    semantic = []
    mode1 = []
    mode2 = []
    authority1 = []
    authority2 = []
    state1 = []
    state2 = []
    pulse1 = []
    pulse2 = []
    action = []
    for ex in examples:
        threat = ex.action in THREAT_ACTIONS
        semantic.append([float(ex.action == "bit"), float(threat)])
        mode1.append(MODES.index(ex.initial_mode))
        mode2.append(MODES.index(ex.final_mode))
        real1 = float(ex.initial_mode == "reality" and threat and ex.patient == "me")
        real2 = float(ex.final_mode == "reality" and ex.state2 in {"injured", "alert"})
        self1 = float(ex.initial_mode == "reality" and ex.patient == "me")
        self2 = float(ex.final_mode == "reality" and ex.patient == "me")
        other2 = float(ex.final_mode == "reality" and ex.state2 == "other_help")
        authority1.append([real1, self1, other2])
        authority2.append([real2, self2, other2])
        state1.append(SELF_STATES.index(ex.state1))
        state2.append(SELF_STATES.index(ex.state2))
        pulse1.append(PULSE_COMMANDS.index(ex.pulse1))
        pulse2.append(PULSE_COMMANDS.index(ex.pulse2))
        action.append(CONTROLLER_ACTIONS.index(ex.controller_action))
    return {
        "semantic": torch.tensor(semantic, dtype=torch.float32),
        "mode1": torch.tensor(mode1, dtype=torch.long),
        "mode2": torch.tensor(mode2, dtype=torch.long),
        "authority1": torch.tensor(authority1, dtype=torch.float32),
        "authority2": torch.tensor(authority2, dtype=torch.float32),
        "state1": torch.tensor(state1, dtype=torch.long),
        "state2": torch.tensor(state2, dtype=torch.long),
        "pulse1": torch.tensor(pulse1, dtype=torch.long),
        "pulse2": torch.tensor(pulse2, dtype=torch.long),
        "action": torch.tensor(action, dtype=torch.long),
    }


class PulseModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim(), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.semantic = nn.Linear(hidden, 2)
        self.mode = nn.Linear(hidden, len(MODES))
        self.authority = nn.Linear(hidden, 3)
        self.pulse = nn.Linear(hidden, len(PULSE_COMMANDS))
        self.state = nn.Linear(hidden, len(SELF_STATES))
        self.action_trunk = nn.Sequential(nn.Linear(action_dim(), hidden), nn.Tanh(), nn.Linear(hidden, len(CONTROLLER_ACTIONS)))

    def evidence(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "semantic": self.semantic(h),
            "mode": self.mode(h),
            "authority": self.authority(h),
            "pulse": self.pulse(h),
            "state": self.state(h),
        }

    @staticmethod
    def pulse_gated_state(pulse_logits: torch.Tensor, state_logits: torch.Tensor, previous_state: torch.Tensor) -> torch.Tensor:
        proposed = F.one_hot(state_logits.argmax(dim=-1), num_classes=len(SELF_STATES)).float()
        pulse = pulse_logits.argmax(dim=-1)
        out = proposed.clone()
        wait_idx = PULSE_COMMANDS.index("wait")
        hold_idx = PULSE_COMMANDS.index("hold")
        uncertain_idx = SELF_STATES.index("uncertain")
        wait_mask = pulse == wait_idx
        hold_mask = pulse == hold_idx
        if wait_mask.any():
            out[wait_mask] = 0.0
            out[wait_mask, uncertain_idx] = 1.0
        if hold_mask.any():
            out[hold_mask] = previous_state[hold_mask]
        return out

    def forward(
        self,
        examples: list[PulseExample],
        *,
        carry_mode: str,
        ablate_phase: bool = False,
        force_phase: str | None = None,
        shuffled_phases: list[str] | None = None,
        force_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        x1 = encode_evidence(examples, step=1, ablate_phase=ablate_phase, force_phase=force_phase)
        step1 = self.evidence(x1)
        if carry_mode == "recursive":
            state1 = F.one_hot(step1["state"].argmax(dim=-1), num_classes=len(SELF_STATES)).float()
        elif carry_mode == "oracle":
            state1 = F.one_hot(labels(examples)["state1"], num_classes=len(SELF_STATES)).float()
        elif carry_mode == "no_commit":
            state1 = None
        else:
            raise ValueError(carry_mode)
        x2 = encode_evidence(
            examples,
            step=2,
            previous_state=state1,
            ablate_phase=ablate_phase,
            force_phase=force_phase,
            shuffled_phases=shuffled_phases,
            force_mode=force_mode,
        )
        step2 = self.evidence(x2)
        if carry_mode == "recursive":
            assert state1 is not None
            state2 = self.pulse_gated_state(step2["pulse"], step2["state"], state1)
        elif carry_mode == "oracle":
            state2 = F.one_hot(labels(examples)["state2"], num_classes=len(SELF_STATES)).float()
        else:
            state2 = None
        action = self.action_trunk(encode_action(state2, len(examples)))
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
            "committed_state2": state2 if state2 is not None else torch.zeros(len(examples), len(SELF_STATES)),
            "action": action,
        }


class StaticFullContextModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * input_dim(), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.semantic = nn.Linear(hidden, 2)
        self.mode = nn.Linear(hidden, len(MODES))
        self.authority = nn.Linear(hidden, 3)
        self.pulse = nn.Linear(hidden, len(PULSE_COMMANDS))
        self.state = nn.Linear(hidden, len(SELF_STATES))
        self.action = nn.Linear(hidden, len(CONTROLLER_ACTIONS))

    def forward(self, examples: list[PulseExample]) -> dict[str, torch.Tensor]:
        x1 = encode_evidence(examples, step=1)
        x2 = encode_evidence(examples, step=2, previous_state=F.one_hot(labels(examples)["state1"], len(SELF_STATES)).float())
        h = self.net(torch.cat([x1, x2], dim=-1))
        return {
            "semantic2": self.semantic(h),
            "mode2": self.mode(h),
            "authority2": self.authority(h),
            "pulse2": self.pulse(h),
            "state2": self.state(h),
            "action": self.action(h),
        }


def pulse_loss(model: PulseModel, examples: list[PulseExample], *, carry_mode: str, ablate_phase: bool = False) -> torch.Tensor:
    pred = model(examples, carry_mode=carry_mode, ablate_phase=ablate_phase)
    tgt = labels(examples)
    oracle_action = model.action_trunk(
        encode_action(F.one_hot(tgt["state2"], num_classes=len(SELF_STATES)).float(), len(examples))
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


def static_loss(model: StaticFullContextModel, examples: list[PulseExample]) -> torch.Tensor:
    pred = model(examples)
    tgt = labels(examples)
    return (
        F.binary_cross_entropy_with_logits(pred["semantic2"], tgt["semantic"])
        + F.cross_entropy(pred["mode2"], tgt["mode2"])
        + F.binary_cross_entropy_with_logits(pred["authority2"], tgt["authority2"])
        + F.cross_entropy(pred["pulse2"], tgt["pulse2"])
        + F.cross_entropy(pred["state2"], tgt["state2"])
        + F.cross_entropy(pred["action"], tgt["action"])
    )


def train_pulse_model(seed: int, split: dict[str, list[PulseExample]], *, args: argparse.Namespace, carry_mode: str, ablate_phase: bool = False) -> PulseModel:
    set_seed(seed)
    model = PulseModel(args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = pulse_loss(model, split["train"], carry_mode=carry_mode, ablate_phase=ablate_phase)
        loss.backward()
        opt.step()
    return model


def train_static_model(seed: int, split: dict[str, list[PulseExample]], *, args: argparse.Namespace) -> StaticFullContextModel:
    set_seed(seed)
    model = StaticFullContextModel(args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = static_loss(model, split["train"])
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def binary_acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(((torch.sigmoid(logits) >= 0.5) == (target >= 0.5)).float().mean().item())


def family_indices(examples: list[PulseExample], families: set[str]) -> list[int]:
    return [i for i, ex in enumerate(examples) if ex.family in families]


def acc_at(pred: torch.Tensor, target: torch.Tensor, idx: list[int]) -> float:
    if not idx:
        return 0.0
    return float((pred[idx] == target[idx]).float().mean().item())


@torch.no_grad()
def metrics_from_predictions(
    examples: list[PulseExample],
    *,
    semantic_logits: torch.Tensor,
    mode_logits: torch.Tensor,
    authority_logits: torch.Tensor,
    pulse_logits: torch.Tensor,
    state_logits: torch.Tensor,
    action_logits: torch.Tensor,
    authority1_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    tgt = labels(examples)
    pulse_pred = pulse_logits.argmax(dim=-1)
    state_pred = state_logits.argmax(dim=-1)
    action_pred = action_logits.argmax(dim=-1)
    semantic = binary_acc(semantic_logits, tgt["semantic"])
    mode = float((mode_logits.argmax(dim=-1) == tgt["mode2"]).float().mean().item())
    state = float((state_pred == tgt["state2"]).float().mean().item())
    action = float((action_pred == tgt["action"]).float().mean().item())
    pulse = float((pulse_pred == tgt["pulse2"]).float().mean().item())
    ambiguous = family_indices(examples, {"delayed_correction", "delayed_confirmation"})
    reframe = family_indices(examples, {"delayed_correction", "reframe_after_commit"})
    recovery = family_indices(examples, {"recovery_update"})
    delayed_correction = family_indices(examples, {"delayed_correction"})
    reframe_after = family_indices(examples, {"reframe_after_commit"})
    nonreal = [i for i, ex in enumerate(examples) if ex.final_mode in {"tv", "game", "dream", "memory"} and ex.patient == "me" and ex.action in THREAT_ACTIONS]
    reality_self = [i for i, ex in enumerate(examples) if ex.final_mode == "reality" and ex.patient == "me" and ex.state2 in {"injured", "alert"}]
    reality_other = [i for i, ex in enumerate(examples) if ex.final_mode == "reality" and ex.state2 == "other_help"]
    predicted_commit = torch.isin(state_pred, torch.tensor([SELF_STATES.index("injured"), SELF_STATES.index("alert")]))
    false_commit_rate = float(predicted_commit[nonreal].float().mean().item()) if nonreal else 0.0
    missed_commit_rate = float((~predicted_commit[reality_self]).float().mean().item()) if reality_self else 0.0
    authority_prob = torch.sigmoid(authority_logits)
    reality_action = float(authority_prob[reality_self, 0].mean().item()) if reality_self else 0.0
    nonreal_leakage = float(authority_prob[nonreal, 0].mean().item()) if nonreal else 0.0
    self_anchor_gain = 0.0
    if reality_self and reality_other:
        self_anchor_gain = float(authority_prob[reality_self, 0].mean().item() - authority_prob[reality_other, 0].mean().item())
    reframe_drop = 0.0
    if authority1_logits is not None and reframe_after:
        reframe_drop = float(torch.sigmoid(authority1_logits)[reframe_after, 0].mean().item() - authority_prob[reframe_after, 0].mean().item())
    return {
        "semantic_accuracy": semantic,
        "grounding_mode_accuracy": mode,
        "committed_state_accuracy": state,
        "action_accuracy": action,
        "hard_counterfactual_action_accuracy": action,
        "commit_timing_accuracy": pulse,
        "wait_when_ambiguous_accuracy": acc_at(pulse_pred, tgt["pulse2"], ambiguous),
        "reframe_accuracy": acc_at(pulse_pred, tgt["pulse2"], reframe),
        "recovery_update_accuracy": acc_at(pulse_pred, tgt["pulse2"], recovery),
        "false_commit_rate": false_commit_rate,
        "missed_commit_rate": missed_commit_rate,
        "reality_action_authority": reality_action,
        "nonreality_action_leakage": nonreal_leakage,
        "self_anchor_gain": self_anchor_gain,
        "reframe_authority_drop": reframe_drop,
        "delayed_correction_action_accuracy": acc_at(action_pred, tgt["action"], delayed_correction),
        "reframe_case_action_accuracy": acc_at(action_pred, tgt["action"], reframe_after),
        "nonreal_action_accuracy": acc_at(action_pred, tgt["action"], nonreal),
        "recovery_case_action_accuracy": acc_at(action_pred, tgt["action"], recovery),
    }


@torch.no_grad()
def evaluate_pulse(
    model: PulseModel,
    examples: list[PulseExample],
    *,
    carry_mode: str,
    ablate_phase: bool = False,
    force_phase: str | None = None,
    shuffled_phase: bool = False,
    force_mode: str | None = None,
) -> dict[str, float]:
    shuffled_phases = None
    if shuffled_phase:
        phase_map = {
            "ambiguity_phase": "hold_phase",
            "commit_phase": "ambiguity_phase",
            "reframe_phase": "commit_phase",
            "update_phase": "hold_phase",
            "hold_phase": "reframe_phase",
        }
        shuffled_phases = [phase_map[ex.phase2] for ex in examples]
    pred = model(
        examples,
        carry_mode=carry_mode,
        ablate_phase=ablate_phase,
        force_phase=force_phase,
        shuffled_phases=shuffled_phases,
        force_mode=force_mode,
    )
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
def evaluate_static(model: StaticFullContextModel, examples: list[PulseExample]) -> dict[str, float]:
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


def deterministic_predictions(examples: list[PulseExample], arm: str) -> dict[str, torch.Tensor]:
    tgt = labels(examples)
    state = []
    pulse = []
    for ex in examples:
        if arm == "always_commit_baseline":
            if ex.action in THREAT_ACTIONS and ex.patient == "me":
                predicted = "injured" if ex.action == "bit" else "alert"
                command = "commit"
            elif ex.action in THREAT_ACTIONS and ex.patient != "me":
                predicted = "other_help"
                command = "commit"
            else:
                predicted = "safe"
                command = "hold"
        elif arm == "never_reframe_baseline":
            if ex.initial_mode == "reality" and ex.state1 in {"injured", "alert", "other_help"}:
                predicted = ex.state1
                command = "hold"
            else:
                predicted = ex.state2 if ex.pulse2 != "update" else ex.state1
                command = ex.pulse2 if ex.pulse2 != "update" else "hold"
        else:
            raise ValueError(arm)
        state.append(SELF_STATES.index(predicted))
        pulse.append(PULSE_COMMANDS.index(command))
    n = len(examples)
    semantic_logits = torch.where(tgt["semantic"] > 0.5, torch.full_like(tgt["semantic"], 6.0), torch.full_like(tgt["semantic"], -6.0))
    mode_logits = F.one_hot(tgt["mode2"], len(MODES)).float() * 12.0 - 6.0
    state_t = torch.tensor(state, dtype=torch.long)
    pulse_t = torch.tensor(pulse, dtype=torch.long)
    state_logits = F.one_hot(state_t, len(SELF_STATES)).float() * 12.0 - 6.0
    pulse_logits = F.one_hot(pulse_t, len(PULSE_COMMANDS)).float() * 12.0 - 6.0
    action_idx = torch.tensor([CONTROLLER_ACTIONS.index(action_for_state(SELF_STATES[int(s)])) for s in state_t], dtype=torch.long)
    action_logits = F.one_hot(action_idx, len(CONTROLLER_ACTIONS)).float() * 12.0 - 6.0
    authority = torch.zeros(n, 3)
    for i, s in enumerate(state_t):
        state_name = SELF_STATES[int(s)]
        authority[i, 0] = float(state_name in {"injured", "alert"})
        authority[i, 1] = float(examples[i].patient == "me")
        authority[i, 2] = float(state_name == "other_help")
    authority_logits = authority * 12.0 - 6.0
    return {
        "semantic": semantic_logits,
        "mode": mode_logits,
        "authority": authority_logits,
        "pulse": pulse_logits,
        "state": state_logits,
        "action": action_logits,
    }


def evaluate_deterministic(examples: list[PulseExample], arm: str) -> dict[str, float]:
    pred = deterministic_predictions(examples, arm)
    return metrics_from_predictions(
        examples,
        semantic_logits=pred["semantic"],
        mode_logits=pred["mode"],
        authority_logits=pred["authority"],
        pulse_logits=pred["pulse"],
        state_logits=pred["state"],
        action_logits=pred["action"],
    )


def run_arm(args: argparse.Namespace, split: dict[str, list[PulseExample]], seed: int, arm: str) -> tuple[dict[str, Any], PulseModel | None]:
    if arm == "pilot_pulse_recursive_model":
        model = train_pulse_model(seed, split, args=args, carry_mode="recursive")
        return {"arm": arm, "seed": seed, "final_test": evaluate_pulse(model, split["final_test"], carry_mode="recursive")}, model
    if arm == "no_pulse_baseline":
        model = train_pulse_model(seed, split, args=args, carry_mode="no_commit", ablate_phase=True)
        return {"arm": arm, "seed": seed, "final_test": evaluate_pulse(model, split["final_test"], carry_mode="no_commit", ablate_phase=True)}, None
    if arm == "oracle_state_upper_bound":
        model = train_pulse_model(seed, split, args=args, carry_mode="oracle")
        return {"arm": arm, "seed": seed, "final_test": evaluate_pulse(model, split["final_test"], carry_mode="oracle")}, None
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
    base_rows = [row for row in records if row["arm"] == "pilot_pulse_recursive_model"]
    for other, key in [
        ("no_pulse_baseline", "recursive_gap_vs_no_pulse"),
        ("always_commit_baseline", "recursive_gap_vs_always_commit"),
        ("never_reframe_baseline", "recursive_gap_vs_never_reframe"),
        ("static_full_context_baseline", "recursive_gap_vs_static"),
    ]:
        other_rows = [row for row in records if row["arm"] == other]
        if base_rows and other_rows:
            out["pilot_pulse_recursive_model"][key] = summarize([
                float(b["final_test"]["hard_counterfactual_action_accuracy"]) - float(o["final_test"]["hard_counterfactual_action_accuracy"])
                for b, o in zip(base_rows, other_rows)
            ])
    return out


def mean_at(agg: dict[str, Any], arm: str, key: str) -> float:
    value = agg.get(arm, {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def control_mean(agg: dict[str, Any], key: str) -> float:
    value = agg.get("pilot_pulse_recursive_model", {}).get("controls", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def pilot_gap(agg: dict[str, Any], key: str) -> float:
    value = agg.get("pilot_pulse_recursive_model", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    pilot = "pilot_pulse_recursive_model"
    static_action = mean_at(agg, "static_full_context_baseline", "action_accuracy")
    return {
        "supports_pilot_pulse_commit_timing": mean_at(agg, pilot, "commit_timing_accuracy") >= 0.90,
        "supports_wait_under_ambiguity": mean_at(agg, pilot, "wait_when_ambiguous_accuracy") >= 0.90,
        "supports_reframe_after_delayed_correction": mean_at(agg, pilot, "reframe_accuracy") >= 0.85 and mean_at(agg, "never_reframe_baseline", "reframe_case_action_accuracy") <= 0.60,
        "supports_reality_confirmation_commit": mean_at(agg, pilot, "missed_commit_rate") <= 0.10,
        "supports_committed_state_update": mean_at(agg, pilot, "recovery_update_accuracy") >= 0.85 and mean_at(agg, pilot, "recovery_case_action_accuracy") >= 0.85,
        "supports_pulse_causal_control": control_mean(agg, "shuffled_pulse_drop") >= 0.20 and control_mean(agg, "wrong_phase_drop") >= 0.05,
        "static_baseline_shortcut_detected": static_action >= 0.95 and pilot_gap(agg, "recursive_gap_vs_static") <= 0.05,
        "full_pilot_v0_strengthened": (
            mean_at(agg, pilot, "action_accuracy") >= 0.95
            and mean_at(agg, pilot, "committed_state_accuracy") >= 0.95
            and mean_at(agg, pilot, "commit_timing_accuracy") >= 0.90
            and mean_at(agg, pilot, "reframe_accuracy") >= 0.85
            and mean_at(agg, pilot, "recovery_update_accuracy") >= 0.85
            and mean_at(agg, pilot, "nonreality_action_leakage") <= 0.10
            and mean_at(agg, pilot, "false_commit_rate") <= 0.10
            and control_mean(agg, "wrong_mode_drop") >= 0.50
            and mean_at(agg, "always_commit_baseline", "nonreal_action_accuracy") <= 0.60
            and mean_at(agg, "never_reframe_baseline", "reframe_case_action_accuracy") <= 0.60
        ),
    }


def fmt(item: dict[str, Any], key: str) -> str:
    value = item.get(key, {}).get("mean")
    return "null" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Pilot Pulse Commit-Reframe Probe",
        "",
        "## Goal",
        "",
        "Stress-test whether a toy Pilot can decide when to wait, commit, reframe, update, and act after delayed grounding evidence.",
        "",
        "The final action step is generic and receives only a hard one-hot committed state, not event/mode/patient fields or a soft hidden state.",
        "",
        "## Main Results",
        "",
        "| Arm | Semantic | Mode | State | Action | Pulse | Wait | Reframe | Update | False Commit | Leakage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in [
        "pilot_pulse_recursive_model",
        "no_pulse_baseline",
        "always_commit_baseline",
        "never_reframe_baseline",
        "static_full_context_baseline",
        "oracle_state_upper_bound",
        "shuffled_pulse_control",
        "wrong_grounding_control",
    ]:
        item = agg[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item, 'semantic_accuracy')}` | `{fmt(item, 'grounding_mode_accuracy')}` "
            f"| `{fmt(item, 'committed_state_accuracy')}` | `{fmt(item, 'action_accuracy')}` "
            f"| `{fmt(item, 'commit_timing_accuracy')}` | `{fmt(item, 'wait_when_ambiguous_accuracy')}` "
            f"| `{fmt(item, 'reframe_accuracy')}` | `{fmt(item, 'recovery_update_accuracy')}` "
            f"| `{fmt(item, 'false_commit_rate')}` | `{fmt(item, 'nonreality_action_leakage')}` |"
        )
    pilot = agg["pilot_pulse_recursive_model"]
    lines.extend([
        "",
        "## Gaps And Controls",
        "",
        f"- recursive_gap_vs_no_pulse: `{pilot['recursive_gap_vs_no_pulse']['mean']:.6f}`",
        f"- recursive_gap_vs_always_commit: `{pilot['recursive_gap_vs_always_commit']['mean']:.6f}`",
        f"- recursive_gap_vs_never_reframe: `{pilot['recursive_gap_vs_never_reframe']['mean']:.6f}`",
        f"- recursive_gap_vs_static: `{pilot['recursive_gap_vs_static']['mean']:.6f}`",
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
        "Positive readout requires correct labels and pulse-specific behavior: waiting under ambiguity, reframing delayed nonreal corrections, committing delayed reality confirmations, updating recovered state, and failing the always-commit / never-reframe controls.",
        "",
        "If the static full-context baseline solves labels, this is reported as a shortcut baseline rather than a recursive Pilot mechanism.",
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
    split = make_split()
    records: list[dict[str, Any]] = []
    base_arms = [
        "pilot_pulse_recursive_model",
        "no_pulse_baseline",
        "always_commit_baseline",
        "never_reframe_baseline",
        "static_full_context_baseline",
        "oracle_state_upper_bound",
    ]
    for seed in range(args.seeds):
        print(f"[pilot-pulse] seed={seed}", flush=True)
        pilot_model: PulseModel | None = None
        for arm_i, arm in enumerate(base_arms):
            row, model = run_arm(args, split, seed + 10_000 * arm_i, arm)
            records.append(row)
            if arm == "pilot_pulse_recursive_model":
                pilot_model = model
        if pilot_model is not None:
            base = evaluate_pulse(pilot_model, split["final_test"], carry_mode="recursive")
            shuffled = evaluate_pulse(pilot_model, split["final_test"], carry_mode="recursive", shuffled_phase=True)
            wrong_phase = evaluate_pulse(pilot_model, split["final_test"], carry_mode="recursive", force_phase="hold_phase")
            wrong_mode = evaluate_pulse(pilot_model, split["final_test"], carry_mode="recursive", force_mode="tv")
            records.append({"arm": "shuffled_pulse_control", "seed": seed, "final_test": shuffled})
            records.append({"arm": "wrong_grounding_control", "seed": seed, "final_test": wrong_mode})
            records[-2]["controls"] = {}
            for row in records:
                if row["arm"] == "pilot_pulse_recursive_model" and row["seed"] == seed:
                    row["controls"] = {
                        "shuffled_pulse_drop": base["commit_timing_accuracy"] - shuffled["commit_timing_accuracy"],
                        "shuffled_pulse_action_drop": base["hard_counterfactual_action_accuracy"] - shuffled["hard_counterfactual_action_accuracy"],
                        "wrong_phase_drop": base["hard_counterfactual_action_accuracy"] - wrong_phase["hard_counterfactual_action_accuracy"],
                        "wrong_mode_drop": base["reality_action_authority"] - wrong_mode["reality_action_authority"],
                        "reframe_authority_drop": base["reframe_authority_drop"],
                    }
                    break
    summary = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "records": records, "aggregate": aggregate(records)}
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
