#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "grounding-mode-authority"
REPORT_PATH = ROOT / "docs" / "research" / "GROUNDING_MODE_AUTHORITY_PROBE.md"
SUMMARY_NAME = "grounding_mode_authority_summary.json"

ACTORS = ["dog", "cat", "snake"]
ACTIONS = ["bit", "chased", "scared"]
PATIENTS = ["me", "john", "child"]
MODES = ["reality", "tv", "game", "dream", "memory"]
FIELDS = ["actor", "action", "patient", "mode"]


@dataclass(frozen=True)
class Example:
    actor: str
    action: str
    patient: str
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounding-mode authority quick probe.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden = 24
        args.epochs = 20
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_examples() -> list[Example]:
    return [Example(a, act, p, m) for a in ACTORS for act in ACTIONS for p in PATIENTS for m in MODES]


def encode(examples: list[Example], *, ablate_mode: bool = False, mode_override: str | None = None) -> torch.Tensor:
    dim = len(ACTORS) + len(ACTIONS) + len(PATIENTS) + len(MODES)
    rows = torch.zeros(len(examples), dim)
    offsets = {
        "actor": 0,
        "action": len(ACTORS),
        "patient": len(ACTORS) + len(ACTIONS),
        "mode": len(ACTORS) + len(ACTIONS) + len(PATIENTS),
    }
    for i, ex in enumerate(examples):
        rows[i, offsets["actor"] + ACTORS.index(ex.actor)] = 1.0
        rows[i, offsets["action"] + ACTIONS.index(ex.action)] = 1.0
        rows[i, offsets["patient"] + PATIENTS.index(ex.patient)] = 1.0
        if not ablate_mode:
            mode = mode_override or ex.mode
            rows[i, offsets["mode"] + MODES.index(mode)] = 1.0
    return rows


def labels(examples: list[Example]) -> torch.Tensor:
    ys = []
    for ex in examples:
        semantic_injury = float(ex.action == "bit")
        semantic_threat = float(ex.action in {"bit", "scared"})
        story_authority = float(ex.mode == "tv")
        game_authority = float(ex.mode == "game")
        memory_authority = float(ex.mode in {"dream", "memory"})
        real_action = float(ex.mode == "reality" and ex.action in {"bit", "scared"} and ex.patient == "me")
        self_relevance = float(ex.patient == "me" and ex.mode == "reality")
        ys.append([
            semantic_injury,
            semantic_threat,
            real_action,
            story_authority,
            game_authority,
            memory_authority,
            self_relevance,
        ])
    return torch.tensor(ys, dtype=torch.float32)


class GroundingNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(len(ACTORS) + len(ACTIONS) + len(PATIENTS) + len(MODES), hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.semantic = nn.Linear(hidden, 2)
        self.authority = nn.Linear(hidden, 5)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {"semantic": self.semantic(h), "authority": self.authority(h), "all": torch.cat([self.semantic(h), self.authority(h)], dim=-1)}


def train_model(seed: int, *, hidden: int, epochs: int, lr: float) -> GroundingNet:
    set_seed(seed)
    examples = make_examples()
    x = encode(examples)
    y = labels(examples)
    model = GroundingNet(hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(x)["all"]
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def binary_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(((torch.sigmoid(logits) >= 0.5) == (y >= 0.5)).float().mean().item())


@torch.no_grad()
def evaluate(model: GroundingNet, examples: list[Example]) -> dict[str, float]:
    y = labels(examples)
    logits = model(encode(examples))["all"]
    probs = torch.sigmoid(logits)
    semantic_acc = binary_accuracy(logits[:, :2], y[:, :2])
    authority_acc = binary_accuracy(logits[:, 2:], y[:, 2:])
    real_idx = MODES.index("reality")
    mode_scores: dict[str, float] = {}
    for mode in MODES:
        subset = [i for i, ex in enumerate(examples) if ex.mode == mode and ex.action == "bit" and ex.patient == "me"]
        mode_scores[f"real_action_prob_{mode}"] = float(probs[subset, 2].mean().item())
    bit_indices = [i for i, ex in enumerate(examples) if ex.actor == "dog" and ex.action == "bit" and ex.patient == "me"]
    semantic_by_mode = {
        f"semantic_injury_prob_{mode}": float(probs[[i for i in bit_indices if examples[i].mode == mode], 0].mean().item())
        for mode in MODES
    }
    me_reality = [i for i, ex in enumerate(examples) if ex.action == "bit" and ex.patient == "me" and ex.mode == "reality"]
    john_reality = [i for i, ex in enumerate(examples) if ex.action == "bit" and ex.patient == "john" and ex.mode == "reality"]
    mode_leak = float(torch.stack([probs[[i for i, ex in enumerate(examples) if ex.mode == mode and ex.action == "bit" and ex.patient == "me"], 2].mean() for mode in ["tv", "game", "dream", "memory"]]).mean().item())
    return {
        "overall_accuracy": binary_accuracy(logits, y),
        "semantic_accuracy": semantic_acc,
        "authority_accuracy": authority_acc,
        "reality_action_authority": mode_scores["real_action_prob_reality"],
        "nonreality_action_leakage": mode_leak,
        "grounding_authority_margin": mode_scores["real_action_prob_reality"] - mode_leak,
        "self_anchor_gain": float(probs[me_reality, 2].mean().item() - probs[john_reality, 2].mean().item()),
        "semantic_consistency_range": max(semantic_by_mode.values()) - min(semantic_by_mode.values()),
        **mode_scores,
        **semantic_by_mode,
    }


@torch.no_grad()
def controls(model: GroundingNet, examples: list[Example], seed: int) -> dict[str, float]:
    y = labels(examples)
    base = torch.sigmoid(model(encode(examples))["all"])
    rng = random.Random(seed)
    shuffled = [Example(ex.actor, ex.action, ex.patient, rng.choice(MODES)) for ex in examples]
    shuffled_logits = model(encode(shuffled))["all"]
    no_mode_logits = model(encode(examples, ablate_mode=True))["all"]
    forced_tv_logits = model(encode(examples, mode_override="tv"))["all"]
    reality_self = [i for i, ex in enumerate(examples) if ex.action == "bit" and ex.patient == "me" and ex.mode == "reality"]
    return {
        "mode_shuffle_accuracy_against_true_labels": binary_accuracy(shuffled_logits, y),
        "mode_ablation_accuracy": binary_accuracy(no_mode_logits, y),
        "wrong_forced_tv_real_action_drop": float((base[reality_self, 2] - torch.sigmoid(forced_tv_logits)[reality_self, 2]).mean().item()),
    }


def summarize(vals: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def aggregate(records: list[dict[str, float]]) -> dict[str, Any]:
    keys = sorted({key for row in records for key in row})
    return {key: summarize([row[key] for row in records]) for key in keys}


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    return {
        "supports_semantic_grounding_split": agg["semantic_accuracy"]["mean"] >= 0.98 and agg["semantic_consistency_range"]["mean"] <= 0.05,
        "supports_grounding_mode_authority": agg["grounding_authority_margin"]["mean"] >= 0.70,
        "supports_self_anchor_authority": agg["self_anchor_gain"]["mean"] >= 0.40,
        "supports_wrong_mode_control": agg["wrong_forced_tv_real_action_drop"]["mean"] >= 0.70,
        "mode_leakage_low": agg["nonreality_action_leakage"]["mean"] <= 0.15,
    }


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Grounding Mode Authority Probe",
        "",
        "## Goal",
        "",
        "Test whether the same semantic event remains stable while action authority changes by grounding mode.",
        "",
        "Modes: `reality`, `tv`, `game`, `dream`, `memory`.",
        "",
        "## Results",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for key in [
        "overall_accuracy",
        "semantic_accuracy",
        "authority_accuracy",
        "reality_action_authority",
        "nonreality_action_leakage",
        "grounding_authority_margin",
        "self_anchor_gain",
        "semantic_consistency_range",
        "wrong_forced_tv_real_action_drop",
        "mode_shuffle_accuracy_against_true_labels",
        "mode_ablation_accuracy",
    ]:
        lines.append(f"| `{key}` | `{agg[key]['mean']:.6f}` | `{agg[key]['std']:.6f}` |")
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
        "This is a quick toy diagnostic. A positive result supports separating semantic event understanding from grounded action authority. It does not prove consciousness, biology, production validity, or natural-language understanding.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples = make_examples()
    records: list[dict[str, float]] = []
    for seed in range(args.seeds):
        print(f"[grounding-mode] seed={seed}", flush=True)
        model = train_model(seed, hidden=args.hidden, epochs=args.epochs, lr=args.learning_rate)
        row = evaluate(model, examples)
        row.update(controls(model, examples, seed + 991))
        records.append(row)
    summary = {
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "records": records,
        "aggregate": aggregate(records),
    }
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
