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
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "composable-skill-preservation"
REPORT_PATH = ROOT / "docs" / "research" / "COMPOSABLE_SKILL_PRESERVATION_PROBE.md"
SUMMARY_NAME = "composable_skill_preservation_summary.json"

P = 7
OPS = ["add", "mul", "add_then_mul", "mul_then_add"]
PRIMITIVE_OPS = ["add", "mul"]
COMPOSITE_OPS = ["add_then_mul", "mul_then_add"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick skill-preservation under composition probe.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--primitive-epochs", type=int, default=350)
    parser.add_argument("--composition-epochs", type=int, default=350)
    parser.add_argument("--module-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden = 16
        args.primitive_epochs = 40
        args.composition_epochs = 40
        args.module_epochs = 80
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def label(op: str, a: int, b: int, c: int) -> int:
    if op == "add":
        return (a + b) % P
    if op == "mul":
        return (a * b) % P
    if op == "add_then_mul":
        return ((a + b) % P * c) % P
    if op == "mul_then_add":
        return ((a * b) % P + c) % P
    raise ValueError(op)


def encode_example(op: str, a: int, b: int, c: int) -> torch.Tensor:
    x = torch.zeros(len(OPS) + 3 * P)
    x[OPS.index(op)] = 1.0
    x[len(OPS) + a] = 1.0
    x[len(OPS) + P + b] = 1.0
    x[len(OPS) + 2 * P + c] = 1.0
    return x


def make_examples(ops: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for op in ops:
        for a in range(P):
            for b in range(P):
                for c in range(P):
                    xs.append(encode_example(op, a, b, c))
                    ys.append(label(op, a, b, c))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def make_binary_examples(op: str) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for a in range(P):
        for b in range(P):
            x = torch.zeros(2 * P)
            x[a] = 1.0
            x[P + b] = 1.0
            xs.append(x)
            ys.append(label(op, a, b, 0))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class SharedNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(OPS) + 3 * P, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, P),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrimitiveModule(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * P, hidden),
            nn.Tanh(),
            nn.Linear(hidden, P),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((model(x).argmax(dim=-1) == y).float().mean().item())


def train_ce(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, epochs: int, lr: float) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()


@torch.no_grad()
def module_predict(module: PrimitiveModule, a: int, b: int) -> int:
    x = torch.zeros(1, 2 * P)
    x[0, a] = 1.0
    x[0, P + b] = 1.0
    return int(module(x).argmax(dim=-1).item())


@torch.no_grad()
def frozen_interpreter_metrics(add_mod: PrimitiveModule, mul_mod: PrimitiveModule) -> dict[str, float]:
    primitive_ok = 0
    primitive_total = 0
    composite_ok = 0
    composite_total = 0
    for a in range(P):
        for b in range(P):
            add_pred = module_predict(add_mod, a, b)
            mul_pred = module_predict(mul_mod, a, b)
            primitive_ok += int(add_pred == (a + b) % P)
            primitive_ok += int(mul_pred == (a * b) % P)
            primitive_total += 2
            for c in range(P):
                add_then_mul = module_predict(mul_mod, add_pred, c)
                mul_then_add = module_predict(add_mod, mul_pred, c)
                composite_ok += int(add_then_mul == label("add_then_mul", a, b, c))
                composite_ok += int(mul_then_add == label("mul_then_add", a, b, c))
                composite_total += 2
    return {
        "primitive_accuracy_after_composition": primitive_ok / primitive_total,
        "composition_accuracy": composite_ok / composite_total,
        "primitive_drift": 0.0,
    }


@dataclass
class SeedResult:
    seed: int
    shared_no_replay: dict[str, float]
    shared_with_replay: dict[str, float]
    frozen_modules: dict[str, float]


def train_shared(seed: int, *, hidden: int, primitive_epochs: int, composition_epochs: int, lr: float, replay: bool) -> dict[str, float]:
    set_seed(seed)
    primitive_x, primitive_y = make_examples(PRIMITIVE_OPS)
    composite_x, composite_y = make_examples(COMPOSITE_OPS)
    model = SharedNet(hidden)
    train_ce(model, primitive_x, primitive_y, epochs=primitive_epochs, lr=lr)
    primitive_before = accuracy(model, primitive_x, primitive_y)
    if replay:
        x2 = torch.cat([composite_x, primitive_x], dim=0)
        y2 = torch.cat([composite_y, primitive_y], dim=0)
    else:
        x2 = composite_x
        y2 = composite_y
    train_ce(model, x2, y2, epochs=composition_epochs, lr=lr)
    primitive_after = accuracy(model, primitive_x, primitive_y)
    composition_after = accuracy(model, composite_x, composite_y)
    return {
        "primitive_accuracy_before_composition": primitive_before,
        "primitive_accuracy_after_composition": primitive_after,
        "composition_accuracy": composition_after,
        "primitive_drift": primitive_before - primitive_after,
    }


def train_frozen_modules(seed: int, *, hidden: int, module_epochs: int, lr: float) -> dict[str, float]:
    set_seed(seed)
    add_x, add_y = make_binary_examples("add")
    mul_x, mul_y = make_binary_examples("mul")
    add_mod = PrimitiveModule(hidden)
    mul_mod = PrimitiveModule(hidden)
    train_ce(add_mod, add_x, add_y, epochs=module_epochs, lr=lr)
    train_ce(mul_mod, mul_x, mul_y, epochs=module_epochs, lr=lr)
    metrics = frozen_interpreter_metrics(add_mod, mul_mod)
    metrics["primitive_accuracy_before_composition"] = metrics["primitive_accuracy_after_composition"]
    return metrics


def summarize(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate(seed_results: list[SeedResult]) -> dict[str, Any]:
    arms = ["shared_no_replay", "shared_with_replay", "frozen_modules"]
    out: dict[str, Any] = {}
    for arm in arms:
        rows = [getattr(row, arm) for row in seed_results]
        keys = sorted({key for row in rows for key in row})
        out[arm] = {key: summarize([row[key] for row in rows]) for key in keys}
    return out


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    no_replay_drift = agg["shared_no_replay"]["primitive_drift"]["mean"]
    replay_drift = agg["shared_with_replay"]["primitive_drift"]["mean"]
    frozen_comp = agg["frozen_modules"]["composition_accuracy"]["mean"]
    return {
        "shared_no_replay_forgets_primitives": no_replay_drift >= 0.20,
        "replay_reduces_forgetting": replay_drift < no_replay_drift - 0.10,
        "frozen_modules_preserve_primitives": agg["frozen_modules"]["primitive_drift"]["mean"] == 0.0,
        "frozen_modules_compose_successfully": frozen_comp >= 0.95,
        "logic_controller_hypothesis_supported": no_replay_drift >= 0.20 and frozen_comp >= 0.95,
    }


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Composable Skill Preservation Probe",
        "",
        "## Goal",
        "",
        "Quickly test whether composition training in a shared end-to-end network overwrites primitive skills, while frozen primitive modules preserve skill identity under composition.",
        "",
        "Tasks use arithmetic modulo 7:",
        "",
        "- primitives: `add`, `mul`",
        "- composites: `add_then_mul`, `mul_then_add`",
        "",
        "## Results",
        "",
        "| Arm | Primitive Before | Primitive After | Drift | Composition |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ["shared_no_replay", "shared_with_replay", "frozen_modules"]:
        item = agg[arm]
        lines.append(
            f"| `{arm}` "
            f"| `{item['primitive_accuracy_before_composition']['mean']:.6f}` "
            f"| `{item['primitive_accuracy_after_composition']['mean']:.6f}` "
            f"| `{item['primitive_drift']['mean']:.6f}` "
            f"| `{item['composition_accuracy']['mean']:.6f}` |"
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
        "This is a quick toy check, not a final architecture claim. The frozen-module arm is an existence/reference check: it uses the intended composition policy and does not prove that the controller can discover that policy. A positive pattern means the next serious probe should compare shared-loss training against a modular controller with frozen primitive regression tests and explicit route/edge survival rules.",
        "",
        "Claim boundary: toy evidence only; no consciousness, biology, or production validation claim.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[SeedResult] = []
    for seed in range(args.seeds):
        print(f"[skill-preserve] seed={seed}", flush=True)
        seed_results.append(
            SeedResult(
                seed=seed,
                shared_no_replay=train_shared(
                    seed,
                    hidden=args.hidden,
                    primitive_epochs=args.primitive_epochs,
                    composition_epochs=args.composition_epochs,
                    lr=args.learning_rate,
                    replay=False,
                ),
                shared_with_replay=train_shared(
                    seed + 10_000,
                    hidden=args.hidden,
                    primitive_epochs=args.primitive_epochs,
                    composition_epochs=args.composition_epochs,
                    lr=args.learning_rate,
                    replay=True,
                ),
                frozen_modules=train_frozen_modules(
                    seed + 20_000,
                    hidden=args.hidden,
                    module_epochs=args.module_epochs,
                    lr=args.learning_rate,
                ),
            )
        )
    summary = {
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "seeds": [
            {
                "seed": row.seed,
                "shared_no_replay": row.shared_no_replay,
                "shared_with_replay": row.shared_with_replay,
                "frozen_modules": row.frozen_modules,
            }
            for row in seed_results
        ],
        "aggregate": aggregate(seed_results),
    }
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
