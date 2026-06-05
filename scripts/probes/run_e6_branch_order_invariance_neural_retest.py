#!/usr/bin/env python3
"""E6 branch-order invariance neural retest over the E4/E5 routing proxy."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
E5_PATH = Path(__file__).with_name("run_e5_substrate_necessity_test.py")
MILESTONE = "E6_BRANCH_ORDER_INVARIANCE_NEURAL_RETEST"
DEFAULT_OUT = Path("target/pilot_wave/e6_branch_order_invariance_neural_retest")
DEFAULT_SEEDS = (77001, 77002, 77003, 77004, 77005)


def load_e5_module() -> Any:
    spec = importlib.util.spec_from_file_location("e5_substrate_test", E5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E5 backend from {E5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e5 = load_e5_module()
e4 = e5.e4
e2 = e5.e2

HEADS = e5.HEADS
CHOICES = e5.CHOICES
FEATURE_DIM = e5.FEATURE_DIM
MAX_CHOICES = e5.MAX_CHOICES
HEAD_LOSS_WEIGHTS = e5.HEAD_LOSS_WEIGHTS
ROUTING_PASS_USEFULNESS = 0.95
ROUTING_PASS_ACCURACY = 0.95
ROUTING_PASS_BAD_RATE = 0.03

SYSTEMS = (
    "e4_top_down_reference",
    "mlp_fixed_order_gradient",
    "mlp_random_order_gradient",
    "recurrent_fixed_order_gradient",
    "recurrent_random_order_gradient",
    "choicewise_shared_random_order_gradient",
    "random_classifier",
)
GRADIENT_SYSTEMS = (
    "mlp_fixed_order_gradient",
    "mlp_random_order_gradient",
    "recurrent_fixed_order_gradient",
    "recurrent_random_order_gradient",
    "choicewise_shared_random_order_gradient",
)
HASH_ARTIFACTS = (
    "e6_invariance_comparison_report.json",
    "e6_branch_order_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    population_size: int
    generations: int
    mutation_sigma: float
    elite_count: int
    gradient_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    mlp_hidden_dim: int = 64
    recurrent_hidden_dim: int = 48
    choicewise_hidden_dim: int = 64
    execution_mode: str = "serial"
    parallel_workers: int = 1
    heartbeat_seconds: float = 20.0


class ChoicewiseSharedScorer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorers = nn.ModuleDict(
            {
                head: nn.Sequential(
                    nn.Linear(FEATURE_DIM, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
                for head in HEADS
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = {}
        for head_i, head in enumerate(HEADS):
            choice_count = len(CHOICES[head])
            logits[head] = self.scorers[head](x[:, head_i, :choice_count, :]).squeeze(-1)
        return logits


def round_float(value: float) -> float:
    return e5.round_float(value)


def parse_seeds(raw: str) -> tuple[int, ...]:
    return e5.parse_seeds(raw)


def resolve_out(path: str | Path) -> Path:
    return e5.resolve_out(path)


def stable_seed(label: str) -> int:
    return e2.stable_seed(f"e6-{label}")


def select_device(requested: str) -> str:
    return e5.select_device(requested)


def set_global_determinism(seed: int, device: str) -> None:
    e5.set_global_determinism(seed, device)


def e5_settings(settings: Settings) -> Any:
    return e5.Settings(
        seeds=settings.seeds,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        population_size=settings.population_size,
        generations=settings.generations,
        mutation_sigma=settings.mutation_sigma,
        elite_count=settings.elite_count,
        gradient_epochs=settings.gradient_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        device=settings.device,
        mlp_hidden_dim=settings.mlp_hidden_dim,
        recurrent_hidden_dim=settings.recurrent_hidden_dim,
        execution_mode=settings.execution_mode,
        parallel_workers=settings.parallel_workers,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def generate_task(settings: Settings) -> dict[str, Any]:
    task = e5.generate_task(e5_settings(settings))
    for split, data in task.items():
        data["split"] = split
    return task


def build_model(model_kind: str, settings: Settings) -> nn.Module:
    if model_kind == "mlp":
        return e5.TinyMLP(settings.mlp_hidden_dim)
    if model_kind == "recurrent":
        return e5.TinyGRULike(settings.recurrent_hidden_dim)
    if model_kind == "choicewise":
        return ChoicewiseSharedScorer(settings.choicewise_hidden_dim)
    raise ValueError(f"unknown model kind: {model_kind}")


def randomize_batch_order(
    x_batch: np.ndarray,
    y_batch: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    shuffled = x_batch.copy()
    targets = {head: values.copy() for head, values in y_batch.items()}
    row_count = shuffled.shape[0]
    for row in range(row_count):
        for head_i, head in enumerate(HEADS):
            choice_count = len(CHOICES[head])
            perm = rng.permutation(choice_count)
            shuffled[row, head_i, :choice_count, :] = shuffled[row, head_i, perm, :]
            old_target = int(targets[head][row])
            targets[head][row] = int(np.where(perm == old_target)[0][0])
    return shuffled, targets


def multi_head_loss(logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros((), dtype=torch.float32, device=next(iter(logits.values())).device)
    for head in HEADS:
        loss = loss + HEAD_LOSS_WEIGHTS[head] * F.cross_entropy(logits[head], targets[head])
    return loss


def torch_predictions(model: nn.Module, split_data: dict[str, Any], device: str, batch_size: int) -> dict[str, np.ndarray]:
    return e5.torch_predictions(model, split_data, device, batch_size)


def branch_order_eval_torch(model: nn.Module, split_data: dict[str, Any], seed: int, device: str, batch_size: int) -> dict[str, Any]:
    return e5.branch_order_eval_torch(model, split_data, seed, device, batch_size)


def branch_order_eval_e4(candidate: dict[str, Any], split_data: dict[str, Any], seed: int) -> dict[str, Any]:
    return e5.branch_order_eval_e4(candidate, split_data, seed)


def branch_order_eval_random(split_data: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pred = {head: rng.integers(0, len(CHOICES[head]), size=len(split_data["rows"]), dtype=np.int64) for head in HEADS}
    return e5.evaluate_predictions(pred, split_data, sample_limit=0)


def evaluate_model_all(system: str, model: nn.Module, task: dict[str, Any], settings: Settings, device: str) -> dict[str, Any]:
    evals = e5.evaluate_torch_model(model, task, device, settings.batch_size)
    branch = {
        split: branch_order_eval_torch(model, data, stable_seed(f"branch-final-{system}-{split}"), device, settings.batch_size)
        for split, data in task.items()
    }
    return {"evals": evals, "branch_order_evals": branch}


def train_gradient_system(system: str, model_kind: str, order_mode: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    seed = stable_seed(f"gradient-{system}-{settings.seeds}-{order_mode}")
    set_global_determinism(seed, device)
    model = build_model(model_kind, settings)
    model.to(device)
    initial_vector = e5.torch_state_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train = e5.padded_inputs(task["train"])
    y_train_np = e5.flatten_targets(task["train"])
    x_val = e5.padded_inputs(task["validation"])
    y_val_np = e5.flatten_targets(task["validation"])
    y_val = e5.target_tensors(task["validation"], device)
    history = []
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    for epoch in range(1, settings.gradient_epochs + 1):
        model.train()
        permutation = rng.permutation(x_train.shape[0])
        epoch_losses = []
        for batch_start in range(0, x_train.shape[0], settings.batch_size):
            indices = permutation[batch_start : batch_start + settings.batch_size]
            batch_x_np = x_train[indices]
            batch_y_np = {head: y_train_np[head][indices] for head in HEADS}
            if order_mode == "randomized":
                batch_x_np, batch_y_np = randomize_batch_order(batch_x_np, batch_y_np, rng)
            batch_x = torch.as_tensor(batch_x_np, dtype=torch.float32, device=device)
            batch_y = {head: torch.as_tensor(batch_y_np[head], dtype=torch.long, device=device) for head in HEADS}
            optimizer.zero_grad(set_to_none=True)
            loss = multi_head_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.as_tensor(x_val, dtype=torch.float32, device=device))
            val_loss = float(multi_head_loss(val_logits, y_val).detach().cpu())
        val_eval = e5.evaluate_predictions(torch_predictions(model, task["validation"], device, settings.batch_size), task["validation"], sample_limit=0)
        val_branch = branch_order_eval_torch(model, task["validation"], stable_seed(f"branch-val-{system}-{epoch}"), device, settings.batch_size)
        row = {
            "system": system,
            "epoch": epoch,
            "order_mode": order_mode,
            "train_loss": round_float(float(np.mean(epoch_losses))),
            "validation_loss": round_float(val_loss),
            "validation_usefulness": val_eval["metrics"]["answer_usefulness_score"],
            "validation_branch_order_usefulness": val_branch["metrics"]["answer_usefulness_score"],
            "validation_level_accuracy": val_eval["metrics"]["decision_relevant_level_accuracy"],
            "validation_path_accuracy": val_eval["metrics"]["causal_path_accuracy"],
            "state_hash": e5.state_hash_from_vector(e5.torch_state_vector(model)),
        }
        history.append(row)
        if out is not None:
            e2.append_progress(out, "epoch_complete", system=system, epoch=epoch, metrics=row)
            e2.write_json(out / f"e6_training_history_{system}.json", training_history_artifact(system, history, device, order_mode))
    runtime = time.perf_counter() - start_time
    final = evaluate_model_all(system, model, task, settings, device)
    initial_state = e5.model_summary(system, model, None) | {
        "state_hash": e5.state_hash_from_vector(initial_vector),
        "parameter_count": int(initial_vector.size),
    }
    final_state = e5.model_summary(system, model, initial_vector)
    model.to("cpu")
    return {
        "system": system,
        "substrate": "neural",
        "training_mode": "gradient",
        "model_kind": model_kind,
        "order_mode": order_mode,
        "device": device,
        "runtime_seconds": round_float(runtime),
        "initial_state": initial_state,
        "final_state": final_state,
        "parameter_count": int(initial_vector.size),
        "training_history": history,
        "final_eval": {"evals": final["evals"]},
        "branch_order_evals": final["branch_order_evals"],
        "_model": model,
        "_task": task,
    }


def run_top_down_reference(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    search = e5.run_e4_style_search("e4_top_down_reference", "top_down_hierarchical_router", task, e5_settings(settings), out)
    branch = {
        split: branch_order_eval_e4(search["_candidate"], data, stable_seed(f"branch-topdown-{split}"))
        for split, data in task.items()
    }
    search["branch_order_evals"] = branch
    return search


def random_classifier_eval(task: dict[str, Any]) -> dict[str, Any]:
    search = e5.random_classifier_eval(task, stable_seed("random-classifier"))
    search["branch_order_evals"] = {
        split: branch_order_eval_random(data, stable_seed(f"random-branch-{split}"))
        for split, data in task.items()
    }
    return search


def run_single_system(system: str, task: dict[str, Any], settings: Settings, out_raw: str | None) -> tuple[str, dict[str, Any]]:
    out = Path(out_raw) if out_raw else None
    started = time.perf_counter()
    if out is not None:
        e2.append_progress(out, "system_start", system=system, execution_mode=settings.execution_mode)
    try:
        if system == "e4_top_down_reference":
            search = run_top_down_reference(task, settings, out)
        elif system == "mlp_fixed_order_gradient":
            search = train_gradient_system(system, "mlp", "fixed", task, settings, out)
        elif system == "mlp_random_order_gradient":
            search = train_gradient_system(system, "mlp", "randomized", task, settings, out)
        elif system == "recurrent_fixed_order_gradient":
            search = train_gradient_system(system, "recurrent", "fixed", task, settings, out)
        elif system == "recurrent_random_order_gradient":
            search = train_gradient_system(system, "recurrent", "randomized", task, settings, out)
        elif system == "choicewise_shared_random_order_gradient":
            search = train_gradient_system(system, "choicewise", "randomized", task, settings, out)
        elif system == "random_classifier":
            search = random_classifier_eval(task)
        else:
            raise ValueError(f"unknown E6 system: {system}")
        if out is not None:
            e2.append_progress(out, "system_complete", system=system, runtime_seconds=round_float(time.perf_counter() - started))
        return system, search
    except Exception as exc:
        if out is not None:
            e2.append_progress(out, "system_failed", system=system, error=repr(exc))
        raise


def run_systems_serial(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    searches = {}
    for system in SYSTEMS:
        returned, search = run_single_system(system, task, settings, out.as_posix() if out else None)
        searches[returned] = search
    return searches


def run_systems_parallel(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    worker_count = settings.parallel_workers if settings.parallel_workers > 0 else min(len(SYSTEMS), max(1, (os.cpu_count() or 4) - 2))
    worker_count = min(max(1, worker_count), len(SYSTEMS))
    out_raw = out.as_posix() if out is not None else None
    if out is not None:
        e2.append_progress(out, "parallel_systems_start", systems=list(SYSTEMS), worker_count=worker_count)
    searches: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_single_system, system, task, settings, out_raw): system for system in SYSTEMS}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
            if not done:
                if out is not None:
                    e2.append_progress(
                        out,
                        "parallel_heartbeat",
                        completed_systems=sorted(searches),
                        pending_systems=sorted(futures[future] for future in pending),
                        worker_count=worker_count,
                    )
                continue
            for future in done:
                expected = futures[future]
                system, search = future.result()
                if system != expected:
                    raise RuntimeError(f"system result mismatch: expected {expected}, got {system}")
                searches[system] = search
                if out is not None:
                    e2.append_progress(out, "parallel_system_result_received", system=system, completed_count=len(searches), pending_count=len(pending))
    if out is not None:
        e2.append_progress(out, "parallel_systems_complete", completed_systems=sorted(searches))
    return searches


def routing_pass(metrics: dict[str, float]) -> bool:
    return (
        metrics["heldout_usefulness"] >= ROUTING_PASS_USEFULNESS
        and metrics["ood_usefulness"] >= ROUTING_PASS_USEFULNESS
        and metrics["counterfactual_usefulness"] >= ROUTING_PASS_USEFULNESS
        and metrics["adversarial_usefulness"] >= ROUTING_PASS_USEFULNESS
        and metrics["heldout_level_accuracy"] >= ROUTING_PASS_ACCURACY
        and metrics["heldout_causal_path_accuracy"] >= ROUTING_PASS_ACCURACY
        and metrics["heldout_stopping_depth_accuracy"] >= ROUTING_PASS_ACCURACY
        and metrics["heldout_over_detail_rate"] <= ROUTING_PASS_BAD_RATE
        and metrics["heldout_irrelevant_branch_rate"] <= ROUTING_PASS_BAD_RATE
    )


def branch_routing_pass(search: dict[str, Any]) -> bool:
    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        metrics = search["branch_order_evals"][split]["metrics"]
        if metrics["answer_usefulness_score"] < ROUTING_PASS_USEFULNESS:
            return False
        if metrics["decision_relevant_level_accuracy"] < ROUTING_PASS_ACCURACY:
            return False
        if metrics["causal_path_accuracy"] < ROUTING_PASS_ACCURACY:
            return False
        if metrics["stopping_depth_accuracy"] < ROUTING_PASS_ACCURACY:
            return False
        if metrics["over_detail_rate"] > ROUTING_PASS_BAD_RATE:
            return False
        if metrics["irrelevant_branch_expansion_rate"] > ROUTING_PASS_BAD_RATE:
            return False
    return True


def system_metrics(search: dict[str, Any]) -> dict[str, Any]:
    final = search["final_eval"]["evals"]
    heldout = final["heldout"]["metrics"]
    train_usefulness = final["train"]["metrics"]["answer_usefulness_score"]
    branch = search["branch_order_evals"]
    result = {
        "system": search["system"],
        "substrate": search.get("substrate", "unknown"),
        "training_mode": search.get("training_mode", "unknown"),
        "model_kind": search.get("model_kind", "unknown"),
        "order_mode": search.get("order_mode", "none"),
        "parameter_count": search.get("parameter_count", 0),
        "train_usefulness": train_usefulness,
        "validation_usefulness": final["validation"]["metrics"]["answer_usefulness_score"],
        "heldout_usefulness": heldout["answer_usefulness_score"],
        "ood_usefulness": final["ood"]["metrics"]["answer_usefulness_score"],
        "counterfactual_usefulness": final["counterfactual"]["metrics"]["answer_usefulness_score"],
        "adversarial_usefulness": final["adversarial"]["metrics"]["answer_usefulness_score"],
        "heldout_verdict_accuracy": heldout["verdict_accuracy"],
        "heldout_level_accuracy": heldout["decision_relevant_level_accuracy"],
        "heldout_causal_path_accuracy": heldout["causal_path_accuracy"],
        "heldout_stopping_depth_accuracy": heldout["stopping_depth_accuracy"],
        "heldout_over_detail_rate": heldout["over_detail_rate"],
        "heldout_under_detail_rate": heldout["under_detail_rate"],
        "heldout_irrelevant_branch_rate": heldout["irrelevant_branch_expansion_rate"],
        "generalization_gap": round_float(train_usefulness - heldout["answer_usefulness_score"]),
        "branch_order_heldout_usefulness": branch["heldout"]["metrics"]["answer_usefulness_score"],
        "branch_order_ood_usefulness": branch["ood"]["metrics"]["answer_usefulness_score"],
        "branch_order_counterfactual_usefulness": branch["counterfactual"]["metrics"]["answer_usefulness_score"],
        "branch_order_adversarial_usefulness": branch["adversarial"]["metrics"]["answer_usefulness_score"],
    }
    result["normal_routing_passed"] = routing_pass(result)
    result["branch_order_routing_passed"] = branch_routing_pass(search)
    result["clean_invariant_passed"] = result["normal_routing_passed"] and result["branch_order_routing_passed"]
    for key in ("mutation_attempt_count", "accepted_mutation_count", "rejected_mutation_count", "rollback_count"):
        if key in search:
            result[key] = search[key]
    return result


def label_shuffled_usefulness(search: dict[str, Any], task: dict[str, Any], settings: Settings) -> float:
    shuffled = e5.label_shuffled_split(task["heldout"], stable_seed(f"label-shuffle-{search['system']}"))
    if "_candidate" in search:
        eval_result = e4.evaluate_candidate(search["_candidate"], shuffled, sample_limit=0)
    elif "_model" in search:
        device = search.get("device", select_device(settings.device))
        pred = torch_predictions(search["_model"], shuffled, device, settings.batch_size)
        eval_result = e5.evaluate_predictions(pred, shuffled, sample_limit=0)
    else:
        eval_result = branch_order_eval_random(shuffled, stable_seed("label-random"))
    return eval_result["metrics"]["answer_usefulness_score"]


def aggregate_metrics(searches: dict[str, Any], task: dict[str, Any], settings: Settings, deterministic: dict[str, Any]) -> dict[str, Any]:
    systems = {system: system_metrics(searches[system]) for system in SYSTEMS}
    label_scores = {system: label_shuffled_usefulness(searches[system], task, settings) for system in SYSTEMS}
    label_control_passed = max(label_scores.values()) < 0.55
    invariant_neural = [
        system
        for system in ("mlp_random_order_gradient", "recurrent_random_order_gradient")
        if systems[system]["clean_invariant_passed"]
    ]
    fixed_order_failures = [
        system
        for system in ("mlp_fixed_order_gradient", "recurrent_fixed_order_gradient")
        if systems[system]["normal_routing_passed"] and not systems[system]["branch_order_routing_passed"]
    ]
    choicewise_clean = systems["choicewise_shared_random_order_gradient"]["clean_invariant_passed"]
    topdown_clean = systems["e4_top_down_reference"]["clean_invariant_passed"]
    return {
        "schema_version": "e6_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "winner": max(SYSTEMS, key=lambda system: systems[system]["heldout_usefulness"]),
        "label_shuffled_usefulness_by_system": label_scores,
        "label_control_passed": label_control_passed,
        "fixed_order_branch_failures": fixed_order_failures,
        "random_order_neural_clean_systems": invariant_neural,
        "choicewise_clean": choicewise_clean,
        "topdown_clean": topdown_clean,
        "deterministic_replay_passed": deterministic["internal_replay_passed"],
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    if not aggregate["label_control_passed"] or not aggregate["deterministic_replay_passed"]:
        decision = "e6_leak_or_replay_failure"
        next_step = "E6L_REPAIR_LABEL_AND_REPLAY_CONTROLS"
    elif aggregate["random_order_neural_clean_systems"]:
        decision = "e6_branch_order_invariance_training_succeeds"
        next_step = "E7_NEURAL_INVARIANCE_STRESS_AND_MUTATION_OPERATOR"
    elif aggregate["choicewise_clean"]:
        decision = "e6_order_equivariant_neural_architecture_viable"
        next_step = "E7_ORDER_EQUIVARIANT_NEURAL_STRESS_AND_MUTATION"
    elif aggregate["topdown_clean"]:
        decision = "e6_non_neural_router_remains_preferred"
        next_step = "E7_HARDER_SYMBOLIC_ROUTING_STRESS"
    else:
        decision = "e6_invariance_retest_inconclusive"
        next_step = "E6R_REPAIR_TASK_AND_BASELINES"
    return {
        "schema_version": "e6_decision_v1",
        "milestone": MILESTONE,
        "decision": decision,
        "winner": aggregate["winner"],
        "next": next_step,
        "topdown_clean": aggregate["topdown_clean"],
        "random_order_neural_clean_systems": aggregate["random_order_neural_clean_systems"],
        "choicewise_clean": aggregate["choicewise_clean"],
        "fixed_order_branch_failures": aggregate["fixed_order_branch_failures"],
        "label_control_passed": aggregate["label_control_passed"],
        "deterministic_replay_passed": aggregate["deterministic_replay_passed"],
    }


def deterministic_stub(passed: bool, comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e6_deterministic_replay_report_v1",
        "internal_replay_executed": True,
        "internal_replay_passed": passed,
        "deterministic_replay_passed": passed,
        "hash_artifacts": list(HASH_ARTIFACTS),
        "hash_comparisons": comparisons,
    }


def strip_search(search: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in search.items() if not key.startswith("_")}
    clean.pop("runtime_seconds", None)
    return clean


def parameter_diff_artifact(system: str, search: dict[str, Any]) -> dict[str, Any]:
    final = search.get("final_state", {})
    return {
        "schema_version": f"e6_parameter_diff_{system}_v1",
        "system": system,
        "actual_parameter_diff_found": final.get("changed_parameter_count", 0) > 0 or system == "random_classifier",
        "changed_parameter_count": final.get("changed_parameter_count", 0),
        "parameter_diff_l2": final.get("parameter_diff_l2", 0.0),
        "state_hash": final.get("state_hash"),
        "parameter_count": search.get("parameter_count", 0),
    }


def training_history_artifact(system: str, history: list[dict[str, Any]], device: str, order_mode: str) -> dict[str, Any]:
    return {
        "schema_version": f"e6_training_history_{system}_v1",
        "system": system,
        "device": device,
        "order_mode": order_mode,
        "optimizer": "AdamW",
        "backprop_used": True,
        "history": history,
    }


def backend_manifest(settings: Settings, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e6_backend_manifest_v1",
        "milestone": MILESTONE,
        "systems": list(SYSTEMS),
        "gradient_systems": list(GRADIENT_SYSTEMS),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": select_device(settings.device),
        "branch_order_randomized_training_used": True,
        "row_level_predictions_used": True,
        "population_size": settings.population_size,
        "generations": settings.generations,
        "gradient_epochs": settings.gradient_epochs,
        "settings": settings.__dict__,
        "git_preflight": git,
    }


def task_generation_report(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e6_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_e5_task_encoding": True,
        "splits": {split: {"row_count": len(data["rows"]), "first_row_id": data["rows"][0]["row_id"] if data["rows"] else None} for split, data in task.items()},
        "heads": list(HEADS),
        "max_choices": MAX_CHOICES,
        "feature_dim": FEATURE_DIM,
    }


def comparison_report(searches: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e6_invariance_comparison_report_v1",
        "milestone": MILESTONE,
        "systems": aggregate["systems"],
        "interpretation": {
            "fixed_order_controls": aggregate["fixed_order_branch_failures"],
            "random_order_neural_clean_systems": aggregate["random_order_neural_clean_systems"],
            "choicewise_clean": aggregate["choicewise_clean"],
            "topdown_clean": aggregate["topdown_clean"],
        },
    }


def branch_order_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e6_branch_order_report_v1",
        "branch_order_scores": {
            system: {
                "heldout": metrics["branch_order_heldout_usefulness"],
                "ood": metrics["branch_order_ood_usefulness"],
                "counterfactual": metrics["branch_order_counterfactual_usefulness"],
                "adversarial": metrics["branch_order_adversarial_usefulness"],
                "branch_order_routing_passed": metrics["branch_order_routing_passed"],
            }
            for system, metrics in aggregate["systems"].items()
        },
        "fixed_order_branch_failures": aggregate["fixed_order_branch_failures"],
        "random_order_neural_clean_systems": aggregate["random_order_neural_clean_systems"],
    }


def row_samples(searches: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "schema_version": f"e6_row_level_eval_sample_{split}_v1",
        "split": split,
        "samples": {system: searches[system]["final_eval"]["evals"][split]["row_level_samples"] for system in SYSTEMS},
    }


def report_md(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        f"# {MILESTONE} Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"winner = {decision['winner']}",
        f"next = {decision['next']}",
        "```",
        "",
        "## Systems",
        "",
    ]
    for system, metrics in aggregate["systems"].items():
        lines.append(
            f"- {system}: clean={metrics['clean_invariant_passed']} normal={metrics['heldout_usefulness']} "
            f"branch={metrics['branch_order_heldout_usefulness']} adv_branch={metrics['branch_order_adversarial_usefulness']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "E6 is a controlled symbolic branch-order invariance retest. It is not evidence for AGI, consciousness, natural-language reasoning, or model-scale behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def summary(decision: dict[str, Any], aggregate: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e6_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "winner": decision["winner"],
        "next": decision["next"],
        "git_status": git["git_status"],
        "topdown_clean": decision["topdown_clean"],
        "random_order_neural_clean_systems": decision["random_order_neural_clean_systems"],
        "choicewise_clean": decision["choicewise_clean"],
        "deterministic_replay_passed": decision["deterministic_replay_passed"],
    }


def compose_artifacts(core: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    searches = core["searches"]
    task = core["task"]
    aggregate = aggregate_metrics(searches, task, core["settings"], deterministic)
    decision = decide(aggregate)
    artifacts: dict[str, Any] = {
        "e6_backend_manifest.json": backend_manifest(core["settings"], core["git"]),
        "e6_task_generation_report.json": task_generation_report(task),
        "e6_invariance_comparison_report.json": comparison_report(searches, aggregate),
        "e6_branch_order_report.json": branch_order_report(aggregate),
        "e6_deterministic_replay_report.json": deterministic,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": summary(decision, aggregate, core["git"]),
        "report.md": report_md(decision, aggregate),
    }
    for system in SYSTEMS:
        search = searches[system]
        artifacts[f"e6_candidate_{system}_summary.json"] = strip_search(search)
        artifacts[f"e6_parameter_diff_{system}.json"] = parameter_diff_artifact(system, search)
        if system in GRADIENT_SYSTEMS:
            artifacts[f"e6_training_history_{system}.json"] = training_history_artifact(
                system,
                search["training_history"],
                search.get("device", "unknown"),
                search.get("order_mode", "unknown"),
            )
    if "history" in searches["e4_top_down_reference"]:
        artifacts["e6_mutation_history_e4_top_down_reference.json"] = e5.mutation_history_artifact(
            "e4_top_down_reference",
            searches["e4_top_down_reference"]["mutation_attempt_count"],
            searches["e4_top_down_reference"]["accepted_mutation_count"],
            searches["e4_top_down_reference"]["rejected_mutation_count"],
            searches["e4_top_down_reference"]["rollback_count"],
            searches["e4_top_down_reference"].get("history", []),
        )
    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        artifacts[f"e6_row_level_eval_sample_{split}.json"] = row_samples(searches, split)
    return artifacts


def write_artifacts(out: Path, core: dict[str, Any], deterministic: dict[str, Any]) -> None:
    artifacts = compose_artifacts(core, deterministic)
    for name, payload in artifacts.items():
        if isinstance(payload, str):
            e2.write_text(out / name, payload)
        else:
            e2.write_json(out / name, payload)
    e2.append_progress(out, "final_artifacts_written", artifact_count=len(artifacts))


def compare_core(primary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    primary_artifacts = compose_artifacts(primary, deterministic_stub(True, {}))
    replay_artifacts = compose_artifacts(replay, deterministic_stub(True, {}))
    comparisons = {}
    for name in HASH_ARTIFACTS:
        primary_hash = e2.payload_sha256(primary_artifacts[name])
        replay_hash = e2.payload_sha256(replay_artifacts[name])
        comparisons[name] = {"primary_hash": primary_hash, "replay_hash": replay_hash, "match": primary_hash == replay_hash}
    return deterministic_stub(all(row["match"] for row in comparisons.values()), comparisons)


def run_core(settings: Settings, out: Path | None = None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_global_determinism(stable_seed(f"core-{settings.seeds}-{device}"), device)
    task = generate_task(settings)
    git = e2.git_preflight()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        e2.append_progress(out, "startup", milestone=MILESTONE, settings=settings.__dict__, systems=list(SYSTEMS), selected_device=device)
    searches = run_systems_parallel(task, settings, out) if settings.execution_mode == "parallel" else run_systems_serial(task, settings, out)
    return {"settings": settings, "task": task, "git": git, "searches": searches}


def build_settings(args: argparse.Namespace) -> Settings:
    return Settings(
        seeds=parse_seeds(args.seeds),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        population_size=args.population_size,
        generations=args.generations,
        mutation_sigma=args.mutation_sigma,
        elite_count=args.elite_count,
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        execution_mode=args.execution_mode,
        parallel_workers=args.parallel_workers,
        heartbeat_seconds=args.heartbeat_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--validation-rows-per-seed", type=int, default=300)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=300)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--mutation-sigma", type=float, default=0.10)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--gradient-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--execution-mode", default="serial", choices=("serial", "parallel"))
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = build_settings(args)
    out = resolve_out(args.out)
    core = run_core(settings, out)
    replay = run_core(settings, out / "_internal_replay")
    deterministic = compare_core(core, replay)
    write_artifacts(out, core, deterministic)
    decision = compose_artifacts(core, deterministic)["decision.json"]
    print(json.dumps({"decision": decision["decision"], "winner": decision["winner"], "next": decision["next"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
