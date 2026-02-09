"""GPU Learning Diagnostic Campaign — AGC oscillation diagnosis.

Runs 4 configs x 200 steps to isolate whether AGC or task difficulty causes
the loss oscillation observed in the initial campaign (commit 93b630a).

Configs:
  1. assoc_byte + AGC ON   (control — reproduces baseline)
  2. assoc_byte + AGC OFF  (experiment 1: isolate AGC)
  3. assoc_clean + AGC ON  (experiment 2: isolate task difficulty)
  4. assoc_clean + AGC OFF  (experiment 2b: full isolation)

Each config runs at 4 step checkpoints (50, 100, 150, 200) as separate fresh
runs to capture loss trajectory without patching train_steps internals.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_DRAFTR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The runner captures: final eval loss, accuracy, update_scale, and loss_slope.
# The synth mode and dataset name are read from env vars set by the parent.
_RUNNER_TEMPLATE = '''
import json, os, sys, time
from pathlib import Path

draftr = Path(r"{draftr}")
sys.path.insert(0, str(draftr))
for c in [str(draftr.parent / "Golden Code"), r"S:\\AI\\Golden Code"]:
    try:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
            break
    except OSError:
        continue

import torch
from vraxion.instnct.absolute_hallway import AbsoluteHallway
from tools import instnct_train_steps, instnct_data

instnct_train_steps.log = lambda m: None

loader, num_classes, collate = instnct_data.get_seq_mnist_loader()
synth_mode = os.environ.get("VRX_SYNTH_MODE", "assoc_byte")
model = AbsoluteHallway(input_dim=1, num_classes=num_classes, ring_len=64, slot_dim=32)
n_params = sum(p.numel() for p in model.parameters())

# Record initial update_scale before training.
scale_init = float(getattr(model, "update_scale", 1.0))

sys.stderr.write(f"params={{n_params}}, num_classes={{num_classes}}, mode={{synth_mode}}\\n")
sys.stderr.flush()

steps = int(os.environ.get("VRX_MAX_STEPS", "200"))
t0 = time.time()
result = instnct_train_steps.train_steps(model, loader, steps, synth_mode, "absolute_hallway")
elapsed = time.time() - t0
sys.stderr.write(f"train done in {{elapsed:.1f}}s\\n")
sys.stderr.flush()

# Capture update_scale after training.
scale_final = float(getattr(model, "update_scale", scale_init))

model.eval()
criterion = torch.nn.CrossEntropyLoss()
total_loss = 0.0
total_correct = 0
total_samples = 0
with torch.no_grad():
    device = next(model.parameters()).device
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += y.size(0)

fl = total_loss / max(1, total_samples)
acc = total_correct / max(1, total_samples)
slope = result.get("loss_slope")

import math as _math
random_chance = 1.0 / max(1, num_classes)
random_loss = _math.log(max(1, num_classes))

print(json.dumps({{
    "n_params": n_params,
    "num_classes": num_classes,
    "steps": steps,
    "loss_slope": slope,
    "final_loss": round(fl, 6),
    "accuracy": round(acc, 4),
    "random_chance": round(random_chance, 4),
    "random_loss": round(random_loss, 4),
    "scale_init": round(scale_init, 6),
    "scale_final": round(scale_final, 6),
    "elapsed_s": round(elapsed, 1),
    "beats_random_loss": fl < random_loss,
    "beats_random_acc": acc > random_chance * 1.5,
    "pass_slope": (slope or 0.0) < 0,
}}))
'''


def run_config(name: str, extra_env: dict, steps: int = 200) -> dict:
    """Run a single training config as a subprocess."""
    env = os.environ.copy()
    env.update({
        "VRX_SYNTH": "1",
        "VRX_SYNTH_ONCE": "1",
        "VRX_SYNTH_MODE": "assoc_byte",
        "VRX_MAX_STEPS": str(steps),
        "VRX_RING_LEN": "64",
        "VRX_SLOT_DIM": "32",
        "VRX_MODULAR_SAVE": "0",
        "VRX_NAN_GUARD": "0",
        "VRX_DISABLE_SYNC": "1",
        "VAR_COMPUTE_DEVICE": "cpu",
    })
    env.update(extra_env)

    script = _RUNNER_TEMPLATE.format(draftr=_DRAFTR)
    print(f"  [{name}] Starting {steps} steps...", flush=True)
    t0 = time.time()

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,  # 60 min max per config
    )

    wall = time.time() - t0
    if proc.stderr:
        for line in proc.stderr.strip().splitlines()[-5:]:
            print(f"  [{name}] {line}", flush=True)

    if proc.returncode != 0:
        print(f"  [{name}] FAILED (rc={proc.returncode}, {wall:.0f}s wall)", flush=True)
        err_tail = proc.stderr[-2000:] if proc.stderr else ""
        return {"config": name, "error": f"exit code {proc.returncode}", "stderr": err_tail}

    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            result = json.loads(line)
            result["config"] = name
            return result

    return {"config": name, "error": "no JSON output"}


def run_config_trajectory(name: str, extra_env: dict,
                          checkpoints: tuple[int, ...] = (50, 100, 150, 200)) -> dict:
    """Run a config at multiple step counts to capture loss trajectory.

    Each checkpoint is a fresh run (independent init), so the trajectory
    shows expected loss at each training horizon rather than a single run's
    internal trajectory. This avoids patching train_steps internals.
    """
    trajectory = {}
    final_result = {}
    for step_count in checkpoints:
        tag = f"{name}@{step_count}"
        r = run_config(tag, extra_env, steps=step_count)
        if "error" in r:
            trajectory[step_count] = {"error": r["error"]}
            continue
        trajectory[step_count] = {
            "final_loss": r["final_loss"],
            "accuracy": r["accuracy"],
            "scale_final": r.get("scale_final"),
            "loss_slope": r.get("loss_slope"),
            "elapsed_s": r.get("elapsed_s"),
        }
        final_result = r  # keep the last successful run's full data

    final_result["config"] = name
    final_result["trajectory"] = trajectory
    return final_result


def analyze_trajectory(trajectory: dict) -> dict:
    """Analyze a loss trajectory for monotonicity and oscillation."""
    steps_sorted = sorted(k for k in trajectory if isinstance(k, int) and "error" not in trajectory[k])
    if len(steps_sorted) < 2:
        return {"trend": "insufficient_data", "monotonic_decreasing": False, "oscillating": False}

    losses = [trajectory[s]["final_loss"] for s in steps_sorted]

    # Check monotonic decrease.
    mono_dec = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))

    # Check oscillation: sign changes in consecutive differences.
    diffs = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
    sign_changes = sum(
        1 for i in range(len(diffs) - 1)
        if (diffs[i] > 0) != (diffs[i + 1] > 0)
    )
    oscillating = sign_changes >= 1 and not mono_dec

    # Overall slope: first to last.
    overall_slope = (losses[-1] - losses[0]) / max(1, steps_sorted[-1] - steps_sorted[0])

    if mono_dec:
        trend = "monotonic_decrease"
    elif overall_slope < -1e-5:
        trend = "decreasing_with_oscillation" if oscillating else "decreasing"
    elif overall_slope > 1e-5:
        trend = "increasing_with_oscillation" if oscillating else "increasing"
    else:
        trend = "flat_oscillating" if oscillating else "flat"

    return {
        "trend": trend,
        "monotonic_decreasing": mono_dec,
        "oscillating": oscillating,
        "overall_slope": round(overall_slope, 8),
        "losses": {s: round(l, 4) for s, l in zip(steps_sorted, losses)},
        "sign_changes": sign_changes,
    }


def format_diagnosis(all_results: list[dict]) -> str:
    """Format the diagnosis table and interpretation."""
    lines = []
    lines.append("=" * 70)
    lines.append("DIAGNOSTIC RESULTS")
    lines.append("=" * 70)
    lines.append("")

    # Summary table.
    lines.append(f"{'Config':<25} {'Loss@50':>8} {'Loss@100':>9} {'Loss@150':>9} "
                 f"{'Loss@200':>9} {'Acc':>7} {'Scale':>8} {'Trend':<25}")
    lines.append("-" * 110)

    for r in all_results:
        if "error" in r and "trajectory" not in r:
            lines.append(f"{r['config']:<25} ERROR: {r.get('error', 'unknown')}")
            continue
        traj = r.get("trajectory", {})
        losses = []
        for s in (50, 100, 150, 200):
            val = traj.get(s, {}).get("final_loss")
            losses.append(f"{val:>8.4f}" if val is not None else "     N/A")

        acc = r.get("accuracy", 0)
        scale = r.get("scale_final", 0)
        analysis = r.get("analysis", {})
        trend = analysis.get("trend", "?")

        lines.append(f"{r['config']:<25} {''.join(losses)} {acc:>7.2%} {scale:>8.4f} {trend:<25}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")

    # Build interpretation from results.
    result_map = {r["config"]: r for r in all_results}

    byte_agc_on = result_map.get("byte_agc_on", {})
    byte_agc_off = result_map.get("byte_agc_off", {})
    clean_agc_on = result_map.get("clean_agc_on", {})
    clean_agc_off = result_map.get("clean_agc_off", {})

    # Experiment 1: AGC OFF with assoc_byte.
    ba_off_analysis = byte_agc_off.get("analysis", {})
    ba_on_analysis = byte_agc_on.get("analysis", {})
    if ba_off_analysis.get("monotonic_decreasing") and not ba_on_analysis.get("monotonic_decreasing"):
        lines.append("Exp 1 (assoc_byte AGC OFF): LOSS SMOOTHLY DECREASES -> AGC is likely the problem.")
    elif ba_off_analysis.get("oscillating"):
        lines.append("Exp 1 (assoc_byte AGC OFF): LOSS STILL OSCILLATES -> AGC is NOT the cause.")
    else:
        lines.append(f"Exp 1 (assoc_byte AGC OFF): Trend = {ba_off_analysis.get('trend', '?')}")

    # Experiment 2: Easy task with AGC ON.
    cl_on_acc = clean_agc_on.get("accuracy", 0)
    if cl_on_acc > 0.70:
        lines.append(f"Exp 2 (assoc_clean AGC ON): ACC = {cl_on_acc:.1%} > 70% -> Model learns; assoc_byte was too hard.")
    elif cl_on_acc > 0.55:
        lines.append(f"Exp 2 (assoc_clean AGC ON): ACC = {cl_on_acc:.1%} — marginal improvement over random 50%.")
    else:
        lines.append(f"Exp 2 (assoc_clean AGC ON): ACC = {cl_on_acc:.1%} ~ random -> Model may have fundamental issue.")

    # Experiment 2b: Easy task + AGC OFF.
    cl_off_acc = clean_agc_off.get("accuracy", 0)
    if cl_off_acc > 0.70:
        lines.append(f"Exp 2b (assoc_clean AGC OFF): ACC = {cl_off_acc:.1%} > 70% -> Model works without AGC.")
    elif cl_off_acc <= 0.55:
        lines.append(f"Exp 2b (assoc_clean AGC OFF): ACC = {cl_off_acc:.1%} ~ random -> Model architecture may be broken.")
    else:
        lines.append(f"Exp 2b (assoc_clean AGC OFF): ACC = {cl_off_acc:.1%}")

    # Scale comparison.
    lines.append("")
    scale_on = byte_agc_on.get("scale_final")
    scale_off = byte_agc_off.get("scale_final")
    if scale_on is not None and scale_off is not None:
        lines.append(f"update_scale: AGC ON -> {scale_on:.6f}, AGC OFF -> {scale_off:.6f}")
        if scale_off == 1.0 or (scale_on is not None and abs(scale_on - scale_off) > 0.1):
            lines.append("  (AGC is actively modifying update_scale)")

    return "\n".join(lines)


def main():
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Diagnostic: AGC oscillation isolation")
    print()

    # 4 diagnostic configs: {task} x {AGC on/off}
    configs = [
        ("byte_agc_on", {
            "VRX_SYNTH_MODE": "assoc_byte",
        }),
        ("byte_agc_off", {
            "VRX_SYNTH_MODE": "assoc_byte",
            "VRX_AGC_ENABLED": "0",
        }),
        ("clean_agc_on", {
            "VRX_SYNTH_MODE": "assoc_clean",
        }),
        ("clean_agc_off", {
            "VRX_SYNTH_MODE": "assoc_clean",
            "VRX_AGC_ENABLED": "0",
        }),
    ]

    all_results = []
    campaign_start = time.time()

    for name, extra_env in configs:
        print(f"=== Config: {name} ===", flush=True)
        r = run_config_trajectory(name, extra_env, checkpoints=(50, 100, 150, 200))

        # Analyze the trajectory.
        if "trajectory" in r:
            r["analysis"] = analyze_trajectory(r["trajectory"])

        all_results.append(r)

        if "error" in r and "trajectory" not in r:
            print(f"  [{name}] ERROR: {r['error']}")
        else:
            traj = r.get("trajectory", {})
            analysis = r.get("analysis", {})
            acc = r.get("accuracy", 0)
            scale = r.get("scale_final", "?")
            trend = analysis.get("trend", "?")
            loss_200 = traj.get(200, {}).get("final_loss", "?")
            print(f"  [{name}] loss@200={loss_200} | acc={acc} | scale={scale} | trend={trend}")
        print(flush=True)

    total_wall = time.time() - campaign_start

    # Print diagnosis.
    print()
    print(format_diagnosis(all_results))
    print()
    print(f"Campaign total wall time: {total_wall:.0f}s ({total_wall/60:.1f} min)")
    print()
    print("=== RESULTS JSON ===")
    print(json.dumps(all_results, indent=2))

    # Write results to file.
    results_path = os.path.join(_DRAFTR, "tools", "_scratch", "campaign_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "diagnostic": "agc_oscillation_isolation",
            "gpu": gpu_name,
            "results": all_results,
            "total_wall_s": round(total_wall, 1),
        }, f, indent=2)
    print(f"\nResults written to: {results_path}")


if __name__ == "__main__":
    main()
