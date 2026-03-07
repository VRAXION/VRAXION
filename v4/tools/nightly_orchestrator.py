#!/usr/bin/env python3
"""Overnight queue orchestrator for canonical nightly runner jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_ROOT = REPO_ROOT / "v4"
TESTS_DIR = V4_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from nightly_research_runner import SURFACES, VARIANTS  # type: ignore[import-not-found]


RUNNER_PATH = TESTS_DIR / "nightly_research_runner.py"
DEFAULT_PLAN_PATH = V4_ROOT / "tools" / "queues" / "overnight_llt_validation.json"
DEFAULT_RUNTIME_BASE = REPO_ROOT / "bench_vault" / "night_runs"
HYBERNATION_PING_DETACH = Path(
    os.environ.get(
        "VRX_HYBERNATION_PING_DETACH",
        "C:/Users/kenes/.codex/skills/HybernationPing/scripts/hybernation_ping_detach.py",
    )
)

DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

ALLOWED_CPU_ROLES = {"cpu_validator"}
ALLOWED_GPU_ROLES = {"gpu_primary", "gpu_secondary"}
SUMMARY_KEYS = (
    "final_acc",
    "best_acc",
    "final_bpc",
    "final_loss",
    "carry_eval_acc",
    "fresh_eval_acc",
    "carry_minus_reset_pp",
    "time_s",
    "s_per_step",
    "max_grad",
)
DEFAULT_LANE_DEFAULTS = {
    "cpu": {"max_restarts": 1, "watchdog_no_output_s": 1800, "restart_delay_s": 10},
    "gpu": {"max_restarts": 1, "watchdog_no_output_s": 2400, "restart_delay_s": 15},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _epoch_to_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat()


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value).timestamp()


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_") or "nightly"


def _default_runtime_root(plan_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNTIME_BASE / f"{stamp}_{plan_name}"


def _job_artifact_path(runtime_root: Path, job_id: str) -> Path:
    return runtime_root / "artifacts" / f"{job_id}.json"


def _job_stdout_path(runtime_root: Path, job_id: str) -> Path:
    return runtime_root / "jobs" / job_id / "stdout.log"


def _job_stderr_path(runtime_root: Path, job_id: str) -> Path:
    return runtime_root / "jobs" / job_id / "stderr.log"


def _job_heartbeat_path(runtime_root: Path, job_id: str) -> Path:
    return runtime_root / "jobs" / job_id / "heartbeat.json"


def _resolve_variant(job: dict[str, Any], decisions: dict[str, Any]) -> str:
    if job.get("variant"):
        return job["variant"]
    ref = job.get("variant_from_decision")
    if not ref:
        raise KeyError(f"Job {job['id']} has neither variant nor variant_from_decision")
    stage_id, key = ref.split(".", 1)
    return decisions[stage_id][key]


def _summary_from_artifact(artifact_path: Path) -> dict[str, Any]:
    payload = _load_json(artifact_path)
    result = payload.get("result", {})
    meta = payload.get("meta", {})
    guards = payload.get("guards", {})
    summary = {key: result.get(key) for key in SUMMARY_KEYS if key in result}
    summary["surface_kind"] = meta.get("surface_kind")
    summary["variant"] = meta.get("variant")
    summary["device"] = meta.get("device")
    summary["seed"] = meta.get("seed")
    summary["effective_global_read"] = guards.get("effective_global_read")
    summary["effective_global_write"] = guards.get("effective_global_write")
    return summary


def _ping_via_hybernation(message: str, timer_sec: int) -> None:
    if not HYBERNATION_PING_DETACH.exists():
        return
    try:
        subprocess.Popen(
            [
                sys.executable,
                str(HYBERNATION_PING_DETACH),
                "--timer-sec",
                str(max(1, int(timer_sec))),
                "--message",
                message,
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception:
        return


def _emit_wake(runtime_root: Path, watch_cfg: dict[str, Any], reason: str, message: str, status_path: Path, summary_path: Path, job_id: str | None = None) -> None:
    wake_path = runtime_root / watch_cfg.get("wake_file", "wake_trigger.json")
    payload = {
        "timestamp": _now_iso(),
        "reason": reason,
        "job_id": job_id,
        "message": message,
        "status_path": str(status_path),
        "summary_path": str(summary_path),
    }
    _safe_json_write(wake_path, payload)
    if watch_cfg.get("ping_enabled", False):
        timer_sec = 1 if reason != "heartbeat" else int(watch_cfg.get("ping_timer_sec", 1))
        _ping_via_hybernation(message, timer_sec=timer_sec)


def _lane_defaults(plan: dict[str, Any], lane: str) -> dict[str, int]:
    defaults = deepcopy(DEFAULT_LANE_DEFAULTS.get(lane, {}))
    plan_defaults = plan.get("lane_defaults", {}).get(lane, {})
    defaults.update({k: int(v) for k, v in plan_defaults.items() if k in defaults})
    return defaults


def _validate_job(job: dict[str, Any], lane: str) -> None:
    required = ("id", "role", "lane", "surface", "steps", "seed", "device", "stop_on_fail", "wake_on_complete", "evidence_required")
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"Job {job.get('id', '<unknown>')} missing required fields: {missing}")
    if job["lane"] != lane:
        raise ValueError(f"Job {job['id']} lane mismatch: {job['lane']} != {lane}")
    if lane == "cpu":
        if job["role"] not in ALLOWED_CPU_ROLES or job["device"] != "cpu":
            raise ValueError(f"CPU job {job['id']} must use role=cpu_validator and device=cpu")
    else:
        if job["role"] not in ALLOWED_GPU_ROLES or job["device"] != "cuda":
            raise ValueError(f"GPU job {job['id']} must use gpu role and device=cuda")
    if job["surface"] not in SURFACES:
        raise ValueError(f"Job {job['id']} uses unknown surface {job['surface']}")
    if "variant" not in job and "variant_from_decision" not in job:
        raise ValueError(f"Job {job['id']} must define variant or variant_from_decision")
    if "variant" in job and job["variant"] not in VARIANTS:
        raise ValueError(f"Job {job['id']} uses unknown variant {job['variant']}")


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("version") != 1:
        raise ValueError("Only version=1 queue plans are supported")
    if not plan.get("name"):
        raise ValueError("Plan must define name")
    watch = plan.get("watch")
    if not isinstance(watch, dict) or int(watch.get("heartbeat_minutes", 0)) <= 0:
        raise ValueError("Plan watch.heartbeat_minutes must be > 0")
    cpu_stages = plan.get("cpu_stages")
    gpu_queue = plan.get("gpu_queue")
    if not isinstance(cpu_stages, list) or not cpu_stages:
        raise ValueError("Plan must define non-empty cpu_stages")
    if not isinstance(gpu_queue, list):
        raise ValueError("Plan must define gpu_queue")
    lane_defaults = plan.get("lane_defaults", {})
    if lane_defaults:
        if not isinstance(lane_defaults, dict):
            raise ValueError("lane_defaults must be an object")
        for lane in ("cpu", "gpu"):
            if lane in lane_defaults and not isinstance(lane_defaults[lane], dict):
                raise ValueError(f"lane_defaults.{lane} must be an object")
    lane_env = plan.get("lane_env", {})
    if lane_env:
        if not isinstance(lane_env, dict):
            raise ValueError("lane_env must be an object")
        for lane in ("cpu", "gpu"):
            if lane in lane_env and not isinstance(lane_env[lane], dict):
                raise ValueError(f"lane_env.{lane} must be an object")
    seen_jobs: set[str] = set()
    for stage in cpu_stages:
        if not stage.get("id"):
            raise ValueError("CPU stage missing id")
        jobs = stage.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError(f"CPU stage {stage['id']} must have jobs")
        for job in jobs:
            _validate_job(job, "cpu")
            if job["id"] in seen_jobs:
                raise ValueError(f"Duplicate job id: {job['id']}")
            seen_jobs.add(job["id"])
    for job in gpu_queue:
        _validate_job(job, "gpu")
        if job["id"] in seen_jobs:
            raise ValueError(f"Duplicate job id: {job['id']}")
        seen_jobs.add(job["id"])


def build_initial_status(plan: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    jobs = {}
    cpu_stages = {}
    for stage in plan["cpu_stages"]:
        sid = stage["id"]
        cpu_stages[sid] = {
            "enabled": bool(stage.get("enabled", False)),
            "status": "pending" if stage.get("enabled", False) else "blocked",
            "decision": None,
            "jobs": [job["id"] for job in stage["jobs"]],
        }
        for job in stage["jobs"]:
            lane_defaults = _lane_defaults(plan, "cpu")
            record = deepcopy(job)
            lane_env = deepcopy(plan.get("lane_env", {}).get("cpu", {}))
            job_env = deepcopy(job.get("env", {}))
            merged_env = {**lane_env, **job_env}
            record.update(
                {
                    "stage_id": sid,
                    "status": "pending" if stage.get("enabled", False) else "blocked",
                    "variant_resolved": None,
                    "artifact_path": None,
                    "stdout_log": None,
                    "stderr_log": None,
                    "heartbeat_path": None,
                    "start_ts": None,
                    "end_ts": None,
                    "duration_s": None,
                    "summary_metrics": None,
                    "wake_reason": None,
                    "exit_code": None,
                    "error": None,
                    "pid": None,
                    "restart_count": 0,
                    "max_restarts": int(job.get("max_restarts", lane_defaults["max_restarts"])),
                    "watchdog_no_output_s": int(job.get("watchdog_no_output_s", lane_defaults["watchdog_no_output_s"])),
                    "restart_delay_s": int(job.get("restart_delay_s", lane_defaults["restart_delay_s"])),
                    "last_output_ts": None,
                    "retry_not_before": None,
                    "env": merged_env,
                }
            )
            jobs[job["id"]] = record
    gpu_ids = []
    for job in plan["gpu_queue"]:
        lane_defaults = _lane_defaults(plan, "gpu")
        record = deepcopy(job)
        lane_env = deepcopy(plan.get("lane_env", {}).get("gpu", {}))
        job_env = deepcopy(job.get("env", {}))
        merged_env = {**lane_env, **job_env}
        record.update(
            {
                "stage_id": None,
                "status": "pending",
                "variant_resolved": None,
                "artifact_path": None,
                "stdout_log": None,
                "stderr_log": None,
                "heartbeat_path": None,
                "start_ts": None,
                "end_ts": None,
                "duration_s": None,
                "summary_metrics": None,
                "wake_reason": None,
                "exit_code": None,
                "error": None,
                "pid": None,
                "restart_count": 0,
                "max_restarts": int(job.get("max_restarts", lane_defaults["max_restarts"])),
                "watchdog_no_output_s": int(job.get("watchdog_no_output_s", lane_defaults["watchdog_no_output_s"])),
                "restart_delay_s": int(job.get("restart_delay_s", lane_defaults["restart_delay_s"])),
                "last_output_ts": None,
                "retry_not_before": None,
                "env": merged_env,
            }
        )
        jobs[job["id"]] = record
        gpu_ids.append(job["id"])
    return {
        "plan_name": plan["name"],
        "runtime_root": str(runtime_root),
        "state": "pending",
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "lanes": {"cpu": {"active_job_id": None, "current_stage": None}, "gpu": {"active_job_id": None, "queue": gpu_ids}},
        "cpu_stages": cpu_stages,
        "decisions": {},
        "jobs": jobs,
    }


def _build_summary(status: dict[str, Any]) -> dict[str, Any]:
    jobs = list(status["jobs"].values())
    return {
        "plan_name": status["plan_name"],
        "runtime_root": status["runtime_root"],
        "state": status["state"],
        "updated_at": status["updated_at"],
        "decisions": status["decisions"],
        "running_jobs": [job for job in jobs if job["status"] == "running"],
        "completed_jobs": [job for job in jobs if job["status"] == "done"],
        "failed_jobs": [job for job in jobs if job["status"] == "failed"],
    }


# decision / scheduler / CLI live below


def _write_status_and_summary(status: dict[str, Any], runtime_root: Path) -> tuple[Path, Path]:
    status["updated_at"] = _now_iso()
    status_path = runtime_root / "status.json"
    summary_path = runtime_root / "summary.json"
    _safe_json_write(status_path, status)
    _safe_json_write(summary_path, _build_summary(status))
    return status_path, summary_path


def _job_terminal(job: dict[str, Any]) -> bool:
    return job["status"] in {"done", "failed", "skipped"}


def _enable_stage(status: dict[str, Any], stage_id: str) -> None:
    stage_state = status["cpu_stages"][stage_id]
    if stage_state["status"] == "skipped":
        return
    stage_state["enabled"] = True
    stage_state["status"] = "pending"
    for job_id in stage_state["jobs"]:
        if status["jobs"][job_id]["status"] == "blocked":
            status["jobs"][job_id]["status"] = "pending"


def _skip_stage(status: dict[str, Any], stage_id: str) -> None:
    stage_state = status["cpu_stages"][stage_id]
    stage_state["enabled"] = False
    stage_state["status"] = "skipped"
    for job_id in stage_state["jobs"]:
        if status["jobs"][job_id]["status"] in {"blocked", "pending"}:
            status["jobs"][job_id]["status"] = "skipped"


def _apply_stage_decision(status: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any] | None:
    decision = stage.get("decision")
    if not decision:
        return None
    if decision["kind"] == "promote_candidate":
        baseline = [status["jobs"][job_id]["summary_metrics"] for job_id in decision["baseline_ids"]]
        candidate = [status["jobs"][job_id]["summary_metrics"] for job_id in decision["candidate_ids"]]
        if any(item is None for item in baseline + candidate):
            raise RuntimeError(f"Stage {stage['id']} missing metrics for promote_candidate")
        acc_delta_pp = (_mean([item["final_acc"] for item in candidate]) - _mean([item["final_acc"] for item in baseline])) * 100.0
        bpc_improve = _mean([item["final_bpc"] for item in baseline]) - _mean([item["final_bpc"] for item in candidate])
        per_seed_pp = [
            (cand["final_acc"] - base["final_acc"]) * 100.0
            for base, cand in zip(baseline, candidate, strict=True)
        ]
        passed = (
            acc_delta_pp >= float(decision["final_acc_gain_min_pp"])
            and bpc_improve >= float(decision["bpc_improve_min"])
            and min(per_seed_pp) >= float(decision["seed_floor_pp"])
        )
        status["decisions"][stage["id"]] = {
            "kind": "promote_candidate",
            "candidate_variant": decision["candidate_variant"],
            "baseline_variant": decision["baseline_variant"],
            "mean_acc_delta_pp": acc_delta_pp,
            "mean_bpc_improve": bpc_improve,
            "per_seed_deltas_pp": per_seed_pp,
            "passed": passed,
        }
        if passed:
            for sid in decision.get("on_pass_enable", []):
                _enable_stage(status, sid)
        else:
            for sid in decision.get("on_fail_skip", []):
                _skip_stage(status, sid)
        return status["decisions"][stage["id"]]

    if decision["kind"] == "select_best_variant":
        stage_jobs = [status["jobs"][job_id] for job_id in decision["job_ids"]]
        if any(job["summary_metrics"] is None for job in stage_jobs):
            raise RuntimeError(f"Stage {stage['id']} missing metrics for select_best_variant")
        ordered = sorted(
            stage_jobs,
            key=lambda job: (
                -float(job["summary_metrics"]["final_acc"]),
                float(job["summary_metrics"]["final_bpc"]),
                float(job["summary_metrics"]["time_s"]),
            ),
        )
        winner = ordered[0]
        winner_variant = winner["variant_resolved"] or winner["variant"]
        winner_key = decision.get("winner_key", "winner_variant")
        status["decisions"][stage["id"]] = {
            "kind": "select_best_variant",
            winner_key: winner_variant,
            "winner_job_id": winner["id"],
            "winner_variant": winner_variant,
            "winner_is_preferred": winner_variant == decision["preferred_variant"],
        }
        for sid in decision.get("on_complete_enable", []):
            _enable_stage(status, sid)
        if winner_variant == decision["preferred_variant"]:
            for sid in decision.get("skip_if_winner_is_preferred", []):
                _skip_stage(status, sid)
        else:
            for sid in decision.get("enable_if_winner_not_preferred", []):
                _enable_stage(status, sid)
        return status["decisions"][stage["id"]]

    raise RuntimeError(f"Unsupported decision kind: {decision['kind']}")


def _resolve_job_record(job: dict[str, Any], runtime_root: Path, decisions: dict[str, Any]) -> dict[str, Any]:
    job["variant_resolved"] = _resolve_variant(job, decisions)
    job["artifact_path"] = str(_job_artifact_path(runtime_root, job["id"]))
    job["stdout_log"] = str(_job_stdout_path(runtime_root, job["id"]))
    job["stderr_log"] = str(_job_stderr_path(runtime_root, job["id"]))
    job["heartbeat_path"] = str(_job_heartbeat_path(runtime_root, job["id"]))
    return job


def _build_runner_cmd(job: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(RUNNER_PATH),
        "--surface",
        job["surface"],
        "--variant",
        job["variant_resolved"],
        "--steps",
        str(job["steps"]),
        "--device",
        job["device"],
        "--seed",
        str(job["seed"]),
        "--heartbeat-out",
        job["heartbeat_path"],
        "--json-out",
        job["artifact_path"],
    ]
    for opt in ("pointer_mode", "pointer_interp_mode", "pointer_seam_mode"):
        if job.get(opt):
            cmd.extend([f"--{opt.replace('_', '-')}", str(job[opt])])
    return cmd


def _start_job(job: dict[str, Any]) -> dict[str, Any]:
    stdout_path = Path(job["stdout_log"])
    stderr_path = Path(job["stderr_log"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = open(stdout_path, "a", encoding="utf-8")
    stderr_handle = open(stderr_path, "a", encoding="utf-8")
    banner = (
        f"\n=== START ts={_now_iso()} job={job['id']} attempt={job['restart_count'] + 1} "
        f"variant={job['variant_resolved']} surface={job['surface']} device={job['device']} ===\n"
    )
    stdout_handle.write(banner)
    stderr_handle.write(banner)
    stdout_handle.flush()
    stderr_handle.flush()
    child_env = os.environ.copy()
    child_env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
    proc = subprocess.Popen(
        _build_runner_cmd(job),
        cwd=str(REPO_ROOT),
        stdout=stdout_handle,
        stderr=stderr_handle,
        env=child_env,
    )
    return {"proc": proc, "stdout_handle": stdout_handle, "stderr_handle": stderr_handle}


def _log_mtime(path_str: str | None) -> float | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return path.stat().st_mtime


def _latest_output_mtime(job: dict[str, Any]) -> float | None:
    mtimes = [
        _log_mtime(job.get("heartbeat_path")),
        _log_mtime(job.get("stdout_log")),
        _log_mtime(job.get("stderr_log")),
    ]
    mtimes = [mtime for mtime in mtimes if mtime is not None]
    return max(mtimes) if mtimes else None


def _job_can_start(job: dict[str, Any], now_epoch: float | None = None) -> bool:
    if job["status"] != "pending":
        return False
    now_epoch = float(now_epoch if now_epoch is not None else time.time())
    retry_epoch = _iso_to_epoch(job.get("retry_not_before"))
    return retry_epoch is None or now_epoch >= retry_epoch


def _should_watchdog_fire(job: dict[str, Any], now_epoch: float, latest_output_epoch: float | None) -> bool:
    timeout = int(job.get("watchdog_no_output_s") or 0)
    if timeout <= 0:
        return False
    baseline = latest_output_epoch
    if baseline is None:
        baseline = _iso_to_epoch(job.get("start_ts")) or now_epoch
    return (now_epoch - baseline) >= timeout


def _job_has_restart_budget(job: dict[str, Any]) -> bool:
    return int(job.get("restart_count") or 0) < int(job.get("max_restarts") or 0)


def _close_proc_handles(proc_info: dict[str, Any]) -> None:
    for key in ("stdout_handle", "stderr_handle"):
        handle = proc_info.get(key)
        if handle and not handle.closed:
            handle.close()


def _terminate_proc(proc_info: dict[str, Any]) -> None:
    proc = proc_info["proc"]
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _schedule_retry(job: dict[str, Any], reason: str, now_epoch: float) -> None:
    job["restart_count"] = int(job.get("restart_count") or 0) + 1
    job["status"] = "pending"
    job["error"] = reason
    job["exit_code"] = None
    job["summary_metrics"] = None
    job["wake_reason"] = "restart"
    job["pid"] = None
    job["last_output_ts"] = None
    job["end_ts"] = _epoch_to_iso(now_epoch)
    job["duration_s"] = None
    job["retry_not_before"] = _epoch_to_iso(now_epoch + int(job.get("restart_delay_s") or 0))


def _find_next_cpu_job(status: dict[str, Any], plan: dict[str, Any]) -> str | None:
    if status["lanes"]["cpu"]["active_job_id"] is not None:
        return None
    now_epoch = time.time()
    for stage in plan["cpu_stages"]:
        sid = stage["id"]
        stage_state = status["cpu_stages"][sid]
        if stage_state["status"] in {"blocked", "skipped", "done"}:
            continue
        pending = [job_id for job_id in stage_state["jobs"] if _job_can_start(status["jobs"][job_id], now_epoch)]
        if pending:
            status["lanes"]["cpu"]["current_stage"] = sid
            return pending[0]
        if all(_job_terminal(status["jobs"][job_id]) for job_id in stage_state["jobs"]):
            stage_state["status"] = "done"
            decision_event = _apply_stage_decision(status, stage)
            if decision_event is not None:
                status["last_decision_event"] = {"stage_id": sid, "decision": decision_event}
            return _find_next_cpu_job(status, plan)
    return None


def _find_next_gpu_job(status: dict[str, Any]) -> str | None:
    if status["lanes"]["gpu"]["active_job_id"] is not None:
        return None
    now_epoch = time.time()
    for job_id in status["lanes"]["gpu"]["queue"]:
        if _job_can_start(status["jobs"][job_id], now_epoch):
            return job_id
    return None


def _start_ready_job(job: dict[str, Any], runtime_root: Path, decisions: dict[str, Any]) -> dict[str, Any]:
    job = _resolve_job_record(job, runtime_root, decisions)
    proc_info = _start_job(job)
    job["status"] = "running"
    job["start_ts"] = _now_iso()
    job["end_ts"] = None
    job["duration_s"] = None
    job["exit_code"] = None
    job["error"] = None
    job["wake_reason"] = None
    job["pid"] = int(proc_info["proc"].pid)
    job["retry_not_before"] = None
    job["last_output_ts"] = _epoch_to_iso(_latest_output_mtime(job) or time.time())
    return proc_info


def run_plan(plan_path: Path, runtime_root: Path, dry_run: bool = False) -> tuple[Path, Path]:
    plan = _load_json(plan_path)
    validate_plan(plan)
    runtime_root.mkdir(parents=True, exist_ok=True)
    status = build_initial_status(plan, runtime_root)
    status["state"] = "dry_run" if dry_run else "running"
    status_path, summary_path = _write_status_and_summary(status, runtime_root)
    if dry_run:
        return status_path, summary_path

    watch_cfg = plan["watch"]
    running: dict[str, dict[str, Any]] = {}
    last_heartbeat = time.time()

    while True:
        cpu_job_id = _find_next_cpu_job(status, plan)
        if cpu_job_id:
            job = status["jobs"][cpu_job_id]
            running[cpu_job_id] = _start_ready_job(job, runtime_root, status["decisions"])
            status["lanes"]["cpu"]["active_job_id"] = cpu_job_id
            status_path, summary_path = _write_status_and_summary(status, runtime_root)

        gpu_job_id = _find_next_gpu_job(status)
        if gpu_job_id:
            job = status["jobs"][gpu_job_id]
            running[gpu_job_id] = _start_ready_job(job, runtime_root, status["decisions"])
            status["lanes"]["gpu"]["active_job_id"] = gpu_job_id
            status_path, summary_path = _write_status_and_summary(status, runtime_root)

        if status.get("last_decision_event") is not None:
            event = status.pop("last_decision_event")
            status_path, summary_path = _write_status_and_summary(status, runtime_root)
            _emit_wake(
                runtime_root,
                watch_cfg,
                "decision",
                f"[NIGHTMODE] decision stage={event['stage_id']} kind={event['decision']['kind']} wake=1s",
                status_path,
                summary_path,
            )

        now_epoch = time.time()
        for job_id, proc in list(running.items()):
            job = status["jobs"][job_id]
            latest_mtime = _latest_output_mtime(job)
            if latest_mtime is not None:
                last_epoch = _iso_to_epoch(job.get("last_output_ts"))
                if last_epoch is None or latest_mtime > last_epoch + 1e-6:
                    job["last_output_ts"] = _epoch_to_iso(latest_mtime)
            rc = proc["proc"].poll()
            watchdog_reason = None
            if rc is None:
                if _should_watchdog_fire(job, now_epoch, latest_mtime):
                    watchdog_reason = f"watchdog:no_output>{job['watchdog_no_output_s']}s"
                    _terminate_proc(proc)
                    rc = proc["proc"].poll()
                else:
                    continue
            _close_proc_handles(proc)
            job["end_ts"] = _now_iso()
            if job["start_ts"]:
                start_dt = datetime.fromisoformat(job["start_ts"])
                end_dt = datetime.fromisoformat(job["end_ts"])
                job["duration_s"] = (end_dt - start_dt).total_seconds()
            job["exit_code"] = int(rc) if rc is not None else -9
            ok = False
            err = None
            if rc == 0 and watchdog_reason is None:
                artifact = Path(job["artifact_path"]) if job["artifact_path"] else None
                if artifact and artifact.exists():
                    try:
                        job["summary_metrics"] = _summary_from_artifact(artifact)
                        ok = True
                    except Exception as exc:
                        err = f"artifact parse failed: {exc}"
                elif not job["evidence_required"]:
                    ok = True
                else:
                    err = f"missing artifact: {artifact}"
            elif watchdog_reason is not None:
                err = watchdog_reason
            else:
                err = f"runner exited with code {rc}"
            job["status"] = "done" if ok else "failed"
            job["error"] = err
            if job["lane"] == "cpu":
                status["lanes"]["cpu"]["active_job_id"] = None
            else:
                status["lanes"]["gpu"]["active_job_id"] = None
            del running[job_id]
            status_path, summary_path = _write_status_and_summary(status, runtime_root)
            if ok and job["wake_on_complete"]:
                job["wake_reason"] = "done"
                status_path, summary_path = _write_status_and_summary(status, runtime_root)
                _emit_wake(runtime_root, watch_cfg, "done", f"[NIGHTMODE] done job={job_id} art={job['artifact_path']} wake=1s", status_path, summary_path, job_id=job_id)
            if not ok and _job_has_restart_budget(job):
                _schedule_retry(job, err or "restart", now_epoch)
                status_path, summary_path = _write_status_and_summary(status, runtime_root)
                _emit_wake(
                    runtime_root,
                    watch_cfg,
                    "restart",
                    f"[NIGHTMODE] restart job={job_id} got={err} next=retry wake=1s",
                    status_path,
                    summary_path,
                    job_id=job_id,
                )
                continue
            if not ok and job["stop_on_fail"]:
                status["state"] = "failed"
                status_path, summary_path = _write_status_and_summary(status, runtime_root)
                _emit_wake(runtime_root, watch_cfg, "fail", f"[NIGHTMODE] fail job={job_id} end={err} wake=1s", status_path, summary_path, job_id=job_id)
                return status_path, summary_path

        cpu_done = all(stage_state["status"] in {"done", "skipped"} for stage_state in status["cpu_stages"].values())
        gpu_done = all(status["jobs"][job_id]["status"] in {"done", "failed", "skipped"} for job_id in status["lanes"]["gpu"]["queue"])
        if cpu_done and gpu_done and not running:
            status["state"] = "done"
            status_path, summary_path = _write_status_and_summary(status, runtime_root)
            _emit_wake(runtime_root, watch_cfg, "queue_done", f"[NIGHTMODE] done plan={plan['name']} end=queue_complete wake=1s", status_path, summary_path)
            return status_path, summary_path

        if time.time() - last_heartbeat >= int(watch_cfg["heartbeat_minutes"]) * 60:
            last_heartbeat = time.time()
            _emit_wake(runtime_root, watch_cfg, "heartbeat", f"[NIGHTMODE] goal={plan['name']} got=heartbeat next=continue wake={watch_cfg['heartbeat_minutes']}m", status_path, summary_path)

        time.sleep(2.0)


def watch_plan(runtime_root: Path, follow: bool = False, interval_s: int = 30) -> int:
    status_path = runtime_root / "status.json"
    summary_path = runtime_root / "summary.json"
    if not status_path.exists():
        print(f"Missing status file: {status_path}")
        return 2
    while True:
        status = _load_json(status_path)
        summary = _load_json(summary_path) if summary_path.exists() else {}
        print("=" * 100)
        print(f"plan={status.get('plan_name')} state={status.get('state')} updated={status.get('updated_at')}")
        print(f"running={len(summary.get('running_jobs', []))} done={len(summary.get('completed_jobs', []))} failed={len(summary.get('failed_jobs', []))}")
        if status.get("decisions"):
            print(json.dumps(status["decisions"], indent=2))
        print("=" * 100)
        if not follow or status.get("state") in {"done", "failed", "dry_run"}:
            return 0
        time.sleep(max(1, int(interval_s)))


def launch_plan(plan_path: Path, runtime_root: Path, dry_run: bool = False) -> Path:
    runtime_root.mkdir(parents=True, exist_ok=True)
    stdout_path = runtime_root / "orchestrator.stdout.log"
    stderr_path = runtime_root / "orchestrator.stderr.log"
    launch_path = runtime_root / "launch.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--plan",
        str(plan_path),
        "--runtime-root",
        str(runtime_root),
    ]
    if dry_run:
        cmd.append("--dry-run")
    with open(stdout_path, "a", encoding="utf-8") as stdout_handle, open(stderr_path, "a", encoding="utf-8") as stderr_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
            creationflags=(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
        )
    payload = {
        "timestamp": _now_iso(),
        "pid": int(proc.pid),
        "plan": str(plan_path),
        "runtime_root": str(runtime_root),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "cmd": cmd,
        "dry_run": bool(dry_run),
    }
    _safe_json_write(launch_path, payload)
    return launch_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Overnight orchestrator for canonical nightly runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Execute queue plan")
    run_p.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    run_p.add_argument("--runtime-root", type=Path, default=None)
    run_p.add_argument("--dry-run", action="store_true")
    launch_p = sub.add_parser("launch", help="Launch queue plan detached in background")
    launch_p.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    launch_p.add_argument("--runtime-root", type=Path, default=None)
    launch_p.add_argument("--dry-run", action="store_true")
    watch_p = sub.add_parser("watch", help="Inspect runtime root")
    watch_p.add_argument("--runtime-root", type=Path, required=True)
    watch_p.add_argument("--follow", action="store_true")
    watch_p.add_argument("--interval-s", type=int, default=30)
    args = parser.parse_args()

    if args.command == "watch":
        return watch_plan(args.runtime_root, follow=args.follow, interval_s=args.interval_s)

    plan = _load_json(args.plan)
    runtime_root = args.runtime_root or _default_runtime_root(_slug(plan["name"]))
    if args.command == "launch":
        launch_path = launch_plan(args.plan, runtime_root, dry_run=args.dry_run)
        print(f"LAUNCH runtime: {runtime_root}")
        print(f"LAUNCH metadata: {launch_path}")
        return 0
    status_path, summary_path = run_plan(args.plan, runtime_root, dry_run=args.dry_run)
    label = "DRY-RUN" if args.dry_run else "RUN"
    print(f"{label} status: {status_path}")
    print(f"{label} summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
