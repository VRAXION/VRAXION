from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from nightly_orchestrator import (  # type: ignore[import-not-found]
    DEFAULT_PLAN_PATH,
    _build_runner_cmd,
    _latest_output_mtime,
    _job_can_start,
    _job_has_restart_budget,
    _should_watchdog_fire,
    _emit_wake,
    _find_next_gpu_job,
    _resolve_variant,
    build_initial_status,
    launch_plan,
    run_plan,
    validate_plan,
)


def test_default_plan_validates():
    import json

    with open(DEFAULT_PLAN_PATH, encoding="utf-8") as f:
        plan = json.load(f)
    validate_plan(plan)


def test_build_initial_status_separates_cpu_stage_and_gpu_queue(tmp_path):
    import json

    with open(DEFAULT_PLAN_PATH, encoding="utf-8") as f:
        plan = json.load(f)
    status = build_initial_status(plan, tmp_path)

    assert status["cpu_stages"]["seed_validation"]["enabled"] is True
    assert status["cpu_stages"]["lag_selection"]["enabled"] is False
    assert status["jobs"]["cpu_ll_s42"]["status"] == "pending"
    assert status["jobs"]["cpu_lag_llt4_s42"]["status"] == "blocked"
    assert status["jobs"]["gpu_llt6_long"]["status"] == "pending"


def test_find_next_gpu_job_is_single_flight(tmp_path):
    import json

    with open(DEFAULT_PLAN_PATH, encoding="utf-8") as f:
        plan = json.load(f)
    status = build_initial_status(plan, tmp_path)
    first = _find_next_gpu_job(status)
    assert first == "gpu_llt6_long"
    status["lanes"]["gpu"]["active_job_id"] = first
    assert _find_next_gpu_job(status) is None


def test_resolve_variant_from_decision():
    job = {"id": "replay", "variant_from_decision": "lag_selection.lag_winner_variant"}
    decisions = {"lag_selection": {"lag_winner_variant": "LLT7"}}
    assert _resolve_variant(job, decisions) == "LLT7"


def test_emit_wake_writes_json(tmp_path):
    status_path = tmp_path / "status.json"
    summary_path = tmp_path / "summary.json"
    status_path.write_text("{}", encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")
    watch_cfg = {
        "wake_file": "wake_trigger.json",
        "ping_enabled": False,
        "heartbeat_minutes": 60,
    }
    _emit_wake(tmp_path, watch_cfg, "heartbeat", "hello", status_path, summary_path, job_id="job-1")
    wake_path = tmp_path / "wake_trigger.json"
    assert wake_path.exists()


def test_run_plan_dry_run_writes_status_and_summary(tmp_path):
    status_path, summary_path = run_plan(DEFAULT_PLAN_PATH, tmp_path, dry_run=True)
    assert status_path.exists()
    assert summary_path.exists()


def test_lane_defaults_are_applied_to_jobs(tmp_path):
    import json

    with open(DEFAULT_PLAN_PATH, encoding="utf-8") as f:
        plan = json.load(f)
    status = build_initial_status(plan, tmp_path)

    cpu_job = status["jobs"]["cpu_ll_s42"]
    gpu_job = status["jobs"]["gpu_llt6_long"]
    assert cpu_job["max_restarts"] >= 1
    assert cpu_job["watchdog_no_output_s"] > 0
    assert cpu_job["env"]["OMP_NUM_THREADS"] == "8"
    assert gpu_job["max_restarts"] >= 1
    assert gpu_job["watchdog_no_output_s"] > 0


def test_job_can_start_honors_retry_not_before():
    job = {"status": "pending", "retry_not_before": "2100-01-01T00:00:00+00:00"}
    assert _job_can_start(job, now_epoch=1.0) is False
    job["retry_not_before"] = None
    assert _job_can_start(job, now_epoch=1.0) is True


def test_watchdog_helper_triggers_after_timeout():
    job = {"watchdog_no_output_s": 10, "start_ts": "2026-03-07T00:00:00+00:00", "last_output_ts": None}
    assert _should_watchdog_fire(job, now_epoch=10.0, latest_output_epoch=5.0) is False
    assert _should_watchdog_fire(job, now_epoch=20.5, latest_output_epoch=5.0) is True


def test_restart_budget_helper():
    job = {"restart_count": 0, "max_restarts": 1}
    assert _job_has_restart_budget(job) is True
    job["restart_count"] = 1
    assert _job_has_restart_budget(job) is False


def test_launch_plan_writes_metadata(tmp_path):
    launch_path = launch_plan(DEFAULT_PLAN_PATH, tmp_path, dry_run=True)
    assert launch_path.exists()


def test_build_runner_cmd_uses_unbuffered_python():
    job = {
        "surface": "wikitext_sequential_carry",
        "variant_resolved": "LL",
        "steps": 10,
        "device": "cpu",
        "seed": 42,
        "heartbeat_path": "heartbeat.json",
        "artifact_path": "out.json",
    }
    cmd = _build_runner_cmd(job)
    assert cmd[1] == "-u"
    assert "--heartbeat-out" in cmd


def test_latest_output_prefers_heartbeat(tmp_path):
    hb = tmp_path / "heartbeat.json"
    out = tmp_path / "stdout.log"
    err = tmp_path / "stderr.log"
    out.write_text("out", encoding="utf-8")
    err.write_text("err", encoding="utf-8")
    hb.write_text("hb", encoding="utf-8")
    job = {
        "heartbeat_path": str(hb),
        "stdout_log": str(out),
        "stderr_log": str(err),
    }
    assert _latest_output_mtime(job) is not None
