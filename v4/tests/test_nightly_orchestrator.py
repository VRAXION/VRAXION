from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from nightly_orchestrator import (  # type: ignore[import-not-found]
    DEFAULT_PLAN_PATH,
    _emit_wake,
    _find_next_gpu_job,
    _resolve_variant,
    build_initial_status,
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
