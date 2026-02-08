#!/usr/bin/env python3
"""Asynchronous CPU catch-up evaluator for VRA-79 checkpoints.

Design:
- Poll the training checkpoint and snapshot it whenever a newer step appears.
- Queue a single scientific eval stream at an adaptive step cadence.
- Evaluate snapshots on CPU without blocking the GPU training process.
- Write append-only `eval_stream.jsonl` and `eval_stream.csv`.
- Emit live catch-up metrics for dashboard progress and backlog status.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import torch


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
EVAL_TOOL = REPO_ROOT / r"Golden Draft\tools\eval_ckpt_assoc_byte.py"
Z95 = 1.959963984540054
Z99 = 2.5758293035489004


@dataclass(frozen=True)
class EvalJob:
    step: int
    eval_n: int
    split: str
    snapshot_path: Path

    @property
    def key(self) -> tuple[int, int, str]:
        return (int(self.step), int(self.eval_n), str(self.split))


@dataclass
class RuntimeState:
    latest_seen_step: int = 0
    latest_eval_step: int = 0
    current_stride: int = 10
    avg_eval_seconds: float = 0.0
    worker_busy: bool = False
    current_eval_step: int = 0
    current_eval_n: int = 0
    current_eval_started_ts: float = 0.0
    recover_ticks: int = 0
    stop_seen: bool = False
    jobs_enqueued_total: int = 0
    jobs_done_total: int = 0


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run asynchronous CPU eval catch-up for VRA-79 checkpoints.")
    ap.add_argument("--run-root", required=True, help="Run root containing train/ checkpoint artifacts.")
    ap.add_argument("--poll-s", type=float, default=2.0, help="Checkpoint polling interval.")
    ap.add_argument("--eval-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--heartbeat-s", type=int, default=20)
    ap.add_argument("--force-eval-disjoint", type=int, default=1, choices=[0, 1])
    ap.add_argument("--force-eval-subset", type=int, default=0, choices=[0, 1])
    ap.add_argument("--stop-on-file", default="", help="Optional sentinel file. Worker exits after queue drain.")
    ap.add_argument("--max-step", type=int, default=0, help="Optional hard stop once latest step reaches this.")
    ap.add_argument("--idle-timeout-s", type=int, default=0, help="Optional idle timeout when no new checkpoints.")
    ap.add_argument("--eval-every", type=int, default=10, help="Base eval stride in training steps.")
    # Safe default: keep per-eval wallclock small on CPU for large hallway models.
    ap.add_argument("--eval-n", type=int, default=32, help="Single-stream eval samples.")
    ap.add_argument("--anchor-steps", default="", help="Optional comma list of extra anchor steps.")
    ap.add_argument("--anchor-eval-n", type=int, default=0, help="Optional eval_n for anchor steps.")
    ap.add_argument("--adaptive-mode", type=int, default=1, choices=[0, 1])
    ap.add_argument("--queue-max-depth", type=int, default=64)
    ap.add_argument("--backlog-soft-s", type=int, default=1200)
    ap.add_argument("--backlog-hard-s", type=int, default=2400)
    ap.add_argument("--soft-stride-mult", type=int, default=2)
    ap.add_argument("--hard-stride-mult", type=int, default=4)
    ap.add_argument("--recover-low-ratio", type=float, default=0.50)
    ap.add_argument("--recover-ticks", type=int, default=3)
    ap.add_argument(
        "--status-path",
        default="",
        help="Optional status json output path (defaults to run_root/eval_catchup_status.json).",
    )
    return ap.parse_args()


def _wilson_bounds(successes: float, n: int, z: float) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p_hat = max(0.0, min(1.0, float(successes)))
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(n))) / float(n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def _parse_int_set(raw: str) -> set[int]:
    out: set[int] = set()
    for chunk in str(raw).split(","):
        txt = chunk.strip()
        if not txt:
            continue
        try:
            value = int(txt)
        except Exception as exc:
            raise ValueError(f"invalid integer list entry: {txt}") from exc
        if value > 0:
            out.add(value)
    return out


def _jsonl_keys(path: Path) -> set[tuple[int, int, str]]:
    keys: set[tuple[int, int, str]] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            step = int(row.get("step", 0))
            eval_n = int(row.get("eval_n", 0) or 0)
            split = str(row.get("split", ""))
            if step > 0:
                keys.add((step, eval_n, split))
        except Exception:
            continue
    return keys


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})


def _pick_checkpoint(train_root: Path) -> Path | None:
    last_good = train_root / "checkpoint_last_good.pt"
    if last_good.exists():
        return last_good
    plain = train_root / "checkpoint.pt"
    if plain.exists():
        return plain
    return None


def _torch_load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def _read_checkpoint_step(path: Path) -> int | None:
    try:
        payload = _torch_load_checkpoint(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    step = payload.get("step")
    if isinstance(step, (int, float)):
        return int(step)
    return None


def _copy_stable_snapshot(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        stat_before = src.stat()
        shutil.copy2(src, tmp)
        stat_after = src.stat()
        if stat_before.st_size != stat_after.st_size or stat_before.st_mtime_ns != stat_after.st_mtime_ns:
            if tmp.exists():
                tmp.unlink()
            return False
        # Load once to ensure we did not snapshot a partially written checkpoint.
        _ = _torch_load_checkpoint(tmp)
        os.replace(tmp, dst)
        return True
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        return False


def _compute_queue_eta_s(queue_depth: int, state: RuntimeState) -> float:
    inflight = 1 if state.worker_busy else 0
    avg_eval = float(state.avg_eval_seconds) if float(state.avg_eval_seconds) > 0.0 else 0.0
    return float(queue_depth + inflight) * avg_eval


def _adaptive_stride(args: argparse.Namespace, state: RuntimeState, queue_depth: int) -> tuple[int, str]:
    base_stride = max(1, int(args.eval_every))
    if int(args.adaptive_mode) == 0:
        state.current_stride = base_stride
        return base_stride, "fixed"

    queue_eta = _compute_queue_eta_s(queue_depth, state)
    soft = max(1, int(args.backlog_soft_s))
    hard = max(soft + 1, int(args.backlog_hard_s))
    soft_stride = max(base_stride, int(base_stride * max(1, int(args.soft_stride_mult))))
    hard_stride = max(soft_stride, int(base_stride * max(1, int(args.hard_stride_mult))))

    mode = "base"
    if queue_eta > float(hard):
        state.current_stride = hard_stride
        state.recover_ticks = 0
        mode = "hard"
    elif queue_eta > float(soft):
        state.current_stride = soft_stride
        state.recover_ticks = 0
        mode = "soft"
    else:
        low_cut = float(soft) * max(0.05, float(args.recover_low_ratio))
        if queue_eta <= low_cut:
            state.recover_ticks += 1
        else:
            state.recover_ticks = 0
        if state.recover_ticks >= max(1, int(args.recover_ticks)):
            state.current_stride = base_stride
            mode = "recovered"
        else:
            mode = "holding"
    return max(1, int(state.current_stride)), mode


def _write_status(path: Path, state: RuntimeState, queue_depth: int, mode: str) -> None:
    catchup_pct = 0.0
    if int(state.latest_seen_step) > 0:
        catchup_pct = (100.0 * float(state.latest_eval_step)) / float(state.latest_seen_step)
    inflight = 1 if state.worker_busy else 0
    queue_total = int(queue_depth) + int(inflight) + max(0, int(state.jobs_done_total))
    queue_catchup_pct = 0.0
    if queue_total > 0:
        queue_catchup_pct = (100.0 * float(state.jobs_done_total)) / float(queue_total)
    current_elapsed = 0.0
    if bool(state.worker_busy) and float(state.current_eval_started_ts) > 0.0:
        current_elapsed = max(0.0, float(time.time() - float(state.current_eval_started_ts)))
    payload = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latest_train_step": int(state.latest_seen_step),
        "latest_eval_step": int(state.latest_eval_step),
        "queue_depth": int(queue_depth),
        "avg_eval_seconds": float(state.avg_eval_seconds),
        "queue_eta_sec": float(_compute_queue_eta_s(queue_depth, state)),
        "current_stride": int(state.current_stride),
        "adaptive_mode": str(mode),
        "eval_catchup_pct": max(0.0, min(100.0, float(catchup_pct))),
        "queue_catchup_pct": max(0.0, min(100.0, float(queue_catchup_pct))),
        "jobs_enqueued_total": int(state.jobs_enqueued_total),
        "jobs_done_total": int(state.jobs_done_total),
        "worker_busy": bool(state.worker_busy),
        "current_eval_step": int(state.current_eval_step),
        "current_eval_n": int(state.current_eval_n),
        "current_eval_elapsed_s": float(current_elapsed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _start_busy_status_heartbeat(
    status_path: Path,
    state: RuntimeState,
    jobs: Queue[EvalJob],
    lock: threading.Lock,
    heartbeat_s: int,
) -> tuple[threading.Event, threading.Thread]:
    interval_s = max(1, int(heartbeat_s))
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval_s):
            with lock:
                _write_status(status_path, state, jobs.qsize(), mode="eval_running")

    th = threading.Thread(target=_loop, name="vra79-catchup-status-heartbeat", daemon=True)
    th.start()
    return stop, th


def _run_eval(job: EvalJob, args: argparse.Namespace, train_root: Path) -> tuple[int, float, str]:
    cmd = [
        sys.executable,
        str(EVAL_TOOL),
        "--run-root",
        str(train_root),
        "--checkpoint",
        str(job.snapshot_path),
        "--eval-samples",
        str(int(job.eval_n)),
        "--batch-size",
        str(int(args.eval_batch_size)),
        "--device",
        str(args.eval_device),
        "--heartbeat-s",
        str(int(args.heartbeat_s)),
    ]
    if int(args.force_eval_disjoint):
        cmd.append("--force-eval-disjoint")
    if int(args.force_eval_subset):
        cmd.append("--force-eval-subset")
    start = time.time()
    run_kwargs: dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "capture_output": True,
        "text": True,
    }
    if os.name == "nt":
        run_kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    cp = subprocess.run(cmd, **run_kwargs)
    elapsed = float(time.time() - start)
    output = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
    return int(cp.returncode), elapsed, output.strip()


def _load_report(train_root: Path) -> dict[str, Any] | None:
    report_path = train_root / "report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _poller(
    args: argparse.Namespace,
    train_root: Path,
    snapshots_dir: Path,
    jobs: Queue[EvalJob],
    stop_event: threading.Event,
    state: RuntimeState,
    seen: set[tuple[int, int, str]],
    enqueued: set[tuple[int, int, str]],
    lock: threading.Lock,
    anchor_steps: set[int],
    anchor_eval_n: int,
) -> None:
    split = "disjoint" if int(args.force_eval_disjoint) else "subset"
    while not stop_event.is_set():
        if args.stop_on_file and Path(args.stop_on_file).exists():
            state.stop_seen = True
            stop_event.set()
            break
        checkpoint_path = _pick_checkpoint(train_root)
        if checkpoint_path is None:
            time.sleep(max(0.5, float(args.poll_s)))
            continue
        step = _read_checkpoint_step(checkpoint_path)
        if step is None or step <= int(state.latest_seen_step):
            time.sleep(max(0.5, float(args.poll_s)))
            continue
        snapshot = snapshots_dir / f"checkpoint_step{int(step):07d}_eval.pt"
        if not snapshot.exists():
            ok = _copy_stable_snapshot(checkpoint_path, snapshot)
            if not ok:
                time.sleep(max(0.5, float(args.poll_s)))
                continue
        state.latest_seen_step = int(step)
        skip_all = True
        queue_full = False
        enq_count = 0
        stride = max(1, int(state.current_stride))
        mode = "unknown"
        candidates: list[EvalJob] = []
        with lock:
            stride, mode = _adaptive_stride(args, state, jobs.qsize())
            queue_depth = jobs.qsize()
            if int(step) % int(stride) == 0:
                candidates.append(
                    EvalJob(
                        step=int(step),
                        eval_n=int(args.eval_n),
                        split=split,
                        snapshot_path=snapshot,
                    )
                )
            if int(anchor_eval_n) > 0 and int(step) in anchor_steps and int(anchor_eval_n) != int(args.eval_n):
                candidates.append(
                    EvalJob(
                        step=int(step),
                        eval_n=int(anchor_eval_n),
                        split=split,
                        snapshot_path=snapshot,
                    )
                )
            compat_keys = {(int(step), 0, split)}
            for job in candidates:
                key = job.key
                if key in seen or key in enqueued:
                    continue
                if any(k in seen for k in compat_keys):
                    continue
                if int(args.queue_max_depth) > 0 and int(jobs.qsize()) >= int(args.queue_max_depth):
                    queue_full = True
                    continue
                jobs.put(job)
                enqueued.add(key)
                state.jobs_enqueued_total = int(state.jobs_enqueued_total) + 1
                enq_count += 1
            skip_all = enq_count <= 0
        if not candidates:
            time.sleep(max(0.5, float(args.poll_s)))
            continue
        if skip_all:
            if queue_full:
                _log(
                    f"queue full depth={jobs.qsize()} max={args.queue_max_depth}; "
                    f"skip enqueue step={step} stride={stride} mode={mode}"
                )
            time.sleep(max(0.5, float(args.poll_s)))
            continue
        _log(
            f"queued eval step={step} added={enq_count} split={split} "
            f"stride={stride} mode={mode}"
        )
        time.sleep(max(0.5, float(args.poll_s)))


def main() -> int:
    args = _parse_args()
    if int(args.force_eval_disjoint) and int(args.force_eval_subset):
        raise SystemExit("cannot pass both --force-eval-disjoint=1 and --force-eval-subset=1")
    run_root = Path(args.run_root).resolve()
    train_root = run_root / "train"
    if not EVAL_TOOL.exists():
        raise FileNotFoundError(f"Missing eval tool: {EVAL_TOOL}")
    if not train_root.exists():
        raise FileNotFoundError(f"Missing train root: {train_root}")

    split = "disjoint" if int(args.force_eval_disjoint) else "subset"
    anchor_steps = _parse_int_set(args.anchor_steps)
    anchor_eval_n = int(args.anchor_eval_n)
    jsonl_path = run_root / "eval_stream.jsonl"
    csv_path = run_root / "eval_stream.csv"
    status_path = Path(args.status_path).resolve() if args.status_path else (run_root / "eval_catchup_status.json")
    snapshots_dir = train_root / "eval_snapshots"
    state = RuntimeState(
        latest_seen_step=0,
        latest_eval_step=0,
        current_stride=max(1, int(args.eval_every)),
        avg_eval_seconds=0.0,
        worker_busy=False,
        recover_ticks=0,
        stop_seen=False,
    )
    jobs: Queue[EvalJob] = Queue()
    stop_event = threading.Event()
    lock = threading.Lock()
    seen = _jsonl_keys(jsonl_path)
    enqueued: set[tuple[int, int, str]] = set()
    idle_since = time.time()

    _log(f"run_root={run_root}")
    _log(f"train_root={train_root}")
    _log(
        "single_stream "
        f"base_stride={args.eval_every} "
        f"eval_n={args.eval_n} "
        f"anchor_eval_n={anchor_eval_n if anchor_eval_n > 0 else 'off'} "
        f"anchor_steps={sorted(anchor_steps) if anchor_steps else []} "
        f"adaptive={args.adaptive_mode} "
        f"soft={args.backlog_soft_s}s hard={args.backlog_hard_s}s "
        f"split={split}"
    )
    if args.stop_on_file:
        _log(f"stop_on_file={Path(args.stop_on_file).resolve()}")
    _log(f"status_path={status_path}")
    _write_status(status_path, state, queue_depth=0, mode="startup")

    thread = threading.Thread(
        target=_poller,
        name="vra79-catchup-poller",
        kwargs={
            "args": args,
            "train_root": train_root,
            "snapshots_dir": snapshots_dir,
            "jobs": jobs,
            "stop_event": stop_event,
            "state": state,
            "seen": seen,
            "enqueued": enqueued,
            "lock": lock,
            "anchor_steps": anchor_steps,
            "anchor_eval_n": anchor_eval_n,
        },
        daemon=True,
    )
    thread.start()

    fields = [
        "utc",
        "step",
        "eval_n",
        "split",
        "eval_acc",
        "eval_loss",
        "chance_acc",
        "acc_delta",
        "ci95_low",
        "ci95_high",
        "ci99_low",
        "ci99_high",
        "eval_seconds",
        "eval_rc",
        "checkpoint_snapshot",
        "report_path",
    ]

    try:
        while True:
            if int(args.max_step) > 0 and int(state.latest_seen_step) >= int(args.max_step) and jobs.empty():
                _log(f"max_step reached ({state.latest_seen_step} >= {args.max_step}); exiting")
                stop_event.set()
                _write_status(status_path, state, jobs.qsize(), mode="max_step_exit")
                break
            if stop_event.is_set() and jobs.empty():
                _log("stop requested and queue drained; exiting")
                _write_status(status_path, state, jobs.qsize(), mode="stop_and_drain_exit")
                break
            try:
                job = jobs.get(timeout=1.0)
            except Empty:
                with lock:
                    _adaptive_stride(args, state, jobs.qsize())
                    _write_status(status_path, state, jobs.qsize(), mode="idle")
                if int(args.idle_timeout_s) > 0:
                    if jobs.empty():
                        idle_elapsed = time.time() - idle_since
                        if idle_elapsed >= int(args.idle_timeout_s):
                            _log(f"idle timeout reached ({int(idle_elapsed)}s); exiting")
                            stop_event.set()
                            _write_status(status_path, state, jobs.qsize(), mode="idle_timeout_exit")
                            break
                continue

            idle_since = time.time()
            _log(f"eval start step={job.step} n={job.eval_n} split={job.split}")
            state.worker_busy = True
            state.current_eval_step = int(job.step)
            state.current_eval_n = int(job.eval_n)
            state.current_eval_started_ts = float(time.time())
            with lock:
                _write_status(status_path, state, jobs.qsize(), mode="eval_start")
            hb_stop, hb_thread = _start_busy_status_heartbeat(
                status_path=status_path,
                state=state,
                jobs=jobs,
                lock=lock,
                heartbeat_s=int(args.heartbeat_s),
            )
            try:
                rc, elapsed_s, output = _run_eval(job, args, train_root)
            except Exception as exc:
                rc = 999
                elapsed_s = max(0.0, float(time.time() - float(state.current_eval_started_ts)))
                output = f"eval lane exception: {exc}"
            finally:
                hb_stop.set()
                hb_thread.join(timeout=1.0)
                state.worker_busy = False
                state.current_eval_step = 0
                state.current_eval_n = 0
                state.current_eval_started_ts = 0.0
            report = _load_report(train_root)
            with lock:
                enqueued.discard(job.key)
            jobs.task_done()

            if report is None:
                _log(f"eval missing report step={job.step} n={job.eval_n} rc={rc}")
                if output:
                    _log(output.splitlines()[-1][:240])
                with lock:
                    _adaptive_stride(args, state, jobs.qsize())
                    _write_status(status_path, state, jobs.qsize(), mode="missing_report")
                continue

            ev = report.get("eval") or {}
            settings = report.get("settings") or {}
            eval_n = int(ev.get("eval_n") or job.eval_n)
            eval_acc = float(ev.get("eval_acc") or 0.0)
            eval_loss = float(ev.get("eval_loss") or 0.0)
            val_range = settings.get("val_range")
            chance_acc = None
            acc_delta = None
            if isinstance(val_range, (int, float)) and float(val_range) > 0.0:
                chance_acc = 1.0 / float(val_range)
                acc_delta = float(eval_acc) - float(chance_acc)
            ci95_low, ci95_high = _wilson_bounds(eval_acc, eval_n, Z95)
            ci99_low, ci99_high = _wilson_bounds(eval_acc, eval_n, Z99)

            row = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "step": int(job.step),
                "eval_n": int(eval_n),
                "split": str(job.split),
                "eval_acc": float(eval_acc),
                "eval_loss": float(eval_loss),
                "chance_acc": chance_acc,
                "acc_delta": acc_delta,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "ci99_low": ci99_low,
                "ci99_high": ci99_high,
                "eval_seconds": float(elapsed_s),
                "eval_rc": int(rc),
                "checkpoint_snapshot": str(job.snapshot_path),
                "report_path": str(train_root / "report.json"),
            }
            _append_jsonl(jsonl_path, row)
            _append_csv(csv_path, row, fields)
            seen.add(job.key)
            state.latest_eval_step = max(int(state.latest_eval_step), int(job.step))
            state.jobs_done_total = int(state.jobs_done_total) + 1
            if float(elapsed_s) > 0.0:
                if float(state.avg_eval_seconds) <= 0.0:
                    state.avg_eval_seconds = float(elapsed_s)
                else:
                    state.avg_eval_seconds = (0.8 * float(state.avg_eval_seconds)) + (0.2 * float(elapsed_s))
            with lock:
                stride, mode = _adaptive_stride(args, state, jobs.qsize())
                _write_status(status_path, state, jobs.qsize(), mode=f"post_eval:{mode}")
            _log(
                f"eval done step={job.step} n={eval_n} acc={eval_acc:.6f} "
                f"delta={acc_delta if acc_delta is not None else 'na'} rc={rc} "
                f"t={elapsed_s:.1f}s stride={stride} queue={jobs.qsize()}"
            )
            if output and rc != 0:
                _log(output.splitlines()[-1][:240])

    except KeyboardInterrupt:
        _log("keyboard interrupt; shutting down")
        stop_event.set()
        _write_status(status_path, state, jobs.qsize(), mode="keyboard_interrupt_exit")

    thread.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
