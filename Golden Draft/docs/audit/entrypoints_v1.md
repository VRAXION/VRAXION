# Entrypoints v1

This document is a pragmatic map of "things you can run" in this repo (scripts and common commands).
It is intentionally conservative: if an entrypoint is likely to start heavy GPU work immediately, it is marked **unsafe to run blindly**.

Legend:
- **Golden Code**: code intended to be reusable / closer to "production".
- **Golden Draft**: experiments + tooling (fast iteration, but still expected to be reproducible).

## Safe entrypoints (recommended for first contact)

| Name | Path / Command | What it does | Expected outputs |
|---|---|---|---|
| Unit tests (CPU) | `python -m unittest discover -s "Golden Draft/tests" -v` | Runs CPU-only regression tests for tooling/contracts. | Console output (no files). |
| Bytecode sanity | `python -m compileall "Golden Code" "Golden Draft"` | Catches obvious syntax/import issues. | Console output (no files). |
| GPU env dump (VRA-29) | `python "Golden Draft/tools/gpu_env_dump.py" --out-dir <DIR> --precision fp16 --amp 1` | Writes a stable `env.json` describing GPU/runtime context (best-effort; works without CUDA). | `<DIR>/env.json` |
| Workload ID | `python "Golden Draft/tools/workload_id.py" <workload_json>` | Canonicalizes a workload spec and prints `workload_id`. | Console output (or JSON with `--json`). |
| GPU capacity probe (VRA-32) | `python "Golden Draft/tools/gpu_capacity_probe.py" --help` | Probe harness for throughput/VRAM/stability metrics. Use `--help` first; runs write artifacts and are overwrite-guarded. | `env.json`, `metrics.json`, `metrics.csv`, `summary.md`, `run_cmd.txt` |
| Lab supervisor / watchdog | `python "Golden Draft/tools/vraxion_lab_supervisor.py" --help` | Supervises long runs (heartbeat/watchdog + optional wake trigger). | Job logs + watchdog artifacts (depends on args). |

## Potentially heavy entrypoints (read first)

| Name | Path | Notes (read before running) |
|---|---|---|
| VRAXION runner | `Golden Draft/vraxion_run.py` | **Unsafe to run blindly.** Appears to start a run immediately (may default to CUDA). Read the file header and config expectations first. |
| Infinite runner | `Golden Draft/VRAXION_INFINITE.py` | **Unsafe to run blindly.** Intended for long-running loops / nightmode-style work. Read first. |
