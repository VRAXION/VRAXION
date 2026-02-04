"""GPU/OS environment dump for reproducible VRAXION benchmarks.

Non-negotiable contract (VRA-29):
- Never assumes CUDA, torch, nvidia-smi, or git are present.
- Always writes env.json with a stable schema (v1) and exits 0 on success.
- JSON output is ASCII-safe (ensure_ascii=True) and atomically written.
- Missing facts are encoded as null plus an entry in errors[] (best-effort).

This file intentionally stays stdlib-only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ENV_SCHEMA_VERSION = "v1"
TOOL_VERSION = "gpu_env_dump/1.0"

# Keep output stable and machine-checkable (exact key set for v1).
ENV_KEYS_V1: Tuple[str, ...] = (
    "env_schema_version",
    "tool_version",
    "generated_utc",
    "os",
    "python_version",
    "torch_version",
    "cuda_version",
    "precision",
    "amp",
    "gpu_name",
    "total_vram_bytes",
    "compute_capability",
    "driver_version",
    "cuda_driver_reported",
    "driver_model",
    "nvidia_smi_query",
    "nvidia_smi_raw",
    "git_commit",
    "git_dirty",
    "git_error",
    "env_vars",
    "errors",
)

ENV_VAR_WHITELIST: Tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _repo_root() -> Path:
    # .../VRAXION_DEV/Golden Draft/tools/gpu_env_dump.py -> repo root
    return Path(__file__).resolve().parents[2]


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 32] + "\n...<truncated>...\n"


def _run(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    timeout_s: float = 2.0,
) -> Tuple[int, str, str, Optional[str]]:
    """Run a subprocess safely and return (rc, stdout, stderr, error_str)."""

    try:
        r = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return r.returncode, r.stdout, r.stderr, None
    except Exception as e:
        return 999, "", "", f"{type(e).__name__}: {e}"


def _collect_torch(env: Dict[str, Any], errors: List[str]) -> None:
    try:
        torch = importlib.import_module("torch")
    except Exception as e:
        errors.append(f"torch_import_failed: {type(e).__name__}: {e}")
        return

    env["torch_version"] = getattr(torch, "__version__", None)

    try:
        cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
        if isinstance(cuda_ver, str) and cuda_ver.strip():
            env["cuda_version"] = cuda_ver.strip()
    except Exception as e:
        errors.append(f"torch_cuda_version_failed: {type(e).__name__}: {e}")

    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return
        if not bool(cuda.is_available()):
            return
        props = cuda.get_device_properties(0)
        env["gpu_name"] = getattr(props, "name", None)
        total = getattr(props, "total_memory", None)
        env["total_vram_bytes"] = int(total) if total is not None else None
        major = getattr(props, "major", None)
        minor = getattr(props, "minor", None)
        if major is not None and minor is not None:
            env["compute_capability"] = f"{major}.{minor}"
    except Exception as e:
        errors.append(f"torch_cuda_props_failed: {type(e).__name__}: {e}")


def _parse_nvidia_smi_header(raw: str) -> Tuple[Optional[str], Optional[str]]:
    # Typical header: "Driver Version: 551.23  CUDA Version: 12.4"
    drv = None
    cuda = None
    m = re.search(r"Driver Version:\s*([0-9.]+)", raw)
    if m:
        drv = m.group(1)
    m = re.search(r"CUDA Version:\s*([0-9.]+)", raw)
    if m:
        cuda = m.group(1)
    return drv, cuda


def _collect_nvidia_smi(env: Dict[str, Any], errors: List[str]) -> None:
    if shutil.which("nvidia-smi") is None:
        errors.append("nvidia_smi_not_found")
        return

    # Always capture raw output (best-effort). Helpful for postmortems.
    rc, out, err, run_err = _run(["nvidia-smi"], timeout_s=3.0)
    raw = out if out.strip() else err
    if run_err is not None:
        errors.append(f"nvidia_smi_raw_failed: {run_err}")
    elif rc != 0:
        errors.append(f"nvidia_smi_raw_nonzero_rc:{rc}")
    if raw:
        env["nvidia_smi_raw"] = _truncate(raw, 65536)
        drv, cuda = _parse_nvidia_smi_header(raw)
        if env["driver_version"] is None and drv is not None:
            env["driver_version"] = drv
        if cuda is not None:
            env["cuda_driver_reported"] = cuda
            if env["cuda_version"] is None:
                # When torch isn't available, fall back to driver-reported CUDA version.
                env["cuda_version"] = cuda

        # Best-effort driver model parse (WDDM/TCC). Not guaranteed.
        if env["driver_model"] is None:
            m = re.search(r"\b(WDDM|TCC)\b", raw)
            if m:
                env["driver_model"] = m.group(1)

    # Structured query (best-effort). Try a richer set first, then fall back.
    query_variants = [
        "driver_version,name,memory.total,compute_cap,driver_model.current",
        "driver_version,name,memory.total,compute_cap",
    ]
    for q in query_variants:
        rc, qout, qerr, run_err = _run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            timeout_s=3.0,
        )
        if run_err is not None:
            errors.append(f"nvidia_smi_query_failed({q}): {run_err}")
            continue
        if rc != 0:
            errors.append(f"nvidia_smi_query_nonzero_rc({q}):{rc}")
            continue
        if not qout.strip():
            errors.append(f"nvidia_smi_query_empty({q})")
            continue

        try:
            rows = list(csv.reader(qout.splitlines()))
            if not rows:
                errors.append(f"nvidia_smi_query_parse_empty({q})")
                continue
            # We only need the first GPU for a single-machine env lock.
            row = rows[0]
            fields = q.split(",")
            if len(row) != len(fields):
                errors.append(f"nvidia_smi_query_field_mismatch({q}): got {len(row)} expected {len(fields)}")
                continue

            d: Dict[str, Any] = {}
            for k, v in zip(fields, row):
                d[k] = v.strip()

            env["nvidia_smi_query"] = d

            if env["driver_version"] is None:
                env["driver_version"] = d.get("driver_version") or None
            if env["gpu_name"] is None:
                env["gpu_name"] = d.get("name") or None
            if env["compute_capability"] is None:
                env["compute_capability"] = d.get("compute_cap") or None

            # memory.total is reported in MiB when nounits is used.
            if env["total_vram_bytes"] is None:
                mem_mib = d.get("memory.total")
                if mem_mib is not None and mem_mib.strip():
                    try:
                        env["total_vram_bytes"] = int(float(mem_mib) * 1024 * 1024)
                    except ValueError:
                        errors.append(f"nvidia_smi_parse_memory_total_failed:{mem_mib!r}")

            if env["driver_model"] is None:
                env["driver_model"] = d.get("driver_model.current") or None

            # First successful structured query wins.
            break
        except Exception as e:
            errors.append(f"nvidia_smi_query_parse_failed({q}): {type(e).__name__}: {e}")


def _collect_git(env: Dict[str, Any], errors: List[str]) -> None:
    if shutil.which("git") is None:
        env["git_error"] = "git_not_found"
        errors.append("git_not_found")
        return

    root = _repo_root()
    rc, out, err, run_err = _run(["git", "rev-parse", "HEAD"], cwd=root, timeout_s=2.0)
    if run_err is not None:
        env["git_error"] = f"git_rev_parse_failed:{run_err}"
        errors.append(env["git_error"])
        return
    if rc != 0:
        env["git_error"] = f"git_rev_parse_nonzero_rc:{rc}:{_truncate(err.strip(), 200)}"
        errors.append(env["git_error"])
        return

    commit = out.strip()
    env["git_commit"] = commit or None

    rc, out, err, run_err = _run(["git", "status", "--porcelain"], cwd=root, timeout_s=2.0)
    if run_err is not None:
        env["git_error"] = f"git_status_failed:{run_err}"
        errors.append(env["git_error"])
        return
    if rc != 0:
        env["git_error"] = f"git_status_nonzero_rc:{rc}:{_truncate(err.strip(), 200)}"
        errors.append(env["git_error"])
        return
    env["git_dirty"] = 1 if out.strip() else 0


def collect_env(*, precision: str, amp: Optional[int]) -> Dict[str, Any]:
    errors: List[str] = []

    env: Dict[str, Any] = {
        "env_schema_version": ENV_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "generated_utc": _utc_iso(),
        "os": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": None,
        "cuda_version": None,
        "precision": precision,
        "amp": amp,
        "gpu_name": None,
        "total_vram_bytes": None,
        "compute_capability": None,
        "driver_version": None,
        "cuda_driver_reported": None,
        "driver_model": None,
        "nvidia_smi_query": None,
        "nvidia_smi_raw": None,
        "git_commit": None,
        "git_dirty": None,
        "git_error": None,
        "env_vars": {k: os.environ.get(k) for k in ENV_VAR_WHITELIST},
        "errors": [],  # filled at end
    }

    _collect_torch(env, errors)
    _collect_nvidia_smi(env, errors)
    _collect_git(env, errors)

    env["errors"] = errors

    # Enforce exact key set contract for v1.
    extra = set(env.keys()) - set(ENV_KEYS_V1)
    missing = set(ENV_KEYS_V1) - set(env.keys())
    if extra or missing:
        # Never raise; record and best-effort normalize.
        if extra:
            env["errors"].append(f"schema_extra_keys:{sorted(extra)}")
            for k in sorted(extra):
                env.pop(k, None)
        if missing:
            env["errors"].append(f"schema_missing_keys:{sorted(missing)}")
            for k in sorted(missing):
                env[k] = None

    # Ensure deterministic insertion order (v1 key order).
    ordered = {k: env.get(k) for k in ENV_KEYS_V1}
    return ordered


def write_env_json(*, out_dir: Path, precision: str, amp: Optional[int]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env(precision=precision, amp=amp)

    tmp_path = out_dir / "env.json.tmp"
    final_path = out_dir / "env.json"

    # Atomic write: write temp then replace.
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(env, f, ensure_ascii=True, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(tmp_path, final_path)
    return final_path.resolve()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write env.json for VRAXION GPU runs (VRA-29).")
    p.add_argument("--out-dir", required=True, help="Directory to write env.json into.")
    p.add_argument("--precision", default="unknown", choices=("fp32", "bf16", "fp16", "unknown"))
    p.add_argument("--amp", type=int, choices=(0, 1), default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        out_dir = Path(args.out_dir)
        env_path = write_env_json(out_dir=out_dir, precision=args.precision, amp=args.amp)
    except Exception as e:
        # Only hard failure: cannot write output directory/file.
        print(f"ERROR: failed to write env.json: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print(f"WROTE env.json: {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

