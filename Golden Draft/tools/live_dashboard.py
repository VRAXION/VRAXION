"""Live training dashboard for VRAXION logs.

This is "Golden Draft" internal tooling.

Recommended usage:
  streamlit run tools/live_dashboard.py -- --log logs/current/vraxion.log

Design goals
- Import-safe without Streamlit installed (parsers are stdlib-only).
- Keep parsing behavior stable vs the original draft script.

Parsing API
- ``parse_log_lines(lines)`` returns stdlib-only parsed rows.
- ``parse_log(log_path)`` returns a pandas DataFrame (requires pandas).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence


REFLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
RESTEP = re.compile(rf"step\s+(?P<step>\d+)(?:/\d+)?\s+\|\s+loss\s+(?P<loss>{REFLOAT})\s+\|(?P<tail>.*)")
REGRAD = re.compile(rf"grad_norm\(theta_ptr\)=(?P<grad>{REFLOAT})")
RERAW = re.compile(rf"raw_delta=(?P<raw_delta>{REFLOAT})")
RERD = re.compile(rf"\bRD:(?P<raw_delta>{REFLOAT})\b")
RESHARD = re.compile(r"shard=(?P<shard_count>\d+)/(?P<shard_size>\d+)")
RETRACT = re.compile(rf"traction=(?P<traction>{REFLOAT})")
REACC = re.compile(rf"\bacc[= ]+(?P<acc>{REFLOAT})")
REANT_ACC = re.compile(r"ant_acc=(?P<ant_acc>[\d.,]+)")
REANT_ROUTE = re.compile(r"ant_route=(?P<ant_route>[\d,]+)")
REANT_ENT = re.compile(rf"ant_ent=(?P<ant_ent>{REFLOAT})")
REANT_ACTIVE = re.compile(r"ant_active=(?P<ant_active>\d+)")

# Additional telemetry fields (2026-02-10 comprehensive telemetry redesign)
RE_TIMING = re.compile(rf"s_per_step=(?P<s_per_step>{REFLOAT})")
RE_ACC_MA = re.compile(rf"acc_ma=(?P<acc_ma>{REFLOAT})")
RE_GNORM_FULL = re.compile(rf"gnorm=(?P<gnorm>{REFLOAT})")
RE_PANIC = re.compile(r"panic=(?P<panic>\w+)")
RE_SCALE = re.compile(rf"scale=(?P<scale>{REFLOAT})")
RE_INERTIA = re.compile(rf"inertia=(?P<inertia>{REFLOAT})")
RE_DEADZONE = re.compile(rf"deadzone=(?P<deadzone>{REFLOAT})")
RE_WALK = re.compile(rf"walk=(?P<walk>{REFLOAT})")
RE_FLIP_RATE = re.compile(rf"flip_rate=(?P<flip_rate>{REFLOAT})")
RE_ORBIT = re.compile(rf"orbit=(?P<orbit>{REFLOAT})")
RE_RESIDUAL = re.compile(rf"residual=(?P<residual>{REFLOAT})")
RE_ANCHOR_CLICKS = re.compile(r"anchor_clicks=(?P<anchor_clicks>\d+)")
RE_AGC_STATUS = re.compile(r"agc=(?P<agc_status>\w+)")


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _tryflt(valstr: str) -> Optional[float]:
    try:
        return float(valstr)
    except Exception:
        return None


def parse_log_lines(lines: Iterable[str]) -> List[Dict[str, Any]]:
    """Parse log lines into dict rows.

    Behavior contract (matches original draft script):
    - A ``grad_norm(theta_ptr)=...`` line sets a temporary grad value.
    - That grad value is attached to the *next* parsed step line only.
    - Step rows contain: step, loss, raw_delta, shard_count, shard_size,
      traction (optional), grad_norm (optional).
    """

    rows6x: List[Dict[str, Any]] = []
    grad6x: Optional[float] = None

    for linstr in lines:
        grdmat = REGRAD.search(linstr)
        if grdmat is not None:
            grdstr = grdmat.group("grad")
            grad6x = _tryflt(grdstr)
            continue

        stpmat = RESTEP.search(linstr)
        if stpmat is None:
            continue

        tail = stpmat.group("tail")
        rawmat = RERAW.search(tail)
        if rawmat is None:
            rawmat = RERD.search(tail)

        shdmat = RESHARD.search(tail)
        trcmat = RETRACT.search(tail)
        accmat = REACC.search(tail)

        raw_delta = _tryflt(rawmat.group("raw_delta")) if rawmat is not None else None
        shard_count = _tryflt(shdmat.group("shard_count")) if shdmat is not None else None
        shard_size = _tryflt(shdmat.group("shard_size")) if shdmat is not None else None
        trac6x = _tryflt(trcmat.group("traction")) if trcmat is not None else None
        acc6x = _tryflt(accmat.group("acc")) if accmat is not None else None

        ant_acc_mat = REANT_ACC.search(tail)
        ant_route_mat = REANT_ROUTE.search(tail)
        ant_ent_mat = REANT_ENT.search(tail)
        ant_active_mat = REANT_ACTIVE.search(tail)

        # Additional telemetry fields
        timing_mat = RE_TIMING.search(tail)
        acc_ma_mat = RE_ACC_MA.search(tail)
        gnorm_mat = RE_GNORM_FULL.search(tail)
        panic_mat = RE_PANIC.search(tail)
        scale_mat = RE_SCALE.search(tail)
        inertia_mat = RE_INERTIA.search(tail)
        deadzone_mat = RE_DEADZONE.search(tail)
        walk_mat = RE_WALK.search(tail)
        flip_rate_mat = RE_FLIP_RATE.search(tail)
        orbit_mat = RE_ORBIT.search(tail)
        residual_mat = RE_RESIDUAL.search(tail)
        anchor_clicks_mat = RE_ANCHOR_CLICKS.search(tail)
        agc_status_mat = RE_AGC_STATUS.search(tail)

        # For legacy step lines without explicit shard metadata.
        if shard_count is None:
            shard_count = 0.0
        if shard_size is None:
            shard_size = 0.0
        if raw_delta is None:
            raw_delta = 0.0

        rowdat: Dict[str, Any] = {
            "step": int(stpmat.group("step")),
            "loss": float(stpmat.group("loss")),
            "raw_delta": float(raw_delta),
            "shard_count": float(shard_count),
            "shard_size": float(shard_size),
            "traction": trac6x,
            "grad_norm": grad6x,
            "acc": acc6x,
            "ant_acc": ant_acc_mat.group("ant_acc") if ant_acc_mat else None,
            "ant_route": ant_route_mat.group("ant_route") if ant_route_mat else None,
            "ant_ent": _tryflt(ant_ent_mat.group("ant_ent")) if ant_ent_mat else None,
            "ant_active": int(ant_active_mat.group("ant_active")) if ant_active_mat else None,
            # Additional telemetry fields
            "s_per_step": _tryflt(timing_mat.group("s_per_step")) if timing_mat else None,
            "acc_ma": _tryflt(acc_ma_mat.group("acc_ma")) if acc_ma_mat else None,
            "gnorm": _tryflt(gnorm_mat.group("gnorm")) if gnorm_mat else None,
            "panic": panic_mat.group("panic") if panic_mat else None,
            "scale": _tryflt(scale_mat.group("scale")) if scale_mat else None,
            "inertia": _tryflt(inertia_mat.group("inertia")) if inertia_mat else None,
            "deadzone": _tryflt(deadzone_mat.group("deadzone")) if deadzone_mat else None,
            "walk": _tryflt(walk_mat.group("walk")) if walk_mat else None,
            "flip_rate": _tryflt(flip_rate_mat.group("flip_rate")) if flip_rate_mat else None,
            "orbit": _tryflt(orbit_mat.group("orbit")) if orbit_mat else None,
            "residual": _tryflt(residual_mat.group("residual")) if residual_mat else None,
            "anchor_clicks": int(anchor_clicks_mat.group("anchor_clicks")) if anchor_clicks_mat else None,
            "agc_status": agc_status_mat.group("agc_status") if agc_status_mat else None,
        }
        rows6x.append(rowdat)

        # Grad applies to only the next step.
        grad6x = None

    return rows6x


def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """Read and parse a log file.

    Missing/unreadable files return an empty list.
    """

    if not os.path.exists(log_path):
        return []

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as filobj:
            linlst = filobj.readlines()
    except OSError:
        return []

    return parse_log_lines(linlst)


def _read_tail_lines(path: str, max_lines: int = 40, max_bytes: int = 2_000_000) -> List[str]:
    """Read up to max_lines from the end of a text file.

    Implementation is intentionally streaming-friendly: it avoids reading the
    whole file into memory (important for long-running saturation runs).
    """
    if not os.path.exists(path):
        return []
    if int(max_lines) <= 0:
        return []
    try:
        with open(path, "rb") as filobj:
            filobj.seek(0, os.SEEK_END)
            size = int(filobj.tell())
            start = max(0, size - int(max(1024, max_bytes)))
            filobj.seek(start, os.SEEK_SET)
            data = filobj.read()
    except OSError:
        return []

    text = data.decode("utf-8", errors="replace")
    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    return lines[-int(max_lines) :]


def _read_new_lines(path: str, pos: int, max_bytes: int = 2_000_000) -> tuple[int, List[str]]:
    """Read newly appended lines starting from pos (bytes)."""
    if not os.path.exists(path):
        return pos, []
    try:
        size = int(os.stat(path).st_size)
    except OSError:
        return pos, []

    # Handle truncation/rotation.
    if int(pos) > size:
        pos = 0

    try:
        with open(path, "rb") as filobj:
            filobj.seek(int(pos), os.SEEK_SET)
            data = filobj.read(int(max(0, max_bytes)))
            new_pos = int(filobj.tell())
    except OSError:
        return pos, []

    if not data:
        return new_pos, []

    text = data.decode("utf-8", errors="replace")
    return new_pos, [line for line in text.splitlines() if line.strip()]


def _infer_run_root(log_path: str) -> str:
    if not log_path:
        return ""
    norm = os.path.normpath(str(log_path))
    parts = norm.split(os.sep)
    if len(parts) >= 2 and parts[-2].lower() == "train":
        return os.path.dirname(os.path.dirname(norm))
    return ""


def _load_json_obj(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as filobj:
            obj = json.load(filobj)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _read_jsonl_tail(path: str, max_lines: int = 4000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in _read_tail_lines(path, max_lines=max_lines, max_bytes=6_000_000):
        txt = line.strip()
        if not txt:
            continue
        try:
            row = json.loads(txt)
            if isinstance(row, dict):
                out.append(row)
        except Exception:
            continue
    return out


def collect_live_status(log_path: str) -> Dict[str, Any]:
    """Collect file-level liveness signals for pre-step phases.

    This avoids a false "nothing running" appearance when training step lines
    are not yet emitted (for example during probe stage).
    """
    out: Dict[str, Any] = {
        "log_exists": bool(os.path.exists(log_path)),
        "log_size_bytes": 0,
        "log_mtime_epoch": None,
        "log_age_s": None,
        "log_tail": [],
        "stderr_tail": [],
        "supervisor_tail": [],
    }
    if not out["log_exists"]:
        return out

    try:
        st = os.stat(log_path)
        out["log_size_bytes"] = int(st.st_size)
        out["log_mtime_epoch"] = float(st.st_mtime)
        out["log_age_s"] = max(0.0, float(time.time()) - float(st.st_mtime))
    except OSError:
        pass

    out["log_tail"] = _read_tail_lines(log_path, max_lines=30)
    base_dir = os.path.dirname(log_path)
    # Support both:
    # - supervisor-captured logs (log_path == .../supervisor_job/child_stdout.log)
    # - run log (log_path == .../train/vraxion.log) where supervisor_job is a sibling.
    cand_dirs = [
        base_dir,
        os.path.join(base_dir, "supervisor_job"),
        os.path.join(os.path.dirname(base_dir), "supervisor_job"),
    ]
    stderr_path = ""
    supervisor_path = ""
    for d in cand_dirs:
        p_err = os.path.join(d, "child_stderr.log")
        p_sup = os.path.join(d, "supervisor.log")
        if not stderr_path and os.path.exists(p_err):
            stderr_path = p_err
        if not supervisor_path and os.path.exists(p_sup):
            supervisor_path = p_sup
    if not stderr_path:
        stderr_path = os.path.join(base_dir, "child_stderr.log")
    if not supervisor_path:
        supervisor_path = os.path.join(base_dir, "supervisor.log")
    out["stderr_tail"] = _read_tail_lines(stderr_path, max_lines=20)
    out["supervisor_tail"] = _read_tail_lines(supervisor_path, max_lines=20)
    return out


def parse_log(log_path: str):
    """Parse a log file into a pandas DataFrame.

    This function requires pandas. Importing this module does not.
    """

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for parse_log()") from exc

    rows6x = parse_log_file(log_path)
    dfobj6 = pd.DataFrame(rows6x)

    if dfobj6.empty:
        return dfobj6

    if "step" in dfobj6.columns:
        dfobj6 = dfobj6.drop_duplicates(subset=["step"]).sort_values("step")

    # Derived metric used by the dashboard.
    # Use new telemetry field names (gnorm, s_per_step) if available, fallback to legacy (grad_norm, raw_delta)
    if "gnorm" in dfobj6.columns and dfobj6["gnorm"].notna().any():
        grad_col = "gnorm"
    else:
        grad_col = "grad_norm"

    if "s_per_step" in dfobj6.columns and dfobj6["s_per_step"].notna().any():
        time_col = "s_per_step"
    else:
        time_col = "raw_delta"

    dfobj6["tension"] = dfobj6[grad_col] * dfobj6[time_col] / 100.0

    # Clip outliers for plotting readability.
    capval = dfobj6["tension"].quantile(0.99)
    dfobj6["tension"] = dfobj6["tension"].clip(upper=capval)

    return dfobj6


def _req_st() -> Any:
    try:
        import streamlit as stmod6  # type: ignore

        return stmod6
    except Exception:
        _eprint("[live_dashboard] ERROR: streamlit is required to run the dashboard")
        _eprint("[live_dashboard] Hint: pip install streamlit")
        raise


def _opt_mod(modnam: str) -> Any:
    try:
        __import__(modnam)
        return sys.modules[modnam]
    except Exception:
        return None


def _autorf() -> Any:
    """Return an autorefresh callable if available, else None."""

    # Prefer streamlit-autorefresh if installed.
    modobj = _opt_mod("streamlit_autorefresh")
    if modobj is not None:
        try:
            fncobj = getattr(modobj, "st_autorefresh")
            return fncobj
        except Exception:
            pass

    # Back-compat: some environments vend an autorefresh helper under streamlit.
    try:
        from streamlit import autorefresh as fncobj  # type: ignore

        return fncobj
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="VRAXION live dashboard")
    parser.add_argument("--log", default=os.path.join("logs", "current", "vraxion.log"))
    parser.add_argument("--eval-stream", default="")
    parser.add_argument("--eval-status", default="")
    parser.add_argument("--refresh", type=int, default=10, help="Auto-refresh seconds (0 disables)")
    parser.add_argument("--max-rows", type=int, default=5000, help="Display at most N rows (0 = all)")

    argobj, _unk6x = parser.parse_known_args(list(argv) if argv is not None else None)

    # Late import so tests can import parsers without Streamlit.
    try:
        stmod6 = _req_st()
    except Exception:
        return

    @stmod6.cache_resource
    def _open_browser_once() -> None:
        """Open browser exactly once per Streamlit session (survives hot-reload)."""
        import webbrowser
        import time
        import os

        # Only open in dev mode (not headless)
        if not os.environ.get("STREAMLIT_SERVER_HEADLESS"):
            time.sleep(1.5)  # Wait for server ready
            webbrowser.open("http://localhost:8501")

    # Auto-open browser once per session
    _open_browser_once()

    stmod6.set_page_config(page_title="VRAXION Live Dashboard", layout="wide")
    stmod6.title("VRAXION Live Dashboard")

    logpth = stmod6.sidebar.text_input("Log path", value=str(argobj.log), key="sb_log_path")
    inferred_run_root = _infer_run_root(logpth)
    default_eval_stream = (
        str(argobj.eval_stream)
        if str(argobj.eval_stream).strip()
        else (os.path.join(inferred_run_root, "eval_stream.jsonl") if inferred_run_root else "")
    )
    default_eval_status = (
        str(argobj.eval_status)
        if str(argobj.eval_status).strip()
        else (os.path.join(inferred_run_root, "eval_catchup_status.json") if inferred_run_root else "")
    )
    eval_stream_path = stmod6.sidebar.text_input("Eval stream path", value=str(default_eval_stream), key="sb_eval_stream")
    eval_status_path = stmod6.sidebar.text_input("Eval status path", value=str(default_eval_status), key="sb_eval_status")
    rfrsec = stmod6.sidebar.number_input(
        "Refresh interval (sec)",
        min_value=0,
        max_value=600,
        value=int(argobj.refresh),
        key="sb_refresh",
    )
    maxrow = stmod6.sidebar.number_input(
        "Max rows",
        min_value=0,
        max_value=500000,
        value=int(argobj.max_rows),
        key="sb_max_rows",
    )
    show_separate_charts = stmod6.sidebar.checkbox(
        "Show separate loss/tension charts",
        value=False,
        key="sb_separate_charts",
    )
    enable_3d_diag = stmod6.sidebar.checkbox(
        "Enable 3D diagnostics",
        value=True,
        key="sb_3d_diag",
    )
    show_3d_eval_markers = stmod6.sidebar.checkbox(
        "Show 3D eval markers",
        value=True,
        key="sb_3d_eval_markers",
    )
    max_3d_points = int(
        stmod6.sidebar.number_input(
            "3D max points",
            min_value=200,
            max_value=20000,
            value=1500,
            step=100,
            key="sb_3d_max_points",
        )
    )
    eval_attach_gap_steps = int(
        stmod6.sidebar.number_input(
            "3D eval gap cap (steps)",
            min_value=0,
            max_value=5000,
            value=100,
            step=10,
            key="sb_3d_eval_gap",
        )
    )
    camera_preset = stmod6.sidebar.selectbox(
        "3D camera",
        options=["isometric", "top", "side"],
        index=0,
        key="sb_camera_preset",
    )

    if int(rfrsec) > 0:
        fncobj = _autorf()
        if fncobj is not None:
            fncobj(interval=int(rfrsec) * 1000, key="auto_refresh")
        else:
            stmod6.sidebar.caption(
                "Auto-refresh helper not available; install streamlit-autorefresh for periodic refresh."
            )

    # Stream-friendly parsing: keep a rolling window of parsed rows in session
    # state, and only read newly appended bytes each refresh. This avoids
    # re-reading huge logs and prevents RAM blowups on long runs.
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        stmod6.error(f"pandas is required for the dashboard: {exc}")
        return

    if "live_log_path" not in stmod6.session_state:
        stmod6.session_state["live_log_path"] = ""
    if "live_log_pos" not in stmod6.session_state:
        stmod6.session_state["live_log_pos"] = 0
    if "live_rows" not in stmod6.session_state:
        stmod6.session_state["live_rows"] = []

    if stmod6.session_state["live_log_path"] != str(logpth):
        stmod6.session_state["live_log_path"] = str(logpth)
        stmod6.session_state["live_log_pos"] = 0
        stmod6.session_state["live_rows"] = []

    # On first load, start near the tail to avoid ingesting a massive file.
    try:
        size0 = int(os.stat(logpth).st_size)
    except OSError:
        size0 = 0
    if stmod6.session_state["live_log_pos"] == 0 and not stmod6.session_state["live_rows"] and size0 > 0:
        stmod6.session_state["live_log_pos"] = max(0, size0 - 2_000_000)

    new_pos, new_lines = _read_new_lines(str(logpth), int(stmod6.session_state["live_log_pos"]))
    stmod6.session_state["live_log_pos"] = int(new_pos)
    if new_lines:
        stmod6.session_state["live_rows"].extend(parse_log_lines(new_lines))

    dfobj6 = pd.DataFrame(stmod6.session_state["live_rows"])
    if not dfobj6.empty and "step" in dfobj6.columns:
        dfobj6 = dfobj6.drop_duplicates(subset=["step"]).sort_values("step")

    # Keep memory bounded even if maxrow is set very high.
    keep_rows = 0
    if int(maxrow) > 0:
        keep_rows = int(maxrow) * 2
    else:
        keep_rows = 20000
    if keep_rows > 0 and len(dfobj6) > keep_rows:
        dfobj6 = dfobj6.tail(keep_rows)

    # Persist the bounded window back to session state.
    stmod6.session_state["live_rows"] = dfobj6.to_dict(orient="records")

    try:
        # Recompute derived metric for plotting readability.
        if not dfobj6.empty:
            # Calculate tension using new telemetry fields (gnorm, s_per_step) or legacy fields
            grad_col = "gnorm" if ("gnorm" in dfobj6.columns and dfobj6["gnorm"].notna().any()) else "grad_norm"
            time_col = "s_per_step" if ("s_per_step" in dfobj6.columns and dfobj6["s_per_step"].notna().any()) else "raw_delta"

            if grad_col in dfobj6.columns and time_col in dfobj6.columns:
                dfobj6["tension"] = dfobj6[grad_col] * dfobj6[time_col] / 100.0
                capval = dfobj6["tension"].quantile(0.99)
                dfobj6["tension"] = dfobj6["tension"].clip(upper=capval)
    except Exception as exc:
        stmod6.error(f"Failed to parse log: {exc}")
        return

    # Expand per-ant telemetry into individual columns
    if not dfobj6.empty:
        if "ant_acc" in dfobj6.columns and dfobj6["ant_acc"].notna().any():
            _aa = dfobj6["ant_acc"].dropna().iloc[0]
            _n_ants = len(str(_aa).split(","))
            _aa_split = dfobj6["ant_acc"].str.split(",", expand=True)
            for _i in range(_aa_split.shape[1]):
                dfobj6[f"ant_{_i}_acc"] = pd.to_numeric(_aa_split[_i], errors="coerce")
        if "ant_route" in dfobj6.columns and dfobj6["ant_route"].notna().any():
            _ar_split = dfobj6["ant_route"].str.split(",", expand=True)
            for _i in range(_ar_split.shape[1]):
                dfobj6[f"ant_{_i}_route"] = pd.to_numeric(_ar_split[_i], errors="coerce")

    if dfobj6.empty:
        stmod6.warning("No step rows parsed yet.")
        stmod6.caption(f"Watching log: {logpth}")

        status = collect_live_status(logpth)
        col1, col2, col3 = stmod6.columns(3)
        with col1:
            stmod6.metric("Log exists", "yes" if status.get("log_exists") else "no")
        with col2:
            stmod6.metric("Log size (bytes)", int(status.get("log_size_bytes") or 0))
        with col3:
            age = status.get("log_age_s")
            stmod6.metric("Last write age (s)", int(age) if isinstance(age, (float, int)) else -1)

        stmod6.info(
            "Run may still be active in probe/eval stage before train step lines. "
            "Use tails below to confirm liveness."
        )

        if status.get("supervisor_tail"):
            stmod6.subheader("Supervisor tail")
            stmod6.code("\n".join(status["supervisor_tail"]), language="text")
        if status.get("stderr_tail"):
            stmod6.subheader("Child stderr tail")
            stmod6.code("\n".join(status["stderr_tail"]), language="text")
        if status.get("log_tail"):
            stmod6.subheader("Child stdout tail")
            stmod6.code("\n".join(status["log_tail"]), language="text")
        # Even before step rows appear, still show eval lane status if present.
        eval_state = _load_json_obj(str(eval_status_path))
        if eval_state:
            stmod6.subheader("Eval catch-up")
            pct_step = float(eval_state.get("eval_catchup_pct", 0.0) or 0.0)
            pct_queue = float(eval_state.get("queue_catchup_pct", 0.0) or 0.0)
            pct = max(float(pct_step), float(pct_queue))
            stmod6.progress(max(0.0, min(1.0, pct / 100.0)), text=f"Eval catch-up: {pct:.1f}%")
            q1, q2, q3, q4 = stmod6.columns(4)
            with q1:
                stmod6.metric("Queue depth", int(eval_state.get("queue_depth", 0) or 0))
            with q2:
                stmod6.metric("Queue ETA (s)", int(float(eval_state.get("queue_eta_sec", 0.0) or 0.0)))
            with q3:
                stmod6.metric("Eval stride", int(eval_state.get("current_stride", 0) or 0))
            with q4:
                stmod6.metric("Eval mode", str(eval_state.get("adaptive_mode", "unknown")))
        return

    if int(maxrow) > 0 and len(dfobj6) > int(maxrow):
        dfobj6 = dfobj6.tail(int(maxrow))

    last6x = dfobj6.iloc[-1]
    latest_step = int(last6x.get("step", 0))
    h1, h2, h3 = stmod6.columns(3)
    with h1:
        stmod6.metric("Step", latest_step)

    eval_state = _load_json_obj(str(eval_status_path))
    eval_rows = _read_jsonl_tail(str(eval_stream_path), max_lines=max(1000, int(maxrow or 0)))
    evdf = pd.DataFrame(eval_rows) if eval_rows else pd.DataFrame()
    if not evdf.empty and "step" in evdf.columns:
        evdf["step"] = pd.to_numeric(evdf["step"], errors="coerce")
        evdf = evdf.dropna(subset=["step"])
        if not evdf.empty:
            evdf["step"] = evdf["step"].astype(int)
            evdf = evdf.drop_duplicates(subset=["step"]).sort_values("step")
    latest_eval_row: Dict[str, Any] = eval_rows[-1] if eval_rows else {}
    catchup_pct_step = float(eval_state.get("eval_catchup_pct", 0.0) or 0.0)
    catchup_pct_queue = float(eval_state.get("queue_catchup_pct", 0.0) or 0.0)
    catchup_pct = max(float(catchup_pct_step), float(catchup_pct_queue))
    if latest_step > 0 and latest_eval_row.get("step") is not None:
        try:
            eval_step = int(latest_eval_row.get("step"))
            catchup_pct = max(catchup_pct, (100.0 * float(eval_step) / float(latest_step)))
        except Exception:
            pass
    with h2:
        stmod6.metric("Eval catch-up %", f"{max(0.0, min(100.0, catchup_pct)):.1f}")
    with h3:
        stmod6.metric("Queue depth", int(eval_state.get("queue_depth", 0) or 0))
    stmod6.progress(max(0.0, min(1.0, float(catchup_pct) / 100.0)), text=f"Eval catch-up: {catchup_pct:.1f}%")

    q1, q2, q3, q4 = stmod6.columns(4)
    with q1:
        stmod6.metric("Queue ETA (s)", int(float(eval_state.get("queue_eta_sec", 0.0) or 0.0)))
    with q2:
        stmod6.metric("Eval stride", int(eval_state.get("current_stride", 0) or 0))
    with q3:
        stmod6.metric("Latest eval n", int(latest_eval_row.get("eval_n", 0) or 0))
    with q4:
        stmod6.metric("Eval mode", str(eval_state.get("adaptive_mode", "unknown")))

    if latest_eval_row:
        ci95_lo = latest_eval_row.get("ci95_low")
        ci95_hi = latest_eval_row.get("ci95_high")
        ci99_lo = latest_eval_row.get("ci99_low")
        ci99_hi = latest_eval_row.get("ci99_high")
        e1, e2, e3 = stmod6.columns(3)
        with e1:
            stmod6.metric("Latest eval step", int(latest_eval_row.get("step", 0) or 0))
        with e2:
            if isinstance(ci95_lo, (int, float)) and isinstance(ci95_hi, (int, float)):
                stmod6.metric("CI95", f"[{float(ci95_lo):.4f}, {float(ci95_hi):.4f}]")
            else:
                stmod6.metric("CI95", "n/a")
        with e3:
            if isinstance(ci99_lo, (int, float)) and isinstance(ci99_hi, (int, float)):
                stmod6.metric("CI99", f"[{float(ci99_lo):.4f}, {float(ci99_hi):.4f}]")
            else:
                stmod6.metric("CI99", "n/a")

    # Chart backend: prefer plotly if installed.
    pltx6x = _opt_mod("plotly.express")
    pltgo6x = _opt_mod("plotly.graph_objects")
    if pltx6x is not None:
        try:
            has_tension = ("tension" in dfobj6.columns) and bool(dfobj6["tension"].notna().any())
            eval_line_df = pd.DataFrame()
            if not evdf.empty and "eval_acc" in evdf.columns:
                eval_line_df = evdf[["step", "eval_acc"]].copy()
                eval_line_df["eval_acc"] = pd.to_numeric(eval_line_df["eval_acc"], errors="coerce")
                eval_line_df = eval_line_df.dropna(subset=["step", "eval_acc"]).sort_values("step")
            has_eval_line = not eval_line_df.empty

            if not bool(show_separate_charts) and pltgo6x is not None:
                figov = pltgo6x.Figure()
                if has_eval_line:
                    figov.add_trace(
                        pltgo6x.Scatter(
                            x=eval_line_df["step"],
                            y=eval_line_df["eval_acc"],
                            name="Eval Acc",
                            mode="lines",
                            line={"color": "#e11d48", "width": 4},
                            yaxis="y",
                        )
                    )
                    figov.add_trace(
                        pltgo6x.Scatter(
                            x=dfobj6["step"],
                            y=dfobj6["loss"],
                            name="Loss",
                            mode="lines",
                            line={"color": "#93c5fd", "width": 2, "dash": "dot"},
                            yaxis="y3",
                        )
                    )
                else:
                    figov.add_trace(
                        pltgo6x.Scatter(
                            x=dfobj6["step"],
                            y=dfobj6["loss"],
                            name="Loss",
                            mode="lines",
                            line={"color": "#4f8df5", "width": 2},
                            yaxis="y",
                        )
                    )
                if has_tension:
                    figov.add_trace(
                        pltgo6x.Scatter(
                            x=dfobj6["step"],
                            y=dfobj6["tension"],
                            name="Tension",
                            mode="lines",
                            line={"color": "#f97316", "width": 2, "dash": "dash"},
                            yaxis="y2",
                        )
                    )
                layout_cfg = {
                    "title": "Eval + Loss + Tension Overlay",
                    "xaxis": {"title": "Step"},
                    "yaxis": {"title": ("Eval Acc" if has_eval_line else "Loss")},
                    "yaxis2": {
                        "title": "Tension",
                        "overlaying": "y",
                        "side": "right",
                        "position": 1.0,
                        "showgrid": False,
                    },
                    "legend": {"orientation": "h", "y": 1.02, "x": 0.01},
                    "margin": {"l": 50, "r": 50, "t": 50, "b": 40},
                }
                if has_eval_line:
                    layout_cfg["yaxis3"] = {
                        "title": "Loss",
                        "overlaying": "y",
                        "side": "right",
                        "position": 0.90,
                        "showgrid": False,
                    }
                figov.update_layout(**layout_cfg)
                stmod6.plotly_chart(figov, width="stretch")

                # ── 3D Surface Variants ───────────────────────────
                if len(dfobj6) >= 20:
                    stmod6.markdown("---")
                    stmod6.markdown(
                        "**3D Surface Variants** — pick your favourite")

                    import numpy as _np_surf

                    _surf_cols = ["step", "loss"]
                    if "acc" in dfobj6.columns:
                        _surf_cols.append("acc")
                    if "tension" in dfobj6.columns:
                        _surf_cols.append("tension")
                    # Include per-ant telemetry columns for E-H charts
                    for _sc in dfobj6.columns:
                        if (_sc.startswith("ant_") and (_sc.endswith("_acc") or _sc.endswith("_route"))
                                or _sc in ("ant_ent", "ant_active")):
                            _surf_cols.append(_sc)
                    surf_df = dfobj6[_surf_cols].dropna(
                        subset=["step", "loss"]).copy()
                    max_surf_pts = 300
                    if len(surf_df) > max_surf_pts:
                        _stride = max(1, len(surf_df) // max_surf_pts)
                        surf_df = surf_df.iloc[::_stride].reset_index(
                            drop=True)

                    step_arr = surf_df["step"].values
                    loss_arr = surf_df["loss"].values

                    # Semantic colorscale: fuchsia (good) → dark purple midpoint → cyan (bad)
                    _cs_vrx = [
                        [0.00, "#ff00ff"],  # bright fuchsia
                        [0.30, "#8b1a8b"],  # deep magenta
                        [0.50, "#2d1b4e"],  # dark purple midpoint
                        [0.70, "#1b4d5e"],  # dark teal
                        [1.00, "#00e5ff"],  # bright cyan
                    ]
                    cam_surf = {"eye": {"x": 1.45, "y": 1.45, "z": 0.95}}
                    _surf_margin = {"l": 5, "r": 5, "t": 35, "b": 5}
                    _surf_windows = [1, 3, 7, 15, 30, 60, 120, 200]

                    sA, sB, sC, sD = stmod6.columns(4)

                    # ── A: Multi-Scale Loss Terrain ───────────────
                    with sA:
                        z_a = []
                        for w in _surf_windows:
                            smoothed = surf_df["loss"].rolling(
                                window=w, min_periods=1).mean()
                            z_a.append(smoothed.values.tolist())
                        fig_a = pltgo6x.Figure(data=[pltgo6x.Surface(
                            x=step_arr.tolist(),
                            y=_surf_windows,
                            z=z_a,
                            colorscale=_cs_vrx,
                            showscale=False,
                            opacity=0.92,
                        )])
                        fig_a.update_layout(
                            title="A: Multi-Scale Loss",
                            scene={
                                "xaxis": {"title": "Step"},
                                "yaxis": {"title": "MA Window",
                                          "type": "log"},
                                "zaxis": {"title": "Loss"},
                                "bgcolor": "rgba(0,0,0,0)",
                            },
                            scene_camera=cam_surf,
                            margin=_surf_margin,
                            height=380,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        stmod6.plotly_chart(fig_a,
                                            width="stretch")

                    # ── B: Loss Volatility Ridge ──────────────────
                    with sB:
                        z_b = []
                        for w in _surf_windows:
                            vol = surf_df["loss"].rolling(
                                window=w, min_periods=1).std()
                            vol = vol.fillna(0.0)
                            z_b.append(vol.values.tolist())
                        fig_b = pltgo6x.Figure(data=[pltgo6x.Surface(
                            x=step_arr.tolist(),
                            y=_surf_windows,
                            z=z_b,
                            colorscale=_cs_vrx,
                            showscale=False,
                            opacity=0.92,
                        )])
                        fig_b.update_layout(
                            title="B: Loss Volatility",
                            scene={
                                "xaxis": {"title": "Step"},
                                "yaxis": {"title": "MA Window",
                                          "type": "log"},
                                "zaxis": {"title": "Std Dev"},
                                "bgcolor": "rgba(0,0,0,0)",
                            },
                            scene_camera=cam_surf,
                            margin=_surf_margin,
                            height=380,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        stmod6.plotly_chart(fig_b,
                                            width="stretch")

                    # ── C: Quantile Envelope Canyon ───────────────
                    with sC:
                        _quantiles = [0.05, 0.15, 0.25, 0.35, 0.45,
                                      0.55, 0.65, 0.75, 0.85, 0.95]
                        z_c = []
                        for q in _quantiles:
                            qv = surf_df["loss"].rolling(
                                window=50, min_periods=1).quantile(q)
                            z_c.append(qv.values.tolist())
                        fig_c = pltgo6x.Figure(data=[pltgo6x.Surface(
                            x=step_arr.tolist(),
                            y=_quantiles,
                            z=z_c,
                            colorscale=_cs_vrx,
                            showscale=False,
                            opacity=0.92,
                        )])
                        fig_c.update_layout(
                            title="C: Quantile Canyon",
                            scene={
                                "xaxis": {"title": "Step"},
                                "yaxis": {"title": "Quantile"},
                                "zaxis": {"title": "Loss"},
                                "bgcolor": "rgba(0,0,0,0)",
                            },
                            scene_camera=cam_surf,
                            margin=_surf_margin,
                            height=380,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        stmod6.plotly_chart(fig_c,
                                            width="stretch")

                    # ── D: Multi-Scale Accuracy Terrain ───────────
                    with sD:
                        if ("acc" in surf_df.columns
                                and surf_df["acc"].notna().any()):
                            z_d = []
                            for w in _surf_windows:
                                smoothed = surf_df["acc"].rolling(
                                    window=w, min_periods=1).mean()
                                z_d.append(smoothed.values.tolist())
                            fig_d = pltgo6x.Figure(
                                data=[pltgo6x.Surface(
                                    x=step_arr.tolist(),
                                    y=_surf_windows,
                                    z=z_d,
                                    colorscale=_cs_vrx,
                                    reversescale=True,
                                    showscale=False,
                                    opacity=0.92,
                                )])
                            fig_d.update_layout(
                                title="D: Multi-Scale Accuracy",
                                scene={
                                    "xaxis": {"title": "Step"},
                                    "yaxis": {"title": "MA Window",
                                              "type": "log"},
                                    "zaxis": {"title": "Accuracy"},
                                    "bgcolor": "rgba(0,0,0,0)",
                                },
                                scene_camera=cam_surf,
                                margin=_surf_margin,
                                height=380,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                            )
                            stmod6.plotly_chart(fig_d,
                                                width="stretch")
                        else:
                            stmod6.caption(
                                "D: Accuracy data not available yet.")

                    # ── ARCHIVED: 3D Ant Swarm Telemetry (E-H plots) ─────────────
                    # This section has been archived to tools/_backup/ant_telemetry_backup.py.txt
                    # as part of the comprehensive telemetry redesign (2026-02-10).
                    # See backup file to restore if needed.
                    pass

                # ============================================================
                # Frequency Analysis — periodic structure in loss dynamics
                # ============================================================
                if len(dfobj6) >= 200:
                    stmod6.markdown("---")
                    stmod6.markdown("**Frequency Analysis** — periodic structure in loss dynamics")

                    try:
                        from scipy import signal
                        import numpy as np

                        # Prepare data (subsample to 300 points for performance)
                        freq_df = dfobj6[["step", "loss"]].dropna().copy()
                        if len(freq_df) > 300:
                            stride = max(1, len(freq_df) // 300)
                            freq_df = freq_df.iloc[::stride].reset_index(drop=True)

                        step_arr = freq_df["step"].values
                        loss_arr = freq_df["loss"].values
                        loss_detrended = signal.detrend(loss_arr)  # Remove linear trend for FFT

                        # Semantic colorscale matching existing plots
                        _cs_vrx = [
                            [0.00, "#ff00ff"],  # bright fuchsia
                            [0.30, "#8b1a8b"],  # deep magenta
                            [0.50, "#2d1b4e"],  # dark purple midpoint
                            [0.70, "#1b4d5e"],  # dark teal
                            [1.00, "#00e5ff"],  # bright cyan
                        ]
                        cam_surf = {"eye": {"x": 1.45, "y": 1.45, "z": 0.95}}
                        _surf_margin = {"l": 5, "r": 5, "t": 35, "b": 5}

                        sI, sJ, sK, sL = stmod6.columns(4)

                        # I: Loss Periodogram (3D Surface)
                        # Sliding window FFT to show which frequencies dominate at different phases
                        with sI:
                            window_size = min(100, len(loss_detrended) // 3)
                            stride_freq = max(10, window_size // 10)
                            if window_size >= 50:
                                freqs_list = []
                                power_list = []
                                time_list = []

                                for i in range(0, len(loss_detrended) - window_size, stride_freq):
                                    window = loss_detrended[i:i+window_size]
                                    freqs, power = signal.periodogram(window, fs=1.0)
                                    # Keep only meaningful frequencies (0.01 to 0.5 cycles/step)
                                    mask = (freqs >= 0.01) & (freqs <= 0.5)
                                    freqs_list.append(freqs[mask])
                                    power_list.append(power[mask])
                                    time_list.append(step_arr[i + window_size // 2])

                                if len(time_list) > 0:
                                    # Create meshgrid for 3D surface
                                    freq_grid = freqs_list[0]
                                    time_grid = np.array(time_list)
                                    power_grid = np.array([p for p in power_list])

                                    fig_i = pltgo6x.Figure(data=[pltgo6x.Surface(
                                        x=time_grid,
                                        y=freq_grid,
                                        z=power_grid.T,
                                        colorscale=_cs_vrx,
                                        showscale=False
                                    )])
                                    fig_i.update_layout(
                                        title="I: Periodogram",
                                        scene=dict(
                                            xaxis_title="Step",
                                            yaxis_title="Freq (cycles/step)",
                                            zaxis_title="Power",
                                            camera=cam_surf
                                        ),
                                        margin=_surf_margin,
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                    )
                                    sI.plotly_chart(fig_i, use_container_width=True)
                                else:
                                    sI.caption("I: Not enough data for periodogram")
                            else:
                                sI.caption("I: Need ≥50 points for periodogram")

                        # J: Autocorrelation Heatmap (3D Surface)
                        # Shows repeating patterns at different time offsets
                        with sJ:
                            max_lag = min(50, len(loss_detrended) // 4)
                            window_ac = min(100, len(loss_detrended) // 3)
                            stride_ac = max(10, window_ac // 10)

                            if window_ac >= 50 and max_lag >= 10:
                                acorr_list = []
                                time_ac_list = []
                                lags = np.arange(1, max_lag + 1)

                                for i in range(0, len(loss_detrended) - window_ac, stride_ac):
                                    window = loss_detrended[i:i+window_ac]
                                    acorr = np.correlate(window, window, mode='full')
                                    acorr = acorr[len(acorr)//2:]
                                    acorr = acorr / acorr[0]  # Normalize
                                    acorr_list.append(acorr[1:max_lag+1])
                                    time_ac_list.append(step_arr[i + window_ac // 2])

                                if len(time_ac_list) > 0:
                                    acorr_grid = np.array(acorr_list)
                                    time_ac_grid = np.array(time_ac_list)

                                    fig_j = pltgo6x.Figure(data=[pltgo6x.Surface(
                                        x=time_ac_grid,
                                        y=lags,
                                        z=acorr_grid.T,
                                        colorscale=_cs_vrx,
                                        showscale=False
                                    )])
                                    fig_j.update_layout(
                                        title="J: Autocorrelation",
                                        scene=dict(
                                            xaxis_title="Step",
                                            yaxis_title="Lag (steps)",
                                            zaxis_title="Correlation",
                                            camera=cam_surf
                                        ),
                                        margin=_surf_margin,
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                    )
                                    sJ.plotly_chart(fig_j, use_container_width=True)
                                else:
                                    sJ.caption("J: Not enough data for autocorrelation")
                            else:
                                sJ.caption("J: Need ≥50 points for autocorrelation")

                        # K: Peak Spacing Histogram (3D Bar Chart)
                        # Shows distribution of distances between consecutive loss peaks
                        with sK:
                            peaks, _ = signal.find_peaks(loss_detrended)
                            if len(peaks) >= 5:
                                spacings = np.diff(peaks)
                                # Bin into early/mid/late training
                                n_bins = min(3, len(step_arr) // 100 + 1)
                                time_bins = np.array_split(range(len(loss_detrended)), n_bins)

                                hist_data = []
                                bin_labels = []
                                spacing_range = range(1, int(max(spacings)) + 1) if len(spacings) > 0 else range(1, 10)

                                for bin_idx, time_bin in enumerate(time_bins):
                                    bin_peaks = peaks[(peaks >= time_bin[0]) & (peaks < time_bin[-1])]
                                    if len(bin_peaks) >= 2:
                                        bin_spacings = np.diff(bin_peaks)
                                        counts, edges = np.histogram(bin_spacings, bins=spacing_range)
                                        hist_data.append(counts)
                                        bin_labels.append(f"Phase {bin_idx+1}")
                                    else:
                                        hist_data.append(np.zeros(len(spacing_range)-1))
                                        bin_labels.append(f"Phase {bin_idx+1}")

                                if len(hist_data) > 0:
                                    hist_array = np.array(hist_data)
                                    x_grid, y_grid = np.meshgrid(bin_labels, list(spacing_range)[:-1])

                                    fig_k = pltgo6x.Figure(data=[pltgo6x.Surface(
                                        x=np.arange(len(bin_labels)),
                                        y=list(spacing_range)[:-1],
                                        z=hist_array.T,
                                        colorscale=_cs_vrx,
                                        showscale=False
                                    )])
                                    fig_k.update_layout(
                                        title="K: Peak Spacing",
                                        scene=dict(
                                            xaxis_title="Training Phase",
                                            yaxis_title="Spacing (steps)",
                                            zaxis_title="Count",
                                            camera=cam_surf
                                        ),
                                        margin=_surf_margin,
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                    )
                                    sK.plotly_chart(fig_k, use_container_width=True)
                                else:
                                    sK.caption("K: Not enough peaks found")
                            else:
                                sK.caption("K: Need ≥5 peaks for histogram")

                        # L: Oscillation Envelope (3D Line Surface)
                        # Separates smooth trend from high-frequency noise
                        with sL:
                            if len(loss_arr) >= 50:
                                # Butterworth lowpass filter
                                nyquist = 0.5  # Nyquist frequency (sampling is 1 step)
                                cutoff = 0.05  # Cutoff at 0.05 cycles/step (20-step period)
                                order = 2
                                b, a = signal.butter(order, cutoff / nyquist, btype='low')
                                trend = signal.filtfilt(b, a, loss_arr)
                                oscillation = loss_arr - trend

                                # Create 3D surface with 3 traces
                                traces = []
                                colors = ["#ff00ff", "#2d1b4e", "#00e5ff"]
                                names = ["Raw Loss", "Trend", "Oscillation"]
                                data_arrays = [loss_arr, trend, oscillation]

                                for idx, (data, color, name) in enumerate(zip(data_arrays, colors, names)):
                                    traces.append(pltgo6x.Scatter3d(
                                        x=step_arr,
                                        y=np.full_like(step_arr, idx),
                                        z=data,
                                        mode='lines',
                                        line=dict(color=color, width=2),
                                        name=name
                                    ))

                                fig_l = pltgo6x.Figure(data=traces)
                                fig_l.update_layout(
                                    title="L: Oscillation Envelope",
                                    scene=dict(
                                        xaxis_title="Step",
                                        yaxis_title="Signal Type",
                                        zaxis_title="Value",
                                        camera=cam_surf,
                                        yaxis=dict(
                                            tickmode='array',
                                            tickvals=[0, 1, 2],
                                            ticktext=names
                                        )
                                    ),
                                    margin=_surf_margin,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                )
                                sL.plotly_chart(fig_l, use_container_width=True)
                            else:
                                sL.caption("L: Need ≥50 points for envelope")

                    except ImportError:
                        stmod6.caption("⚠ Frequency analysis requires scipy. Install: pip install scipy")
                    except Exception as e:
                        stmod6.caption(f"⚠ Frequency analysis error: {e}")

                if not has_eval_line:
                    stmod6.caption("Eval line appears once eval_acc points exist.")
                if not has_tension:
                    stmod6.caption("Tension data not available yet; showing loss only.")
            else:
                figlos = pltx6x.line(dfobj6, x="step", y="loss", title="Loss")
                stmod6.plotly_chart(figlos, width="stretch")
                if has_tension:
                    figten = pltx6x.line(dfobj6, x="step", y="tension", title="Tension")
                    stmod6.plotly_chart(figten, width="stretch")
                else:
                    stmod6.caption("Tension data not available yet.")

            if bool(enable_3d_diag) and pltgo6x is not None:
                if not has_tension:
                    stmod6.caption("3D diagnostics waiting for tension values.")
                else:
                    t3d = dfobj6[["step", "loss", "tension"]].dropna().sort_values("step")
                    if len(t3d) > int(max_3d_points):
                        stride3d = max(1, int(len(t3d) // max(1, int(max_3d_points))))
                        t3d = t3d.iloc[::stride3d]
                    if t3d.empty:
                        stmod6.caption("3D diagnostics skipped: no valid training points.")
                    else:
                        camera_map = {
                            "isometric": {"eye": {"x": 1.45, "y": 1.45, "z": 0.95}},
                            "top": {"eye": {"x": 0.01, "y": 0.01, "z": 2.2}},
                            "side": {"eye": {"x": 2.2, "y": 0.01, "z": 0.2}},
                        }
                        cam = camera_map.get(str(camera_preset), camera_map["isometric"])
                        c_left, c_right = stmod6.columns(2)

                        with c_left:
                            fig3d_left = pltgo6x.Figure()
                            fig3d_left.add_trace(
                                pltgo6x.Scatter3d(
                                    x=t3d["step"],
                                    y=t3d["loss"],
                                    z=t3d["tension"],
                                    mode="lines",
                                    name="Training path",
                                    line={"color": "#60a5fa", "width": 4},
                                    hovertemplate=(
                                        "step=%{x}<br>loss=%{y:.6f}<br>tension=%{z:.6f}<extra>train</extra>"
                                    ),
                                )
                            )
                            fig3d_left.update_layout(
                                title="3D Train Dynamics: Step x Loss x Tension",
                                scene={
                                    "xaxis": {"title": "Step"},
                                    "yaxis": {"title": "Loss"},
                                    "zaxis": {"title": "Tension"},
                                },
                                scene_camera=cam,
                                margin={"l": 20, "r": 20, "t": 55, "b": 10},
                                legend={"orientation": "h", "y": 1.02, "x": 0.01},
                            )
                            stmod6.plotly_chart(fig3d_left, width="stretch")

                        with c_right:
                            if evdf.empty:
                                stmod6.caption("3D eval panel waiting for eval rows.")
                            else:
                                train_attach = dfobj6[["step", "tension"]].dropna().sort_values("step")
                                e3d = pd.merge_asof(
                                    evdf.sort_values("step"),
                                    train_attach,
                                    on="step",
                                    direction="backward",
                                    tolerance=(None if int(eval_attach_gap_steps) <= 0 else int(eval_attach_gap_steps)),
                                )
                                e3d = e3d.dropna(subset=["tension"])
                                if e3d.empty:
                                    stmod6.caption("3D eval panel waiting for matched tension points.")
                                else:
                                    eval_acc_num = (
                                        pd.to_numeric(e3d["eval_acc"], errors="coerce")
                                        if "eval_acc" in e3d.columns
                                        else pd.Series(index=e3d.index, dtype=float)
                                    )
                                    acc_delta_num = (
                                        pd.to_numeric(e3d["acc_delta"], errors="coerce")
                                        if "acc_delta" in e3d.columns
                                        else pd.Series(index=e3d.index, dtype=float)
                                    )
                                    chance_num = (
                                        pd.to_numeric(e3d["chance_acc"], errors="coerce")
                                        if "chance_acc" in e3d.columns
                                        else pd.Series(index=e3d.index, dtype=float)
                                    )
                                    quality = acc_delta_num.copy()
                                    miss = quality.isna()
                                    quality.loc[miss] = eval_acc_num.loc[miss] - chance_num.loc[miss]
                                    miss = quality.isna()
                                    quality.loc[miss] = eval_acc_num.loc[miss]
                                    e3d["quality"] = quality
                                    e3d = e3d.dropna(subset=["quality"]).sort_values("step")
                                    has_raw_delta = "acc_delta" in e3d.columns
                                    if not has_raw_delta or bool(pd.to_numeric(e3d["acc_delta"], errors="coerce").isna().any()):
                                        stmod6.caption(
                                            "Eval panel uses acc_delta where present; "
                                            "fallback points use eval_acc-chance (or eval_acc)."
                                        )
                                    if len(e3d) > int(max_3d_points):
                                        stride_eval = max(1, int(len(e3d) // max(1, int(max_3d_points))))
                                        e3d = e3d.iloc[::stride_eval]
                                    if e3d.empty:
                                        stmod6.caption("3D eval panel has no quality points yet.")
                                    else:
                                        fig3d_right = pltgo6x.Figure()
                                        fig3d_right.add_trace(
                                            pltgo6x.Scatter3d(
                                                x=e3d["step"],
                                                y=e3d["quality"],
                                                z=e3d["tension"],
                                                mode="lines",
                                                name="Eval quality path",
                                                line={"color": "#f43f5e", "width": 3, "dash": "dot"},
                                                hovertemplate=(
                                                    "step=%{x}<br>acc_delta=%{y:.6f}<br>"
                                                    "tension=%{z:.6f}<extra>eval-line</extra>"
                                                ),
                                            )
                                        )
                                        if bool(show_3d_eval_markers):
                                            if "eval_n" in e3d.columns:
                                                nvals = pd.to_numeric(e3d["eval_n"], errors="coerce").fillna(0.0)
                                                nmax = max(1.0, float(nvals.max()))
                                                marker_size = (4.0 + (6.0 * (nvals / nmax))).tolist()
                                            else:
                                                marker_size = 6
                                            has_eval_cols = all(
                                                col in e3d.columns
                                                for col in [
                                                    "eval_acc",
                                                    "acc_delta",
                                                    "ci95_low",
                                                    "ci95_high",
                                                    "ci99_low",
                                                    "ci99_high",
                                                    "eval_n",
                                                ]
                                            )
                                            fig3d_right.add_trace(
                                                pltgo6x.Scatter3d(
                                                    x=e3d["step"],
                                                    y=e3d["quality"],
                                                    z=e3d["tension"],
                                                    mode="markers",
                                                    name="Eval points",
                                                    marker={
                                                        "size": marker_size,
                                                        "color": e3d["quality"],
                                                        "colorscale": "Viridis",
                                                        "colorbar": {"title": "acc_delta"},
                                                        "opacity": 0.9,
                                                    },
                                                    customdata=(
                                                        e3d[
                                                            [
                                                                "eval_acc",
                                                                "acc_delta",
                                                                "ci95_low",
                                                                "ci95_high",
                                                                "ci99_low",
                                                                "ci99_high",
                                                                "eval_n",
                                                            ]
                                                        ]
                                                        if has_eval_cols
                                                        else None
                                                    ),
                                                    hovertemplate=(
                                                        "step=%{x}<br>acc_delta=%{y:.6f}<br>tension=%{z:.6f}<br>"
                                                        "eval_acc=%{customdata[0]:.6f}<br>"
                                                        "raw_delta=%{customdata[1]:.6f}<br>"
                                                        "ci95=[%{customdata[2]:.6f}, %{customdata[3]:.6f}]<br>"
                                                        "ci99=[%{customdata[4]:.6f}, %{customdata[5]:.6f}]<br>"
                                                        "n=%{customdata[6]}<extra>eval</extra>"
                                                    )
                                                    if has_eval_cols
                                                    else "step=%{x}<br>acc_delta=%{y:.6f}<br>tension=%{z:.6f}<extra>eval</extra>",
                                                )
                                            )
                                        fig3d_right.update_layout(
                                            title="3D Eval Dynamics: Step x acc_delta x Tension",
                                            scene={
                                                "xaxis": {"title": "Step"},
                                                "yaxis": {"title": "acc_delta"},
                                                "zaxis": {"title": "Tension"},
                                            },
                                            scene_camera=cam,
                                            margin={"l": 20, "r": 20, "t": 55, "b": 10},
                                            legend={"orientation": "h", "y": 1.02, "x": 0.01},
                                        )
                                        stmod6.plotly_chart(fig3d_right, width="stretch")
        except Exception as exc:
            stmod6.warning(f"Plotly render failed, falling back to table only: {exc}")
    else:
        stmod6.caption("Plotly not installed; showing table only.")

    # ============================================================
    # Comprehensive Training Telemetry (2026-02-10 redesign)
    # ============================================================
    if len(dfobj6) >= 1:  # Show as soon as we have data
        stmod6.markdown("---")
        stmod6.markdown("**Comprehensive Telemetry** — complete training state view")

        # Helper function: Tier 1 - Summary Lines
        def _render_summary_lines(df):
            """Render compact summary lines for recent steps."""
            n_lines = min(50, len(df))
            if n_lines == 0:
                stmod6.caption("No data yet.")
                return

            recent = df.tail(n_lines).copy()

            # Compute deltas
            recent['loss_delta'] = recent['loss'].diff()
            recent['loss_mean'] = recent['loss'].rolling(window=20, min_periods=1).mean()
            recent['loss_std'] = recent['loss'].rolling(window=20, min_periods=1).std()

            # Compute gradient EMA if available
            if 'gnorm' in recent.columns and recent['gnorm'].notna().any():
                recent['gnorm_ema'] = recent['gnorm'].ewm(span=20, adjust=False).mean()
            else:
                recent['gnorm_ema'] = None

            lines = []
            for idx, row in recent.iterrows():
                step = int(row['step'])
                loss = row['loss']
                loss_delta = row.get('loss_delta', 0.0) or 0.0

                # Loss change indicator
                if abs(loss_delta) < 0.001:
                    delta_sym = "="
                elif loss_delta > 0:
                    delta_sym = "↑"
                else:
                    delta_sym = "↓"

                # Color code based on deviation
                loss_mean = row.get('loss_mean', loss)
                loss_std = row.get('loss_std', 0.1) or 0.1
                deviation = abs(loss - loss_mean) / loss_std if loss_std > 0 else 0

                if deviation > 2.0:
                    color = "🔴"  # Red - spike
                elif deviation > 1.0:
                    color = "🟡"  # Yellow - warning
                else:
                    color = "🟢"  # Green - normal

                # Build line
                acc = row.get('acc', 0.0) or 0.0
                gnorm = row.get('gnorm', 0.0) or 0.0
                gnorm_ema = row.get('gnorm_ema', gnorm)
                scale = row.get('scale', 1.0) or 1.0
                inertia = row.get('inertia', 0.0) or 0.0
                deadzone = row.get('deadzone', 0.0) or 0.0
                flip_rate = row.get('flip_rate', 0.0) or 0.0
                orbit = row.get('orbit', 0.0) or 0.0
                agc_status = row.get('agc_status', 'N/A') or 'N/A'

                # Format flags
                flags = []
                if agc_status and agc_status != 'N/A' and agc_status.upper() != 'OFF':
                    flags.append(f"⚠AGC:{agc_status}")
                panic = row.get('panic')
                if panic and panic != 'None' and panic != 'False':
                    flags.append("⚠PANIC")

                flag_str = " ".join(flags) if flags else ""

                line = (
                    f"{color} {step:05d} | "
                    f"L:{loss:.4f}{delta_sym} | "
                    f"A:{acc*100:5.1f}% | "
                    f"G:{gnorm:.2f}→{gnorm_ema:.2f} | "
                    f"S:{scale:.2f} I:{inertia:.2f} D:{deadzone:.3f} | "
                    f"F:{flip_rate:.2f} O:{orbit:.1f} | "
                    f"{flag_str}"
                )
                lines.append(line)

            summary_text = "\n".join(lines)
            stmod6.code(summary_text, language=None)

        # Helper function: Tier 2 - Event Detection
        def _detect_events(df):
            """Detect anomalies and events in training."""
            if len(df) < 20:
                return []

            events = []

            # Compute rolling statistics
            df = df.copy()
            df['loss_ma'] = df['loss'].rolling(window=20, min_periods=1).mean()
            df['loss_std'] = df['loss'].rolling(window=20, min_periods=1).std()

            if 'gnorm' in df.columns:
                df['gnorm_ema'] = df['gnorm'].ewm(span=20, adjust=False).mean()

            if 'acc' in df.columns:
                df['acc_ma'] = df['acc'].rolling(window=20, min_periods=1).mean()

            # Scan for events
            for idx, row in df.iterrows():
                step = int(row['step'])

                # Loss spike detection
                if row.get('loss_std', 0) and row['loss_std'] > 0:
                    deviation = (row['loss'] - row['loss_ma']) / row['loss_std']
                    if deviation > 2.0:
                        events.append({
                            'step': step,
                            'type': 'LOSS_SPIKE',
                            'severity': deviation,
                            'details': {
                                'loss': row['loss'],
                                'loss_ma': row['loss_ma'],
                                'deviation_sigma': deviation
                            }
                        })

                # Gradient spike detection
                if 'gnorm' in row and 'gnorm_ema' in row:
                    gnorm = row.get('gnorm', 0) or 0
                    gnorm_ema = row.get('gnorm_ema', 1) or 1
                    if gnorm > 3 * gnorm_ema and gnorm > 1.0:
                        events.append({
                            'step': step,
                            'type': 'GRADIENT_SPIKE',
                            'severity': gnorm / gnorm_ema,
                            'details': {
                                'gnorm': gnorm,
                                'gnorm_ema': gnorm_ema,
                                'multiplier': gnorm / gnorm_ema
                            }
                        })

                # Accuracy drop detection
                if 'acc' in row and 'acc_ma' in row:
                    acc = row.get('acc', 0) or 0
                    acc_ma = row.get('acc_ma', 0) or 0
                    if acc_ma > 0.1 and acc < acc_ma - 0.10:
                        events.append({
                            'step': step,
                            'type': 'ACCURACY_DROP',
                            'severity': acc_ma - acc,
                            'details': {
                                'acc': acc,
                                'acc_ma': acc_ma,
                                'drop': acc_ma - acc
                            }
                        })

                # Pointer freeze detection
                if 'flip_rate' in row:
                    flip_rate = row.get('flip_rate', 0) or 0
                    if flip_rate < 0.01 and step > 100:
                        # Check if it's been frozen for a while
                        recent = df[df['step'].between(step - 10, step)]
                        if 'flip_rate' in recent.columns:
                            avg_flip = recent['flip_rate'].mean()
                            if avg_flip < 0.01:
                                events.append({
                                    'step': step,
                                    'type': 'POINTER_FREEZE',
                                    'severity': 10,  # Duration
                                    'details': {
                                        'flip_rate': flip_rate,
                                        'avg_flip_10': avg_flip
                                    }
                                })

            return events

        # Helper function: Tier 2 - Render Event Log
        def _render_event_log(events, df):
            """Render event cards with context."""
            if not events:
                stmod6.caption("No anomalies detected in current data.")
                return

            for evt in events[-20:]:  # Show last 20 events
                step = evt['step']
                evt_type = evt['type']
                severity = evt['severity']
                details = evt['details']

                # Color code by type
                if evt_type == 'LOSS_SPIKE':
                    icon = "📈"
                    color = "red"
                    desc = f"Loss spike: {details['loss']:.4f} (baseline: {details['loss_ma']:.4f}, {details['deviation_sigma']:.1f}σ)"
                elif evt_type == 'GRADIENT_SPIKE':
                    icon = "⚡"
                    color = "orange"
                    desc = f"Gradient spike: {details['gnorm']:.2f} ({details['multiplier']:.1f}x over EMA {details['gnorm_ema']:.2f})"
                elif evt_type == 'ACCURACY_DROP':
                    icon = "📉"
                    color = "yellow"
                    desc = f"Accuracy drop: {details['acc']:.2%} (baseline: {details['acc_ma']:.2%}, drop: {details['drop']:.2%})"
                elif evt_type == 'POINTER_FREEZE':
                    icon = "❄️"
                    color = "blue"
                    desc = f"Pointer frozen: flip_rate={details['flip_rate']:.3f}"
                else:
                    icon = "⚠️"
                    color = "gray"
                    desc = f"Event type: {evt_type}"

                stmod6.markdown(f"**{icon} [{evt_type} @ step {step}]**")
                stmod6.caption(desc)
                stmod6.divider()

        # Helper function: Tier 3 - Deep Telemetry
        def _render_deep_telemetry(df):
            """Render full state dump for selected step."""
            if df.empty:
                stmod6.caption("No data available.")
                return

            # Step selector
            step_options = df['step'].tolist()
            selected_step = stmod6.selectbox(
                "Select step to inspect:",
                options=step_options,
                index=len(step_options) - 1 if step_options else 0,
                key="deep_telem_step"
            )

            row = df[df['step'] == selected_step].iloc[0]

            # Build full state dump
            dump_sections = []

            # === LOSS SECTION ===
            dump_sections.append("=== LOSS ===")
            dump_sections.append(f"  total:       {row.get('loss', 0):.6f}")
            if 'acc' in row:
                dump_sections.append(f"  accuracy:    {row.get('acc', 0):.4f}")
            if 'acc_ma' in row:
                dump_sections.append(f"  acc_ma:      {row.get('acc_ma', 0):.4f}")
            dump_sections.append("")

            # === GRADIENTS SECTION ===
            dump_sections.append("=== GRADIENTS ===")
            if 'gnorm' in row and row.get('gnorm') is not None:
                dump_sections.append(f"  gnorm:       {row.get('gnorm', 0):.3f}")
            if 'grad_norm' in row and row.get('grad_norm') is not None:
                dump_sections.append(f"  theta_ptr:   {row.get('grad_norm', 0):.3f}")
            if 'scale' in row and row.get('scale') is not None:
                dump_sections.append(f"  AGC scale:   {row.get('scale', 1.0):.3f}")
            if 'agc_status' in row and row.get('agc_status') is not None:
                dump_sections.append(f"  AGC status:  {row.get('agc_status', 'N/A')}")
            dump_sections.append("")

            # === POINTER STATE SECTION ===
            dump_sections.append("=== POINTER STATE ===")
            if 'flip_rate' in row and row.get('flip_rate') is not None:
                dump_sections.append(f"  flip_rate:   {row.get('flip_rate', 0):.4f}")
            if 'orbit' in row and row.get('orbit') is not None:
                dump_sections.append(f"  orbit:       {row.get('orbit', 0):.2f}")
            if 'residual' in row and row.get('residual') is not None:
                dump_sections.append(f"  residual:    {row.get('residual', 0):.4f}")
            if 'anchor_clicks' in row and row.get('anchor_clicks') is not None:
                dump_sections.append(f"  clicks:      {row.get('anchor_clicks', 0)}")
            dump_sections.append("")

            # === CONTROL STATE SECTION ===
            dump_sections.append("=== CONTROL ===")
            if 'scale' in row and row.get('scale') is not None:
                dump_sections.append(f"  update_scale: {row.get('scale', 1.0):.3f}")
            if 'inertia' in row and row.get('inertia') is not None:
                dump_sections.append(f"  inertia:      {row.get('inertia', 0):.3f}")
            if 'deadzone' in row and row.get('deadzone') is not None:
                dump_sections.append(f"  deadzone:     {row.get('deadzone', 0):.4f}")
            if 'walk' in row and row.get('walk') is not None:
                dump_sections.append(f"  walk_prob:    {row.get('walk', 0):.3f}")
            if 'panic' in row and row.get('panic') is not None:
                dump_sections.append(f"  panic:        {row.get('panic', 'None')}")
            dump_sections.append("")

            # === TIMING SECTION ===
            dump_sections.append("=== TIMING ===")
            if 's_per_step' in row:
                dump_sections.append(f"  s_per_step:  {row.get('s_per_step', 0):.3f}s")
            dump_sections.append("")

            dump_text = "\n".join(dump_sections)
            stmod6.code(dump_text, language=None)

        # Render three tiers
        with stmod6.expander("📊 Tier 1: Summary View (Last 50 Steps)", expanded=True):
            _render_summary_lines(dfobj6)

        events = _detect_events(dfobj6)
        with stmod6.expander(f"⚡ Tier 2: Event Log ({len(events)} anomalies detected)", expanded=bool(events)):
            _render_event_log(events, dfobj6)

        with stmod6.expander("🔬 Tier 3: Deep Telemetry (Step-by-Step State Dump)", expanded=False):
            _render_deep_telemetry(dfobj6)

    # ── ARCHIVED: Ant telemetry panel (probe JSONL) ──────────────────────────────────
    # This section has been archived to tools/_backup/ant_telemetry_backup.py.txt
    # as part of the comprehensive telemetry redesign (2026-02-10).
    # See backup file to restore if needed.
    if False:  # ARCHIVED - Code preserved in backup file
        pass  # Full implementation in tools/_backup/ant_telemetry_backup.py.txt
        """
        # Original ant telemetry panel code (359 lines) archived
        # Restore from backup if needed
        ant_telem_path = str(logpth).replace(".log", "_ant_telemetry.jsonl")
        ant_rows = _read_jsonl_tail(ant_telem_path, max_lines=5000)
        # ... (see backup for full code)
        """
        n_ants_telem = 0  # Stub to prevent reference errors
        # END ARCHIVED ANT TELEMETRY SECTION

    # ── Clear display button ────────────────────────────────────────────────
    stmod6.divider()
    if stmod6.button("Clear display (keeps log data on disk)"):
        stmod6.session_state["live_log_pos"] = 0
        stmod6.session_state["live_rows"] = []
        stmod6.session_state["live_log_path"] = ""
        stmod6.rerun()

    stmod6.subheader("Latest rows")
    stmod6.dataframe(dfobj6.tail(50), width="stretch")
    if not evdf.empty:
        stmod6.subheader("Latest eval rows")
        try:
            stmod6.dataframe(evdf.tail(50), width="stretch")
        except Exception:
            pass


if __name__ == "__main__":
    main()
