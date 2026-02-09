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

        raw_delta = _tryflt(rawmat.group("raw_delta")) if rawmat is not None else None
        shard_count = _tryflt(shdmat.group("shard_count")) if shdmat is not None else None
        shard_size = _tryflt(shdmat.group("shard_size")) if shdmat is not None else None
        trac6x = _tryflt(trcmat.group("traction")) if trcmat is not None else None

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
    dfobj6["tension"] = dfobj6["grad_norm"] * dfobj6["raw_delta"] / 100.0

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

    stmod6.set_page_config(page_title="VRAXION Live Dashboard", layout="wide")
    stmod6.title("VRAXION Live Dashboard")

    logpth = stmod6.sidebar.text_input("Log path", value=str(argobj.log))
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
    eval_stream_path = stmod6.sidebar.text_input("Eval stream path", value=str(default_eval_stream))
    eval_status_path = stmod6.sidebar.text_input("Eval status path", value=str(default_eval_status))
    rfrsec = stmod6.sidebar.number_input(
        "Refresh interval (sec)",
        min_value=0,
        max_value=600,
        value=int(argobj.refresh),
    )
    maxrow = stmod6.sidebar.number_input(
        "Max rows",
        min_value=0,
        max_value=500000,
        value=int(argobj.max_rows),
    )
    show_separate_charts = stmod6.sidebar.checkbox(
        "Show separate loss/tension charts",
        value=False,
    )
    enable_3d_diag = stmod6.sidebar.checkbox(
        "Enable 3D diagnostics",
        value=True,
    )
    show_3d_eval_markers = stmod6.sidebar.checkbox(
        "Show 3D eval markers",
        value=True,
    )
    max_3d_points = int(
        stmod6.sidebar.number_input(
            "3D max points",
            min_value=200,
            max_value=20000,
            value=1500,
            step=100,
        )
    )
    eval_attach_gap_steps = int(
        stmod6.sidebar.number_input(
            "3D eval gap cap (steps)",
            min_value=0,
            max_value=5000,
            value=100,
            step=10,
        )
    )
    camera_preset = stmod6.sidebar.selectbox(
        "3D camera",
        options=["isometric", "top", "side"],
        index=0,
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
            if "grad_norm" in dfobj6.columns and "raw_delta" in dfobj6.columns:
                dfobj6["tension"] = dfobj6["grad_norm"] * dfobj6["raw_delta"] / 100.0
                capval = dfobj6["tension"].quantile(0.99)
                dfobj6["tension"] = dfobj6["tension"].clip(upper=capval)
    except Exception as exc:
        stmod6.error(f"Failed to parse log: {exc}")
        return

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
                stmod6.plotly_chart(figov, use_container_width=True)
                if not has_eval_line:
                    stmod6.caption("Eval line appears once eval_acc points exist.")
                if not has_tension:
                    stmod6.caption("Tension data not available yet; showing loss only.")
            else:
                figlos = pltx6x.line(dfobj6, x="step", y="loss", title="Loss")
                stmod6.plotly_chart(figlos, use_container_width=True)
                if has_tension:
                    figten = pltx6x.line(dfobj6, x="step", y="tension", title="Tension")
                    stmod6.plotly_chart(figten, use_container_width=True)
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
                            stmod6.plotly_chart(fig3d_left, use_container_width=True)

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
                                        stmod6.plotly_chart(fig3d_right, use_container_width=True)
        except Exception as exc:
            stmod6.warning(f"Plotly render failed, falling back to table only: {exc}")
    else:
        stmod6.caption("Plotly not installed; showing table only.")

    stmod6.subheader("Latest rows")
    stmod6.dataframe(dfobj6.tail(50), use_container_width=True)
    if not evdf.empty:
        stmod6.subheader("Latest eval rows")
        try:
            stmod6.dataframe(evdf.tail(50), use_container_width=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
