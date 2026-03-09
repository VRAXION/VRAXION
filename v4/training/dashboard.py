"""VRAXION v4 Dashboard — config editor + training monitor.

Native GPU-accelerated window using DearPyGui.
Three tabs: Config (YAML editor), Training (live charts + subprocess control), Eval (placeholder).

Usage:
    python dashboard.py                        # auto-find config + log
    python dashboard.py --config path.yaml     # explicit config
    python dashboard.py --log path.csv         # explicit training log
    python dashboard.py --refresh 5            # chart refresh interval
"""

import argparse
import csv
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg
import yaml

# ── Paths ────────────────────────────────────────────────────────

V4_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = V4_ROOT / 'config' / 'vraxion_config.yaml'
DEFAULT_CSV = V4_ROOT / 'training_output' / 'train_log.csv'
TRAIN_SCRIPT = Path(__file__).resolve().parent / 'train.py'
EVAL_SCRIPT  = Path(__file__).resolve().parent / 'eval.py'
EVAL_DATA_DIR = V4_ROOT / 'eval_data'


# ══════════════════════════════════════════════════════════════════
#  YAML — load + comment-preserving save
# ══════════════════════════════════════════════════════════════════

def _load_yaml(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _fmt_yaml(value: Any) -> str:
    """Format a Python value for YAML-safe string output."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        if 0 < abs(value) < 0.01:
            return f'{value:.1e}'
        s = f'{value:g}'
        if '.' not in s and 'e' not in s:
            s += '.0'
        return s
    return str(value)


def _save_yaml_patch(path: Path, changes: dict[str, dict[str, Any]]):
    """Patch changed values in YAML, preserving all comments and formatting.

    Regex replaces the value portion of ``key: value  # comment`` lines.
    Only keys whose names are unique across the file are safe (all current
    vraxion_config.yaml keys are unique).
    """
    text = path.read_text(encoding='utf-8')
    for _section, kvs in changes.items():
        for key, value in kvs.items():
            val_str = _fmt_yaml(value)
            # Groups: (indent + key: )(value)(trailing spaces + comment)
            pattern = rf'^(\s*{re.escape(key)}:\s*)\S+(.*?)$'
            text = re.sub(pattern, rf'\g<1>{val_str}\g<2>', text,
                          count=1, flags=re.MULTILINE)
    path.write_text(text, encoding='utf-8')


# ══════════════════════════════════════════════════════════════════
#  CSV — training log reader
# ══════════════════════════════════════════════════════════════════

def _read_csv(path: Path) -> dict[str, list[float]]:
    """Read train_log.csv into column lists. Returns empty lists if missing."""
    cols: dict[str, list[float]] = {
        'step': [], 'raw_loss': [], 'masked_loss': [],
        'accuracy': [], 'masked_acc': [],
        'lr': [], 'elapsed_s': [], 'samples_seen': [], 'mask_frac': [],
    }
    if not path.exists():
        return cols
    try:
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                for key in cols:
                    cols[key].append(float(row[key]))
    except (ValueError, KeyError):
        pass  # partial write — use what we have
    return cols


# ══════════════════════════════════════════════════════════════════
#  CONFIG TAB — state + helpers
# ══════════════════════════════════════════════════════════════════

# ── Field descriptions: (section, key) → "What it is.  Tip: ..." ──
_FIELD_INFO: dict[tuple[str, str], str] = {
    # MODEL
    ('model', 'M'):
        "Ring buffer capacity — how many memory slots the experts share."
        "  Tip: Must be >= 2x seq_len. 256 covers sequences up to 128.",
    ('model', 'D'):
        "Legacy dimension — fallback when hidden_dim / slot_dim are absent."
        "  Tip: Don't touch. Use hidden_dim + slot_dim instead.",
    ('model', 'hidden_dim'):
        "Expert brain width — capacity for pattern recognition."
        "  Tip: Can be very large (4096+) without VRAM cost. This is where the intelligence lives.",
    ('model', 'slot_dim'):
        "Ring buffer cell width — how much data fits in each memory slot."
        "  Tip: Keep small (32-64). Gets cloned T x N times during backprop — directly drives VRAM.",
    ('model', 'N'):
        "Number of independent experts sharing the ring."
        "  Tip: N=2 sweet spot (speed/quality). N=6 best loss but 3x slower.",
    ('model', 'R'):
        "Attention window half-width — each expert reads 2R+1 slots around its pointer."
        "  Tip: R=2 is optimal. R=1 too narrow, R=4+ diminishing returns.",
    ('model', 'S'):
        "Context scale — how much the ring read blends into hidden state."
        "  Tip: 0.05 was tuned at D=256. May need 0.10-0.15 with large hidden_dim.",
    ('model', 'learned_jump_gate'):
        "Learned jump gate — each expert gets nn.Linear(hidden→1) to decide jump vs walk."
        "  Tip: Jump/walk is hardcoded 0.5. This would replace it with per-timestep learned prob.",
    ('model', 'jump_gate_tau'):
        "Sigmoid temperature for learned jump gate — lower = sharper decisions."
        "  Tip: 0.5 from Diamond Code. Only used when learned_jump_gate = true.",
    ('model', 'checkpoint_chunks'):
        "Gradient checkpointing chunk size for the T×N forward loop."
        "  Tip: 0 = off (fastest, most VRAM). 32 = ~13× less ring VRAM.",
    ('model', 'expert_weighting'):
        "Gradient-based expert write confidence with 1-frame delay."
        "  Tip: Experts that produce better gradients get higher write weight.",
    ('model', 'embed_encoding'):
        "Input encoding for byte tokens."
        "  Tip: 'bitlift' = byte→8 bits→Linear (18K params). 'learned' = Embedding (524K params).",
    ('model', 'output_encoding'):
        "Output projection for byte tokens."
        "  Tip: 'learned' is required — fixed encodings can't decode properly.",

    # DATA
    ('data', 'block'):
        "Pattern granularity in bytes — generators structure data in N-byte chunks."
        "  Tip: 16 is standard. Only change when designing new data tasks.",
    ('data', 'echo_repeat'):
        "How many times each block is repeated in the echo task."
        "  Tip: 8 = 87.5% predictable. Higher = easier, lower = harder.",
    ('data', 'delay_gap'):
        "Random filler blocks between original and its echo."
        "  Tip: 4 x block = 64 bytes gap. Forces ring memory use, not just local context.",
    ('data', 'flip_prob'):
        "Bit flip probability in the denoise task."
        "  Tip: 0.1 = enough corruption to challenge, not so much the signal is lost.",

    # TRAINING
    ('training', 'lr'):
        "Learning rate for Adam optimizer."
        "  Tip: 1e-3 works across all sweeps. Try 3e-4 for fine-tuning.",
    ('training', 'batch_size'):
        "Sequences per training step."
        "  Tip: Larger = stabler gradients. 32 is good default. Limited by GPU VRAM.",
    ('training', 'seq_len'):
        "Bytes per sequence — each timestep processes one byte."
        "  Tip: 128 is standard. Must be <= M/2 for ring to hold full context.",
    ('training', 'steps'):
        "Total training steps."
        "  Tip: 10K for quick experiments, 50K+ for real runs. Watch the loss curve.",
    ('training', 'log_every'):
        "CSV + console log interval (in steps)."
        "  Tip: 100 is fine. Lower (10-25) for detailed curves, higher (500+) for long runs.",
    ('training', 'save_every'):
        "Checkpoint save interval (in steps)."
        "  Tip: 50 = frequent, good for monitoring. 500-1000 for long runs to save disk.",
    ('training', 'heartbeat_every'):
        "Lightweight progress print interval — console only, not logged to CSV."
        "  Tip: 10 is a nice pulse. Just for visual comfort during training.",
    ('training', 'embed_mode'):
        "true = byte tokens (256 vocab, CrossEntropy) / false = binary bits (8-wide, MSE)."
        "  Tip: Embed mode is the default and generally better for byte-level tasks.",
    ('training', 'max_grad_norm'):
        "Gradient clipping threshold — caps global gradient norm."
        "  Tip: 1.0 prevents explosions. 0 = disabled. Don't go below 0.5.",
    ('training', 'warmup_steps'):
        "Linear LR warmup from ~0 to lr, then cosine decay."
        "  Tip: 200 is safe. Prevents early instability. 0 = constant LR (no warmup).",
    ('training', 'patience'):
        "Early stopping — save intervals without improvement before stopping."
        "  Tip: 0 = disabled. Set 10-20 for long runs to save GPU time.",
    ('training', 'data_dir'):
        "Where .traindat + .mask training files live."
        "  Tip: Relative to v4/ root. Can be a directory or a specific .traindat file.",
    ('training', 'out_dir'):
        "Where checkpoints + logs go."
        "  Tip: Relative to v4/ root. Contains ckpt_latest.pt and train_log.csv.",
    ('training', 'device'):
        "Compute device — auto picks GPU if available, else CPU."
        "  Tip: Use 'cpu' for debugging or smaller models. 'cuda' for GPU training.",
    ('training', 'cpu_threads'):
        "CPU thread count for PyTorch — only applies when device = cpu."
        "  Tip: Set to your physical core count (e.g. 18). 0 = PyTorch default.",
}

# ── Dropdown options: fields with a fixed set of valid values ──
_FIELD_OPTIONS: dict[tuple[str, str], list[str]] = {
    ('model', 'embed_encoding'):  ['bitlift', 'learned', 'hadamard', 'sincos'],
    ('model', 'output_encoding'): ['bitlift', 'learned', 'hadamard', 'sincos'],
    ('training', 'device'):       ['auto', 'cpu', 'cuda'],
}

_originals: dict[str, Any] = {}   # tag → original value
_widgets: dict[str, str] = {}     # "section.key" → tag
_dirty_theme: int = 0
_clean_theme: int = 0


def _detect_type(value: Any) -> str:
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'float'
    return 'str'


def _on_cfg_change(sender: int, app_data: Any, user_data: str):
    """Highlight changed fields yellow, reset unchanged to default."""
    tag = user_data
    current = dpg.get_value(tag)
    original = _originals.get(tag)
    if isinstance(original, float):
        dirty = abs(current - original) > 1e-10
    else:
        dirty = current != original
    dpg.bind_item_theme(tag, _dirty_theme if dirty else _clean_theme)


def _add_cfg_widget(section: str, key: str, value: Any):
    """Create the right input widget for a config value (inside active container)."""
    tag = f"cfg__{section}__{key}"
    vtype = _detect_type(value)
    _originals[tag] = value
    _widgets[f"{section}.{key}"] = tag

    kw: dict[str, Any] = dict(tag=tag, callback=_on_cfg_change, user_data=tag)
    options = _FIELD_OPTIONS.get((section, key))
    if options is not None:
        # Dropdown (combo) for fields with known valid values
        dpg.add_combo(options, label=key, default_value=str(value), width=200, **kw)
    elif vtype == 'bool':
        dpg.add_checkbox(label=key, default_value=value, **kw)
    elif vtype == 'int':
        dpg.add_input_int(label=key, default_value=value, width=200, **kw)
    elif vtype == 'float':
        dpg.add_input_float(label=key, default_value=value, width=200,
                            format='%.6g', **kw)
    else:
        dpg.add_input_text(label=key, default_value=str(value), width=200, **kw)

    # Show description + tip below the widget (dim gray, non-interactive)
    info = _FIELD_INFO.get((section, key))
    if info:
        dpg.add_text(info, color=(110, 110, 120, 180), wrap=500)
        dpg.add_spacer(height=4)


def _save_config(yaml_path: Path):
    """Gather changed values, patch YAML, reset dirty state."""
    changes: dict[str, dict[str, Any]] = {}
    for dotkey, tag in _widgets.items():
        section, key = dotkey.split('.', 1)
        current = dpg.get_value(tag)
        original = _originals[tag]
        if isinstance(original, float):
            changed = abs(current - original) > 1e-10
        else:
            changed = current != original
        if changed:
            changes.setdefault(section, {})[key] = current

    if not changes:
        dpg.set_value("cfg_status", "No changes.")
        return

    try:
        _save_yaml_patch(yaml_path, changes)
        # Reset originals + themes
        for tag in _originals:
            _originals[tag] = dpg.get_value(tag)
            dpg.bind_item_theme(tag, _clean_theme)
        n = sum(len(v) for v in changes.values())
        dpg.set_value("cfg_status", f"Saved {n} change(s).")
    except Exception as e:
        dpg.set_value("cfg_status", f"Error: {e}")


# ══════════════════════════════════════════════════════════════════
#  TRAINING SUBPROCESS — control
# ══════════════════════════════════════════════════════════════════

_proc: list = [None]          # [subprocess.Popen | None]
_output_queue: queue.Queue = queue.Queue()
_log_lines: list = []         # rolling stdout buffer (newest last)
_csv_path: list = [None]      # [Path] — set by _build_ui, used by _start_training
_nan_count: list = [0]        # consecutive NaN count from stdout

# ── Eval subprocess state ─────────────────────────────────────────
_eval_proc: list = [None]
_eval_output_queue: queue.Queue = queue.Queue()
_eval_log_lines: list = []
_eval_last_mtime: list = [0.0]   # mtime of last evaluated checkpoint
_eval_results: dict = {}         # parsed [RESULT] / [BOOT] values
_eval_history: list = []         # [{step, masked_loss, masked_acc}, ...] across all evals
_ckpt_path: list = [None]        # set in _build_ui from csv_path.parent
_status_run_theme: int = 0    # green text — running
_status_off_theme: int = 0    # gray text  — stopped
_status_ok_theme: int = 0     # bright green — done OK
_status_err_theme: int = 0    # red text   — error
_status_compile_theme: int = 0  # amber text — compile warmup


def _reader_thread(proc: Any, q: 'queue.Queue'):
    """Background thread: reads subprocess stdout line-by-line into the queue."""
    try:
        for line in proc.stdout:
            q.put(line)
    except (ValueError, OSError):
        pass
    q.put(None)  # sentinel: EOF


def _is_running() -> bool:
    return _proc[0] is not None and _proc[0].poll() is None


def _get_train_device() -> str:
    """Read current device setting from the Config tab widget."""
    tag = "cfg__training__device"
    if dpg.does_item_exist(tag):
        return str(dpg.get_value(tag)).strip().lower()
    return 'auto'


def _set_train_status(msg: str, theme: int):
    dpg.set_value("train_ctrl_status", msg)
    dpg.bind_item_theme("train_ctrl_status", theme)


def _update_train_buttons():
    running = _is_running()
    dpg.configure_item("btn_start", enabled=not running)
    dpg.configure_item("btn_stop", enabled=running)


def _start_training():
    if _is_running():
        return
    if not TRAIN_SCRIPT.exists():
        _set_train_status("x train.py not found", _status_err_theme)
        return
    # Drain leftover output from previous run
    _log_lines.clear()
    while not _output_queue.empty():
        try:
            _output_queue.get_nowait()
        except queue.Empty:
            break
    # Read controls from dashboard inputs
    log_every = dpg.get_value("ctrl_log_every") if dpg.does_item_exist("ctrl_log_every") else 10
    device_tag = "cfg__training__device"
    device = dpg.get_value(device_tag) if dpg.does_item_exist(device_tag) else "auto"
    cmd = [sys.executable, str(TRAIN_SCRIPT),
           '--log-every', str(int(log_every)),
           '--device', str(device).strip()]

    # Pass --embed if embed_mode is true in config
    embed_tag = "cfg__training__embed_mode"
    if dpg.does_item_exist(embed_tag) and dpg.get_value(embed_tag):
        cmd.append('--embed')

    # Pass --compile if compile is true in config
    compile_tag = "cfg__training__compile"
    if dpg.does_item_exist(compile_tag) and dpg.get_value(compile_tag):
        cmd.append('--compile')

    # Pass --out so train.py writes to the correct directory
    csv_p = _csv_path[0]
    if csv_p is not None:
        try:
            out_rel = str(csv_p.parent.relative_to(V4_ROOT))
            cmd.extend(['--out', out_rel])
        except ValueError:
            cmd.extend(['--out', str(csv_p.parent)])

    # Pass --data from config
    data_tag = "cfg__training__data_dir"
    if dpg.does_item_exist(data_tag):
        data_val = str(dpg.get_value(data_tag)).strip()
        if data_val:
            cmd.extend(['--data', data_val])

    # Auto-resume from latest checkpoint if one exists
    ckpt_latest = _ckpt_path[0]
    if ckpt_latest and ckpt_latest.exists():
        cmd.extend(['--resume', str(ckpt_latest)])
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    try:
        _proc[0] = subprocess.Popen(
            cmd, cwd=str(V4_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        threading.Thread(
            target=_reader_thread, args=(_proc[0], _output_queue),
            daemon=True,
        ).start()
        _set_train_status("* Running...", _status_run_theme)
    except Exception as e:
        _set_train_status(f"x {e}", _status_err_theme)
    _update_train_buttons()


def _stop_training():
    if not _is_running():
        return
    _proc[0].terminate()
    _proc[0] = None
    _set_train_status("o Stopped", _status_off_theme)
    _update_train_buttons()


def _restart_training():
    _stop_training()
    _start_training()


# ══════════════════════════════════════════════════════════════════
#  EVAL SUBPROCESS — control + result parsing
# ══════════════════════════════════════════════════════════════════

def _eval_is_running() -> bool:
    return _eval_proc[0] is not None and _eval_proc[0].poll() is None


def _parse_eval_line(line: str):
    """Extract step and [RESULT] values into _eval_results."""
    if '[BOOT] Step' in line:
        parts = line.split()
        try:
            _eval_results['step'] = parts[parts.index('Step') + 1]
        except (ValueError, IndexError):
            pass
    elif line.startswith('[RESULT]'):
        parts = line.split(None, 2)
        if len(parts) >= 3:
            key = parts[1]
            val = parts[2].split()[0]   # first token = the number
            _eval_results[key] = val


def _update_eval_display():
    """Push parsed _eval_results into the Eval tab widgets and history charts."""
    step = _eval_results.get('step', '–')
    dpg.set_value("eval_last_step",    f"Last eval: step {step}")
    dpg.set_value("eval_masked_loss",  f"masked_loss   {_eval_results.get('masked_loss', '–')}")
    dpg.set_value("eval_masked_acc",   f"masked_acc    {_eval_results.get('masked_acc', '–')}")
    dpg.set_value("eval_accuracy",     f"accuracy      {_eval_results.get('accuracy', '–')}")
    dpg.set_value("eval_wall_time",    f"wall_time     {_eval_results.get('wall_time', '–')}")

    # Append to history and update charts
    try:
        s  = int(_eval_results['step'])
        ml = float(_eval_results['masked_loss'])
        ma = float(_eval_results['masked_acc'])
        _eval_history.append({'step': s, 'masked_loss': ml, 'masked_acc': ma})
        steps  = [h['step']        for h in _eval_history]
        losses = [h['masked_loss'] for h in _eval_history]
        accs   = [h['masked_acc']  for h in _eval_history]
        dpg.set_value("eval_line_loss", [steps, losses])
        dpg.set_value("eval_line_acc",  [steps, accs])
        for ax in ('eval_loss_x', 'eval_loss_y', 'eval_acc_x', 'eval_acc_y'):
            dpg.fit_axis_data(ax)
    except (KeyError, ValueError, TypeError):
        pass


def _start_eval():
    if _eval_is_running():
        return
    ckpt = _ckpt_path[0]
    if not ckpt or not ckpt.exists():
        dpg.set_value("eval_status", "No checkpoint found")
        dpg.bind_item_theme("eval_status", _status_err_theme)
        return
    if not EVAL_SCRIPT.exists():
        dpg.set_value("eval_status", "eval.py not found")
        dpg.bind_item_theme("eval_status", _status_err_theme)
        return
    if not EVAL_DATA_DIR.exists():
        dpg.set_value("eval_status", "eval_data/ not found")
        dpg.bind_item_theme("eval_status", _status_err_theme)
        return

    # Drain leftover output from previous eval
    _eval_log_lines.clear()
    _eval_results.clear()
    while not _eval_output_queue.empty():
        try:
            _eval_output_queue.get_nowait()
        except queue.Empty:
            break

    samples = 128
    if dpg.does_item_exist("eval_samples"):
        samples = int(dpg.get_value("eval_samples"))
    eval_seq = 256
    if dpg.does_item_exist("eval_seq_len"):
        eval_seq = int(dpg.get_value("eval_seq_len"))
    cmd = [sys.executable, str(EVAL_SCRIPT),
           '--checkpoint', str(ckpt),
           '--data', str(EVAL_DATA_DIR),
           '--samples', str(samples),
           '--seq_len', str(eval_seq)]
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    try:
        _eval_proc[0] = subprocess.Popen(
            cmd, cwd=str(V4_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        threading.Thread(
            target=_reader_thread, args=(_eval_proc[0], _eval_output_queue),
            daemon=True,
        ).start()
        dpg.set_value("eval_status", "* Running...")
        dpg.bind_item_theme("eval_status", _status_run_theme)
        dpg.configure_item("btn_eval_now", enabled=False)
    except Exception as e:
        dpg.set_value("eval_status", f"Error: {e}")
        dpg.bind_item_theme("eval_status", _status_err_theme)


# ══════════════════════════════════════════════════════════════════
#  BUILD UI — tabs + themes + refresh loop
# ══════════════════════════════════════════════════════════════════

def _build_ui(csv_path: Path, yaml_path: Path, refresh_s: float):
    global _dirty_theme, _clean_theme
    global _status_run_theme, _status_off_theme, _status_ok_theme, _status_err_theme, _status_compile_theme

    # Set paths derived from csv output directory
    _csv_path[0] = csv_path
    _ckpt_path[0] = csv_path.parent / 'ckpt_latest.pt'
    # Seed last-mtime so auto-eval doesn't fire on existing checkpoints at startup
    if _ckpt_path[0].exists():
        _eval_last_mtime[0] = _ckpt_path[0].stat().st_mtime

    dpg.create_context()

    # ── Global theme: dark background ──
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (24, 24, 28))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 30, 36))

    # ── Config dirty/clean themes ──
    with dpg.theme() as _dirty_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (80, 70, 0))
    with dpg.theme() as _clean_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (37, 37, 38))

    # ── Training status color themes ──
    with dpg.theme() as _status_run_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (80, 220, 80))
    with dpg.theme() as _status_off_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (140, 140, 140))
    with dpg.theme() as _status_ok_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 255, 100))
    with dpg.theme() as _status_err_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 80, 80))
    with dpg.theme() as _status_compile_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 200, 50))

    # ── Line series color themes ──
    line_themes = {}
    for name, color in [('blue', (100, 149, 237)), ('red', (255, 99, 71)),
                        ('green', (50, 205, 50)), ('gold', (255, 215, 0)),
                        ('purple', (147, 112, 219))]:
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color,
                                    category=dpg.mvThemeCat_Plots)
        line_themes[name] = t

    # ══════════════════════════════════════════════════════════════
    #  Main window with tab bar
    # ══════════════════════════════════════════════════════════════

    with dpg.window(tag="main", label="VRAXION Dashboard"):
        with dpg.tab_bar(tag="tab_bar"):

            # ═══ CONFIG TAB ══════════════════════════════════════
            with dpg.tab(label="Config"):
                dpg.add_text(f"Config: {yaml_path.name}")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save",
                                   callback=lambda: _save_config(yaml_path))
                    dpg.add_text("", tag="cfg_status")
                dpg.add_separator()

                cfg = _load_yaml(yaml_path)
                for section_name in ('model', 'data', 'training'):
                    section = cfg.get(section_name, {})
                    if not isinstance(section, dict):
                        continue
                    with dpg.collapsing_header(label=section_name.upper(),
                                               default_open=True):
                        for key, value in section.items():
                            if value is None:
                                continue
                            _add_cfg_widget(section_name, key, value)

            # ═══ TRAINING TAB ════════════════════════════════════
            with dpg.tab(label="Training"):
                # ── Control bar ──────────────────────────────────
                with dpg.group(horizontal=True):
                    dpg.add_button(label="  Start  ", tag="btn_start",
                                   callback=_start_training)
                    dpg.add_button(label="  Stop  ", tag="btn_stop",
                                   callback=_stop_training, enabled=False)
                    dpg.add_button(label="  Restart  ", tag="btn_restart",
                                   callback=_restart_training)
                    dpg.add_text("   log every")
                    dpg.add_input_int(tag="ctrl_log_every", default_value=10,
                                      width=70, min_value=1, max_value=10000,
                                      min_clamped=True, max_clamped=True)
                    dpg.add_text("steps    ")
                    dpg.add_text("o Stopped", tag="train_ctrl_status")
                dpg.add_separator()

                # ── Live stats line ──────────────────────────────
                dpg.add_text("Waiting for data...", tag="status_text")
                dpg.add_separator()

                # ── Train output (stdout pipe) ───────────────────
                with dpg.collapsing_header(label="Train Output", default_open=True):
                    dpg.add_input_text(
                        tag="train_log_text",
                        multiline=True, readonly=True,
                        width=-1, height=160,
                        default_value="",
                    )
                dpg.add_separator()

                # Loss
                with dpg.child_window(width=-1, height=280, border=False):
                    with dpg.plot(label="Loss", width=-1, height=260,
                                  tag="plot_loss"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="step",
                                          tag="loss_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="loss",
                                           tag="loss_y", log_scale=True):
                            dpg.add_line_series([], [], label="raw",
                                                tag="line_raw_loss")
                            dpg.add_line_series([], [], label="masked",
                                                tag="line_masked_loss")

                # Accuracy
                with dpg.child_window(width=-1, height=280, border=False):
                    with dpg.plot(label="Accuracy", width=-1, height=260,
                                  tag="plot_acc"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="step",
                                          tag="acc_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="accuracy",
                                           tag="acc_y"):
                            dpg.add_line_series([], [], label="raw",
                                                tag="line_raw_acc")
                            dpg.add_line_series([], [], label="masked",
                                                tag="line_masked_acc")

                # Learning rate
                with dpg.child_window(width=-1, height=280, border=False):
                    with dpg.plot(label="Learning Rate", width=-1, height=260,
                                  tag="plot_lr"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="step",
                                          tag="lr_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="lr",
                                           tag="lr_y", log_scale=True):
                            dpg.add_line_series([], [], label="lr",
                                                tag="line_lr")

                # Mask fraction
                with dpg.child_window(width=-1, height=280, border=False):
                    with dpg.plot(label="Mask Fraction", width=-1, height=260,
                                  tag="plot_mf"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="step",
                                          tag="mf_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="mask_frac",
                                           tag="mf_y"):
                            dpg.add_line_series([], [], label="mask",
                                                tag="line_mf")

            # ═══ EVAL TAB ════════════════════════════════════════
            with dpg.tab(label="Eval"):
                # ── Control bar ──────────────────────────────────
                with dpg.group(horizontal=True):
                    dpg.add_button(label="  Run Now  ", tag="btn_eval_now",
                                   callback=_start_eval)
                    dpg.add_checkbox(label="Auto (new checkpoint)",
                                     tag="eval_auto", default_value=True)
                    dpg.add_text("  Samples:")
                    dpg.add_input_int(tag="eval_samples", default_value=128,
                                      min_value=16, max_value=4096,
                                      min_clamped=True, max_clamped=True,
                                      width=80)
                    dpg.add_text("  Seq Len:")
                    dpg.add_input_int(tag="eval_seq_len", default_value=256,
                                      min_value=64, max_value=4096,
                                      min_clamped=True, max_clamped=True,
                                      width=80)
                    dpg.add_text("    ")
                    dpg.add_text("Idle", tag="eval_status")
                dpg.add_separator()

                # ── Last result ───────────────────────────────────
                dpg.add_text("Last eval: –", tag="eval_last_step")
                dpg.add_separator()
                dpg.add_text("masked_loss   –", tag="eval_masked_loss")
                dpg.add_text("masked_acc    –", tag="eval_masked_acc")
                dpg.add_text("accuracy      –", tag="eval_accuracy")
                dpg.add_text("wall_time     –", tag="eval_wall_time")
                dpg.add_separator()

                # ── Eval history charts ───────────────────────────
                with dpg.child_window(width=-1, height=250, border=False):
                    with dpg.plot(label="Eval Loss (held-out)",
                                  width=-1, height=230, tag="eval_plot_loss"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="step",
                                          tag="eval_loss_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="masked_loss",
                                           tag="eval_loss_y", log_scale=True):
                            dpg.add_line_series([], [], label="masked_loss",
                                                tag="eval_line_loss")

                with dpg.child_window(width=-1, height=250, border=False):
                    with dpg.plot(label="Eval Accuracy (held-out)",
                                  width=-1, height=230, tag="eval_plot_acc"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="step",
                                          tag="eval_acc_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="masked_acc",
                                           tag="eval_acc_y"):
                            dpg.add_line_series([], [], label="masked_acc",
                                                tag="eval_line_acc")
                dpg.add_separator()

                # ── Eval output log ───────────────────────────────
                with dpg.collapsing_header(label="Eval Output",
                                           default_open=True):
                    dpg.add_input_text(
                        tag="eval_log_text",
                        multiline=True, readonly=True,
                        width=-1, height=300,
                        default_value="",
                    )

    dpg.bind_theme(global_theme)
    dpg.set_primary_window("main", True)

    # ── Bind line colors ──
    dpg.bind_item_theme("line_raw_loss", line_themes['blue'])
    dpg.bind_item_theme("line_masked_loss", line_themes['red'])
    dpg.bind_item_theme("line_raw_acc", line_themes['blue'])
    dpg.bind_item_theme("line_masked_acc", line_themes['green'])
    dpg.bind_item_theme("line_lr", line_themes['gold'])
    dpg.bind_item_theme("line_mf", line_themes['purple'])

    # ── Bind eval chart colors ──
    dpg.bind_item_theme("eval_line_loss", line_themes['red'])
    dpg.bind_item_theme("eval_line_acc",  line_themes['green'])

    # ── Bind initial training status theme (gray = stopped) ──
    dpg.bind_item_theme("train_ctrl_status", _status_off_theme)

    # ══════════════════════════════════════════════════════════════
    #  Chart refresh logic
    # ══════════════════════════════════════════════════════════════

    _last_n = [0]

    def _refresh():
        data = _read_csv(csv_path)
        n = len(data['step'])
        if n == 0:
            dpg.set_value("status_text",
                          f"Waiting for data... ({csv_path.name})")
            return
        if n == _last_n[0]:
            return
        _last_n[0] = n

        steps = data['step']
        dpg.set_value("line_raw_loss", [steps, data['raw_loss']])
        dpg.set_value("line_masked_loss", [steps, data['masked_loss']])
        dpg.set_value("line_raw_acc", [steps, data['accuracy']])
        dpg.set_value("line_masked_acc", [steps, data['masked_acc']])
        dpg.set_value("line_lr", [steps, data['lr']])
        dpg.set_value("line_mf", [steps, data['mask_frac']])

        for ax in ('loss_x', 'loss_y', 'acc_x', 'acc_y',
                    'lr_x', 'lr_y', 'mf_x', 'mf_y'):
            dpg.fit_axis_data(ax)

        s = int(steps[-1])
        ml = data['masked_loss'][-1]
        bl = min(data['masked_loss'])
        ma = data['masked_acc'][-1]
        lr = data['lr'][-1]
        el = data['elapsed_s'][-1]
        sps = el / s if s > 0 else 0
        dpg.set_value("status_text",
                      f"step {s}  |  loss {ml:.4f}  |  best {bl:.4f}  |  "
                      f"acc {ma:.1%}  |  lr {lr:.6f}  |  {sps:.1f}s/step  "
                      f"|  {csv_path.name}")

    _refresh()

    # Timer-based auto-refresh
    _elapsed = [0.0]

    def _tick():
        _elapsed[0] += dpg.get_delta_time()
        if _elapsed[0] >= refresh_s:
            _elapsed[0] = 0.0
            _refresh()

        # Poll subprocess for natural completion (stop sets _proc[0]=None so won't fire)
        if _proc[0] is not None:
            rc = _proc[0].poll()
            if rc is not None:
                if rc == 0:
                    _set_train_status("Done (OK)", _status_ok_theme)
                else:
                    _set_train_status(f"Error (rc={rc})", _status_err_theme)
                _proc[0] = None   # clear so we don't poll again
                _update_train_buttons()

        # Drain training stdout queue → Train Output panel (newest at top)
        new_lines = []
        while True:
            try:
                line = _output_queue.get_nowait()
                if line is None:
                    break
                new_lines.append(line.rstrip('\n'))
            except queue.Empty:
                break
        if new_lines:
            # Parse lines for status signals
            for line in new_lines:
                if '[NAN]' in line:
                    m = re.search(r'\((\d+) consecutive\)', line)
                    if m:
                        _nan_count[0] = int(m.group(1))
                    _set_train_status(
                        f"! NaN detected ({_nan_count[0]} consecutive)",
                        _status_err_theme)
                elif 'Unable to hit fast path of CUDAGraphs' in line:
                    _set_train_status(
                        "! CUDA graph issue -- may need restart",
                        _status_err_theme)
                elif '[RING-PROBE]' in line or '[compile]' in line:
                    _set_train_status(
                        "* Compiling (warmup)...", _status_compile_theme)
                elif 's/step' in line and _is_running():
                    m = re.search(r'(\d+\.\d+)s/step', line)
                    if m:
                        sps = float(m.group(1))
                        if sps < 1.0:
                            _set_train_status(
                                "* Training (compiled)", _status_run_theme)
                            _nan_count[0] = 0
                        elif sps > 5.0:
                            _set_train_status(
                                "* Compiling (warmup)...",
                                _status_compile_theme)

            _log_lines.extend(new_lines)
            del _log_lines[:-80]
            dpg.set_value("train_log_text",
                          '\n'.join(reversed(_log_lines)))

        # ── Eval subprocess poll + drain ──────────────────────────
        new_eval_lines = []
        while True:
            try:
                line = _eval_output_queue.get_nowait()
                if line is None:
                    break
                new_eval_lines.append(line.rstrip('\n'))
            except queue.Empty:
                break
        if new_eval_lines:
            for line in new_eval_lines:
                _parse_eval_line(line)
            _eval_log_lines.extend(new_eval_lines)
            del _eval_log_lines[:-80]
            dpg.set_value("eval_log_text",
                          '\n'.join(reversed(_eval_log_lines)))

        if _eval_proc[0] is not None:
            rc = _eval_proc[0].poll()
            if rc is not None:
                _eval_proc[0] = None
                dpg.configure_item("btn_eval_now", enabled=True)
                if rc == 0:
                    _update_eval_display()
                    dpg.set_value("eval_status", "Done")
                    dpg.bind_item_theme("eval_status", _status_ok_theme)
                else:
                    dpg.set_value("eval_status", f"Error (rc={rc})")
                    dpg.bind_item_theme("eval_status", _status_err_theme)

        # ── Auto-eval: trigger when checkpoint is updated ─────────
        # Skip auto-eval while training is running on CPU — both processes
        # would compete for CPU and slow each other down.
        cpu_training = _is_running() and _get_train_device() == 'cpu'
        if not cpu_training and not _eval_is_running() and dpg.get_value("eval_auto"):
            ckpt = _ckpt_path[0]
            if ckpt and ckpt.exists():
                mtime = ckpt.stat().st_mtime
                if mtime > _eval_last_mtime[0]:
                    _eval_last_mtime[0] = mtime
                    _start_eval()

    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_F5,
                                  callback=lambda: _refresh())

    # ── Viewport + render loop ──
    dpg.create_viewport(title="VRAXION Dashboard", width=1200, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        _tick()
        dpg.render_dearpygui_frame()

    # Terminate subprocesses if still running when dashboard closes
    if _is_running():
        print("[DASHBOARD] Terminating training subprocess...")
        _proc[0].terminate()
    if _eval_is_running():
        print("[DASHBOARD] Terminating eval subprocess...")
        _eval_proc[0].terminate()

    dpg.destroy_context()


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='VRAXION Dashboard')
    ap.add_argument('--config', default=None,
                    help='path to vraxion_config.yaml (default: auto-detect)')
    ap.add_argument('--log', default=None,
                    help='path to train_log.csv (default: auto-detect)')
    ap.add_argument('--out', default=None,
                    help='training output directory (overrides config out_dir)')
    ap.add_argument('--refresh', type=float, default=5.0,
                    help='chart refresh interval in seconds (default: 5)')
    args = ap.parse_args()

    yaml_path = Path(args.config) if args.config else DEFAULT_CONFIG

    # Resolve CSV path: --log > --out > config out_dir > auto-detect
    if args.log:
        csv_path = Path(args.log)
    elif args.out:
        out_p = Path(args.out)
        if not out_p.is_absolute():
            out_p = V4_ROOT / out_p
        csv_path = out_p / 'train_log.csv'
    else:
        # Auto-detect from config's out_dir — check subdirs for most recent log
        try:
            cfg = _load_yaml(yaml_path)
            out_dir = cfg.get('training', {}).get('out_dir', 'training_output')
        except Exception:
            out_dir = 'training_output'
        out_path = Path(out_dir)
        if not out_path.is_absolute():
            out_path = V4_ROOT / out_path
        candidates = list(out_path.glob('*/train_log.csv'))
        if candidates:
            csv_path = max(candidates, key=lambda p: p.stat().st_mtime)
        elif (out_path / 'train_log.csv').exists():
            csv_path = out_path / 'train_log.csv'
        else:
            csv_path = DEFAULT_CSV

    if not yaml_path.exists():
        print(f'[ERROR] Config not found: {yaml_path}')
        print('  Use --config to specify the path.')
        sys.exit(1)

    csv_note = '' if csv_path.exists() else ' (waiting for training)'
    print(f'[DASHBOARD] Config:  {yaml_path}')
    print(f'[DASHBOARD] Log:     {csv_path}{csv_note}')
    print(f'[DASHBOARD] Refresh: {args.refresh}s  |  F5 = manual  |  Ctrl+Q = quit')
    print(f'[DASHBOARD] Train:   {TRAIN_SCRIPT}')
    _build_ui(csv_path, yaml_path, args.refresh)


if __name__ == '__main__':
    main()
