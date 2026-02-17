"""
Diamond Code — InfluxDB Metrics Writer

Thin wrapper for writing training metrics to InfluxDB for Grafana visualization.
Uses ASYNCHRONOUS batch writes (non-blocking, no training loop slowdown).

Graceful degradation: if INFLUX_TOKEN env var not set, all calls are no-ops.

Setup:
    1. Start InfluxDB:  influxd.exe
    2. Create org "vraxion", bucket "diamond" at http://localhost:8086
    3. Set env var:  set INFLUX_TOKEN=<your-token>
    4. Training loop calls init() once, log_step()/log_being()/log_bits() per step
"""

import time
import os

_client = None
_write_api = None
_bucket = "diamond"
_org = "vraxion"
_url = "http://localhost:8086"
_available = False

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import ASYNCHRONOUS
    _available = True
except ImportError:
    _available = False


def init(url="http://localhost:8086", token=None, org="vraxion", bucket="diamond"):
    """Initialize InfluxDB client. No-op if token not set or library missing."""
    global _client, _write_api, _bucket, _org, _url
    if not _available:
        return
    token = token or os.environ.get("INFLUX_TOKEN")
    if not token:
        # fallback: read from token file next to this script
        _token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".influx_token")
        if os.path.exists(_token_file):
            with open(_token_file) as f:
                token = f.read().strip()
    if not token:
        return
    _bucket = bucket
    _org = org
    _url = url
    try:
        _client = InfluxDBClient(url=url, token=token, org=org)
        _write_api = _client.write_api(write_options=ASYNCHRONOUS)
    except Exception:
        pass  # silently fail — don't break training


def flush_bucket():
    """Delete all data from the diamond bucket. Call before each new run.
    Ensures Grafana always shows only the current run's data."""
    if not _client:
        return
    try:
        from datetime import datetime, timezone
        delete_api = _client.delete_api()
        delete_api.delete(
            start=datetime(1970, 1, 1, tzinfo=timezone.utc),
            stop=datetime(2099, 12, 31, tzinfo=timezone.utc),
            predicate='',
            bucket=_bucket,
            org=_org
        )
    except Exception as e:
        print(f"  [influx] flush failed: {e}")


def harvest_run(archive_dir, config_name="unknown"):
    """Export current run's training curve from InfluxDB to archive files.
    archive_dir: path to write archive files.
    config_name: descriptive name for the run.
    Returns summary dict or None on failure.
    """
    if not _client:
        return None
    try:
        import json as _json
        query_api = _client.query_api()
        tables = query_api.query(f'''
            from(bucket: "{_bucket}")
            |> range(start: -30d)
            |> filter(fn: (r) => r._measurement == "training_step")
            |> filter(fn: (r) => r.is_eval == "false")
            |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["step"])
        ''')
        os.makedirs(archive_dir, exist_ok=True)

        # Collect training curve rows
        rows = []
        for table in tables:
            for record in table.records:
                rows.append({
                    'step': int(record.values.get('step', 0)),
                    'loss': float(record.values.get('loss', 0)),
                    'bit_acc': float(record.values.get('bit_acc', 0)),
                    'byte_match': float(record.values.get('byte_match', 0)),
                    'oracle': float(record.values.get('oracle', 0)),
                    's_per_step': float(record.values.get('s_per_step', 0)),
                })

        # Write training curve
        if rows:
            curve_file = os.path.join(archive_dir, "training_curve.json")
            with open(curve_file, 'w') as f:
                _json.dump(rows, f)

        # Write summary
        summary = {
            'config_name': config_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_steps': len(rows),
            'final_loss': rows[-1]['loss'] if rows else None,
            'final_bit_acc': rows[-1]['bit_acc'] if rows else None,
            'final_oracle': rows[-1]['oracle'] if rows else None,
        }
        summary_file = os.path.join(archive_dir, "summary.json")
        with open(summary_file, 'w') as f:
            _json.dump(summary, f, indent=2)

        return summary
    except Exception as e:
        print(f"  [influx] harvest failed: {e}")
        return None


def log_step(run_id, step, loss, bit_acc=0, byte_match=0, oracle=0,
             bit_oracle=0, ensemble_benefit=0, coverage=0, clustering=0,
             circular_spread=0, s_per_step=0, is_eval=False, n_bits=0,
             gate_entropy=0, effort_level=-1, think_ticks=0,
             batch_size=0, use_lcx=0, effort_name="", current_stage="", **kwargs):
    """Log one training step. Non-blocking."""
    if not _write_api:
        return
    p = (Point("training_step")
         .tag("run_id", run_id)
         .tag("is_eval", "true" if is_eval else "false")
         .tag("effort_name", str(effort_name) if effort_name else "")
         .field("step", int(step))
         .field("loss", float(loss))
         .field("bit_acc", float(bit_acc))
         .field("byte_match", float(byte_match))
         .field("oracle", float(oracle))
         .field("bit_oracle", float(bit_oracle))
         .field("ensemble_benefit", float(ensemble_benefit))
         .field("coverage", float(coverage))
         .field("clustering", float(clustering))
         .field("circular_spread", float(circular_spread))
         .field("s_per_step", float(s_per_step))
         .field("n_bits", int(n_bits))
         .field("gate_entropy", float(gate_entropy))
         .field("effort_level", int(effort_level))
         .field("think_ticks", int(think_ticks))
         .field("batch_size", int(batch_size))
         .field("use_lcx", int(use_lcx))
         .field("current_stage", str(current_stage) if current_stage else "")
         .field("agc_norm", float(kwargs.get('agc_norm', 0)))
         .field("agc_scale", float(kwargs.get('agc_scale', 1.0)))
         .time(time.time_ns(), WritePrecision.NS))
    _write_api.write(bucket=_bucket, record=p)


def log_dream(run_id, step, dream_step, dream_mode="consolidation",
              dream_loss=0, dream_bit_acc=0, dream_lcx_norm=0,
              dream_zoom_gate=0, dream_step_time=0, dream_think_ticks=0,
              dream_binarized=False, dream_score_margin=0):
    """Log one dream step. Non-blocking."""
    if not _write_api:
        return
    p = (Point("dream_step")
         .tag("run_id", run_id)
         .tag("dream_mode", dream_mode)
         .field("step", int(step))
         .field("dream_step", int(dream_step))
         .field("dream_loss", float(dream_loss))
         .field("dream_bit_acc", float(dream_bit_acc))
         .field("dream_lcx_norm", float(dream_lcx_norm))
         .field("dream_zoom_gate", float(dream_zoom_gate))
         .field("dream_step_time", float(dream_step_time))
         .field("dream_think_ticks", int(dream_think_ticks))
         .field("dream_binarized", 1 if dream_binarized else 0)
         .field("dream_score_margin", float(dream_score_margin))
         .time(time.time_ns(), WritePrecision.NS))
    _write_api.write(bucket=_bucket, record=p)


def log_being(run_id, step, being_id, accuracy=0, masked_acc=0,
              jump_rate=0, k_bits=0, unique_bits=0, redundant_bits=0,
              ctx_scale=0):
    """Log per-being metrics."""
    if not _write_api:
        return
    p = (Point("being_metric")
         .tag("run_id", run_id)
         .tag("being_id", str(being_id))
         .field("step", int(step))
         .field("accuracy", float(accuracy))
         .field("masked_acc", float(masked_acc))
         .field("jump_rate", float(jump_rate))
         .field("k_bits", int(k_bits))
         .field("unique_bits", int(unique_bits))
         .field("redundant_bits", int(redundant_bits))
         .field("ctx_scale", float(ctx_scale))
         .time(time.time_ns(), WritePrecision.NS))
    _write_api.write(bucket=_bucket, record=p)


def log_bits(run_id, step, bit_accs):
    """Log per-bit accuracy (for heatmap). bit_accs is a list of floats."""
    if not _write_api:
        return
    ts = time.time_ns()
    for i, acc in enumerate(bit_accs):
        p = (Point("bit_accuracy")
             .tag("run_id", run_id)
             .tag("bit_index", str(i))
             .field("step", int(step))
             .field("accuracy", float(acc))
             .time(ts + i, WritePrecision.NS))  # +i to avoid dedup
        _write_api.write(bucket=_bucket, record=p)


def log_masks(run_id, masks, num_bits):
    """Log mask assignments (which ant owns which bit).
    masks: dict {being_id: [bit_indices]}.  Logged once at run start."""
    if not _write_api:
        return
    ts = time.time_ns()
    for being_id, bits in masks.items():
        bits_set = set(bits)
        for b in range(num_bits):
            p = (Point("mask_assignment")
                 .tag("run_id", run_id)
                 .tag("being_id", str(being_id))
                 .tag("bit_index", str(b))
                 .field("assigned", 1 if b in bits_set else 0)
                 .time(ts + being_id * num_bits + b, WritePrecision.NS))
            _write_api.write(bucket=_bucket, record=p)


def log_ant_bit_acc(run_id, step, being_id, bit_accs):
    """Log per-ant-per-bit accuracy. bit_accs: dict {bit_index: accuracy}."""
    if not _write_api:
        return
    ts = time.time_ns()
    for bit_idx, acc in bit_accs.items():
        p = (Point("ant_bit_accuracy")
             .tag("run_id", run_id)
             .tag("being_id", str(being_id))
             .tag("bit_index", str(bit_idx))
             .field("step", int(step))
             .field("accuracy", float(acc))
             .time(ts + being_id * 100 + bit_idx, WritePrecision.NS))
        _write_api.write(bucket=_bucket, record=p)


def log_gem(run_id, step, gem_values):
    """Log GEM (Global Embedding Matrix) cell values. gem_values is a list of floats."""
    if not _write_api:
        return
    ts = time.time_ns()
    for i, val in enumerate(gem_values):
        p = (Point("gem_value")
             .tag("run_id", run_id)
             .tag("cell_index", str(i))
             .field("step", int(step))
             .field("value", float(val))
             .time(ts + i, WritePrecision.NS))
        _write_api.write(bucket=_bucket, record=p)


def log_lcx(run_id, step, lcx_values, num_bits):
    """Log LCX grayscale image cells with row/col tags."""
    if not _write_api:
        return
    ts = time.time_ns()
    side = num_bits  # 8 for 8x8
    for i, val in enumerate(lcx_values):
        row = i // side
        col = i % side
        p = (Point("lcx_cell")
             .tag("run_id", run_id)
             .tag("row", str(row))
             .tag("col", str(col))
             .tag("cell", f"R{row}C{col}")
             .field("step", int(step))
             .field("value", float(val))
             .time(ts + i, WritePrecision.NS))
        _write_api.write(bucket=_bucket, record=p)


def log_frame_snapshot(run_id, step, frame_type, values, side=8):
    """Log a frame snapshot with row/col tags for Grafana table display.
    frame_type: 'input' | 'output' | 'lcx_before' | 'lcx_after'
    values: flat list of floats (side*side values)
    """
    if not _write_api:
        return
    ts = time.time_ns()
    for i, val in enumerate(values):
        row = i // side
        col = i % side
        p = (Point("frame_snapshot")
             .tag("run_id", run_id)
             .tag("frame_type", frame_type)
             .tag("row", str(row))
             .tag("col", str(col))
             .field("step", int(step))
             .field("value", float(val))
             .time(ts + i, WritePrecision.NS))
        _write_api.write(bucket=_bucket, record=p)


def log_lcx_channel_norms(run_id, step, r_norm, g_norm, b_norm, effort_level):
    """Log per-channel LCX norms for Grafana RGB time series (legacy dense LCX)."""
    if not _write_api:
        return
    p = (Point("lcx_rgb")
         .tag("run_id", run_id)
         .field("step", int(step))
         .field("r_norm", float(r_norm))
         .field("g_norm", float(g_norm))
         .field("b_norm", float(b_norm))
         .field("effort_level", int(effort_level))
         .time(time.time_ns(), WritePrecision.NS))
    _write_api.write(bucket=_bucket, record=p)


def log_lcx_level_norms(run_id, step, level_norms, num_levels, zoom_gate=None,
                        level_used=None, level_total=None, max_active_level=None,
                        slot_norms=None, lcx_route_grad=None, lcx_write_grad=None,
                        zoom_gate_grad=None, write_differentiable=None,
                        current_stage=None,
                        lcx_write_aux_loss=None, lcx_read_aux_loss=None,
                        heat_stats=None):
    """Log per-level Zoom LCX norms for Grafana."""
    if not _write_api:
        return
    p = (Point("lcx_levels")
         .tag("run_id", run_id)
         .field("step", int(step))
         .field("num_levels", int(num_levels)))
    for lvl, norm in level_norms.items():
        p = p.field(f"L{lvl}_norm", float(norm))
    if zoom_gate is not None:
        p = p.field("zoom_gate", float(zoom_gate))
    if level_used:
        for lvl, used in level_used.items():
            p = p.field(f"L{lvl}_used", int(used))
    if level_total:
        for lvl, total in level_total.items():
            p = p.field(f"L{lvl}_total", int(total))
    if max_active_level is not None:
        p = p.field("max_active_level", int(max_active_level))
    # Per-slot norms for all levels (native Grafana stat panels)
    if slot_norms:
        for lvl, norms_list in slot_norms.items():
            for si, sv in enumerate(norms_list):
                p = p.field(f"L{lvl}_s{si}", float(sv))
    # LCX gradient telemetry (asymmetric write visibility)
    if lcx_route_grad is not None:
        p = p.field("lcx_route_grad", float(lcx_route_grad))
    if lcx_write_grad is not None:
        p = p.field("lcx_write_grad", float(lcx_write_grad))
    if zoom_gate_grad is not None:
        p = p.field("zoom_gate_grad", float(zoom_gate_grad))
    if write_differentiable is not None:
        p = p.field("write_differentiable", int(write_differentiable))
    if current_stage is not None:
        p = p.field("current_stage", str(current_stage))
    if lcx_write_aux_loss is not None:
        p = p.field("lcx_write_aux_loss", float(lcx_write_aux_loss))
    if lcx_read_aux_loss is not None:
        p = p.field("lcx_read_aux_loss", float(lcx_read_aux_loss))
    # Hot-bin heat stats (sparse ring subdivision telemetry)
    if heat_stats:
        p = p.field("allocated_levels", int(heat_stats.get('allocated_levels', 0)))
        for _hl in range(num_levels):
            _hk = f'L{_hl}_hot_slots'
            if _hk in heat_stats:
                p = p.field(_hk, int(heat_stats[_hk]))
                p = p.field(f'L{_hl}_max_heat', int(heat_stats.get(f'L{_hl}_max_heat', 0)))
            _vk = f'L{_hl}_valid_slots'
            if _vk in heat_stats:
                p = p.field(_vk, int(heat_stats[_vk]))
        # Sorted rank heat bins (for Grafana bar chart)
        for _hl in range(num_levels):
            for _bi in range(100):
                _rk = f'L{_hl}_rk{_bi:02d}'
                if _rk in heat_stats:
                    p = p.field(_rk, int(heat_stats[_rk]))
        # Entropy metrics
        for _hl in range(num_levels):
            for _mk in ('entropy_pct', 'eff_slots', 'active_slots', 'total_slots', 'val_diversity', 'top1_mass', 'participation_ratio', 'top6_mass', 'active_pct', 'part_ratio_pct', 'score_margin', 'score_top1'):
                _ek = f'L{_hl}_{_mk}'
                if _ek in heat_stats:
                    p = p.field(_ek, float(heat_stats[_ek]))
    p = p.time(time.time_ns(), WritePrecision.NS)
    _write_api.write(bucket=_bucket, record=p)


def close():
    """Flush and close. Call at training end."""
    global _write_api, _client
    if _write_api:
        try:
            _write_api.close()
        except Exception:
            pass
        _write_api = None
    if _client:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
