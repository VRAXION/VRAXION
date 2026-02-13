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
_available = False

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import ASYNCHRONOUS
    _available = True
except ImportError:
    _available = False


def init(url="http://localhost:8086", token=None, org="vraxion", bucket="diamond"):
    """Initialize InfluxDB client. No-op if token not set or library missing."""
    global _client, _write_api, _bucket
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
    try:
        _client = InfluxDBClient(url=url, token=token, org=org)
        _write_api = _client.write_api(write_options=ASYNCHRONOUS)
    except Exception:
        pass  # silently fail — don't break training


def log_step(run_id, step, loss, bit_acc=0, byte_match=0, oracle=0,
             bit_oracle=0, ensemble_benefit=0, coverage=0, clustering=0,
             circular_spread=0, s_per_step=0, is_eval=False, n_bits=0, **kwargs):
    """Log one training step. Non-blocking."""
    if not _write_api:
        return
    p = (Point("training_step")
         .tag("run_id", run_id)
         .tag("is_eval", "true" if is_eval else "false")
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
         .time(time.time_ns(), WritePrecision.NS))
    _write_api.write(bucket=_bucket, record=p)


def log_being(run_id, step, being_id, accuracy=0, masked_acc=0,
              jump_rate=0, k_bits=0, unique_bits=0, redundant_bits=0):
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
