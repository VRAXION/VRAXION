"""
Diamond Code Log Parser - Regex-based utilities for parsing training logs.

This module provides stdlib-only parsing (no Streamlit dependency) that can be
imported by the dashboard and other analysis tools.

Log Format:
    step N | loss X.XXXXXX | acc=X.XXXX | jump_gate=X.XX | ...
"""

import re
import os
from typing import List, Dict, Tuple, Optional


# ============================================================================
# Regex Patterns
# ============================================================================

# Core step/loss pattern (follows VRAXION format for compatibility)
RESTEP = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+\|(?P<tail>.*)"
)

# Diamond-specific metrics
RE_ACC = re.compile(r"acc=(?P<acc>[-+]?\d+(?:\.\d+)?)")
RE_JUMP_GATE = re.compile(r"jump_gate=(?P<jump_gate>[-+]?\d+(?:\.\d+)?)")
RE_CYCLES = re.compile(r"cyc=(?P<cycles>\d+)")
RE_SELF_LOOPS = re.compile(r"sl=(?P<self_loops>\d+)")
RE_ATT_ENT = re.compile(r"att_ent=(?P<att_ent>[-+]?\d+(?:\.\d+)?)")
RE_PTR_POS = re.compile(r"ptr_pos=(?P<ptr_pos>[-+]?\d+(?:\.\d+)?)")
RE_RING_SPARSITY = re.compile(r"ring_sparsity=(?P<ring_sparsity>[-+]?\d+(?:\.\d+)?)")
RE_TIMING = re.compile(r"s_per_step=(?P<s_per_step>[-+]?\d+(?:\.\d+)?)")
RE_GRAD_JUMP = re.compile(r"grad_j=(?P<grad_jump>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
RE_GRAD_OTHER = re.compile(r"grad_o=(?P<grad_other>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_log_line(line: str) -> Optional[Dict[str, float]]:
    """
    Parse a single log line into a dict of metrics.

    Args:
        line: Log line string (e.g., "step 100 | loss 1.234 | acc=0.85 | ...")

    Returns:
        Dict with parsed metrics, or None if line doesn't match format
    """
    # Try to match core step/loss pattern
    match = RESTEP.match(line.strip())
    if not match:
        return None

    # Extract step and loss
    row = {
        'step': int(match.group('step')),
        'loss': float(match.group('loss'))
    }

    # Extract optional tail metrics
    tail = match.group('tail')
    if tail:
        # Accuracy
        m = RE_ACC.search(tail)
        if m:
            row['acc'] = float(m.group('acc'))

        # Jump gate activation
        m = RE_JUMP_GATE.search(tail)
        if m:
            row['jump_gate'] = float(m.group('jump_gate'))

        # Cycles
        m = RE_CYCLES.search(tail)
        if m:
            row['cycles'] = int(m.group('cycles'))

        # Self-loops
        m = RE_SELF_LOOPS.search(tail)
        if m:
            row['self_loops'] = int(m.group('self_loops'))

        # Attention entropy
        m = RE_ATT_ENT.search(tail)
        if m:
            row['att_ent'] = float(m.group('att_ent'))

        # Pointer position
        m = RE_PTR_POS.search(tail)
        if m:
            row['ptr_pos'] = float(m.group('ptr_pos'))

        # Ring sparsity
        m = RE_RING_SPARSITY.search(tail)
        if m:
            row['ring_sparsity'] = float(m.group('ring_sparsity'))

        # Timing
        m = RE_TIMING.search(tail)
        if m:
            row['s_per_step'] = float(m.group('s_per_step'))

        # Gradient norms
        m = RE_GRAD_JUMP.search(tail)
        if m:
            row['grad_jump'] = float(m.group('grad_jump'))

        m = RE_GRAD_OTHER.search(tail)
        if m:
            row['grad_other'] = float(m.group('grad_other'))

    return row


def parse_log_lines(lines: List[str]) -> List[Dict[str, float]]:
    """
    Parse multiple log lines into a list of metric dicts.

    Args:
        lines: List of log line strings

    Returns:
        List of parsed metric dicts (skips unparseable lines)
    """
    rows = []
    for line in lines:
        row = parse_log_line(line)
        if row is not None:
            rows.append(row)
    return rows


def read_new_lines(path: str, last_pos: int) -> Tuple[List[str], int]:
    """
    Read only newly appended lines from a log file (incremental read).

    This is memory-efficient for large log files - tracks file position
    and only reads new data on each call.

    Args:
        path: Path to log file
        last_pos: Last byte position read (0 for first read)

    Returns:
        (new_lines, new_position) tuple
    """
    if not os.path.exists(path):
        return [], last_pos

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(last_pos)
            new_lines = f.readlines()
            new_pos = f.tell()
        return new_lines, new_pos
    except Exception as e:
        print(f"Warning: Error reading log file: {e}")
        return [], last_pos


def read_and_parse_log(path: str, last_pos: int = 0) -> Tuple[List[Dict], int]:
    """
    Read new lines from log and parse them into metric dicts.

    Convenience function combining read_new_lines() and parse_log_lines().

    Args:
        path: Path to log file
        last_pos: Last byte position read (0 for first read)

    Returns:
        (parsed_rows, new_position) tuple
    """
    new_lines, new_pos = read_new_lines(path, last_pos)
    rows = parse_log_lines(new_lines)
    return rows, new_pos


# ============================================================================
# Testing / Main
# ============================================================================

def _test_parser():
    """Test the parser with sample log lines."""
    print("Testing Diamond Code Log Parser")
    print("=" * 70)

    # Sample log lines
    test_lines = [
        "step 0 | loss 2.5846 | acc=0.04 | jump_gate=0.50 | s_per_step=1.234",
        "step 100 | loss 1.2345 | acc=0.85 | jump_gate=0.65 | att_ent=2.34 | ptr_pos=32.5",
        "step 200 | loss 0.8765 | acc=0.92 | jump_gate=0.72 | cyc=3 | sl=2 | grad_j=0.042",
        "invalid line that should be skipped",
        "step 300 | loss 0.5432 | acc=0.96 | jump_gate=0.78 | ring_sparsity=0.45",
    ]

    print(f"Parsing {len(test_lines)} test lines...")
    print()

    rows = parse_log_lines(test_lines)

    print(f"Successfully parsed {len(rows)} rows:")
    print()

    for row in rows:
        print(f"Step {row['step']:4d}: loss={row['loss']:.4f}, metrics={len(row)-2}")
        for key, val in row.items():
            if key not in ['step', 'loss']:
                print(f"  {key:15s} = {val}")
        print()

    print("=" * 70)
    print(f"Test complete: {len(rows)}/{len(test_lines)} lines parsed successfully")


if __name__ == "__main__":
    _test_parser()
