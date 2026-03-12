"""Render a clear profiler breakdown chart from proxy-step artifacts.

Input:
  - JSON from profile_sweep_step_wikitext.py
  - ops table TXT from the same run

Output:
  - single PNG with:
    1. total step breakdown
    2. forward breakdown
    3. source scope breakdown (when present)
    4. top CUDA ops by self CUDA %
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def _default_output_path(json_path: Path) -> Path:
    return json_path.with_name(json_path.stem + '_breakdown.png')


def load_json(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def parse_ops_table(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip():
                continue
            if line.startswith('-') or line.startswith('Self CPU time total'):
                continue
            if line.lstrip().startswith('Name'):
                continue
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) < 8:
                continue
            name = parts[0]
            try:
                self_cuda_pct = float(parts[7].replace('%', ''))
            except ValueError:
                continue
            rows.append(
                {
                    'name': name,
                    'self_cuda_pct': self_cuda_pct,
                }
            )
    return rows


def build_stage_rows(payload: dict) -> list[dict]:
    return sorted(
        payload['coarse_stage_breakdown'],
        key=lambda row: row['seconds'],
        reverse=True,
    )


def build_forward_rows(payload: dict) -> list[dict]:
    forward_ms = payload['coarse_stage_breakdown'][1]['seconds'] * 1000.0
    fn_rows = [row for row in payload['function_breakdown'] if row['total_ms'] > 0]
    used_ms = sum(row['total_ms'] for row in fn_rows)
    other_ms = max(0.0, forward_ms - used_ms)
    rows = [
        {
            'name': row['name'],
            'ms': row['total_ms'],
            'pct_forward': 100.0 * row['total_ms'] / max(forward_ms, 1e-9),
        }
        for row in fn_rows
    ]
    rows.append(
        {
            'name': 'other_forward_work',
            'ms': other_ms,
            'pct_forward': 100.0 * other_ms / max(forward_ms, 1e-9),
        }
    )
    return sorted(rows, key=lambda row: row['ms'], reverse=True)


def build_top_cuda_rows(ops_rows: list[dict], top_n: int) -> list[dict]:
    filtered = [
        row for row in ops_rows
        if row['name'].startswith('aten::')
        and row['name'] not in {'forward_loss', 'backward'}
    ]
    filtered.sort(key=lambda row: row['self_cuda_pct'], reverse=True)
    return filtered[:top_n]


def build_scope_rows(payload: dict) -> list[dict]:
    rows = payload.get('source_scope_breakdown', [])
    return sorted(rows, key=lambda row: row['self_device_us'], reverse=True)


def short_label(name: str) -> str:
    return (
        name
        .replace('func_', '')
        .replace('_tns', '')
        .replace('_', ' ')
        .replace('aten::', '')
    )


def plot_chart(payload: dict, ops_rows: list[dict], output_path: Path, top_n: int):
    stage_rows = build_stage_rows(payload)
    forward_rows = build_forward_rows(payload)
    scope_rows = build_scope_rows(payload)
    top_cuda_rows = build_top_cuda_rows(ops_rows, top_n)

    ncols = 4 if scope_rows else 3
    fig, axes = plt.subplots(1, ncols, figsize=(22 if scope_rows else 18, 7))
    fig.suptitle(
        f'Proxy-Step Perf Breakdown | batch={payload["config"]["batch"]} seq={payload["config"]["seq"]} '
        f'C={payload["config"]["c_value"]:.4f} impl={payload["config"]["impl"]} write={payload["config"]["write_impl"]}',
        fontsize=12,
    )

    stage_labels = [short_label(row['stage']) for row in stage_rows]
    stage_vals = [row['pct_total'] for row in stage_rows]
    axes[0].barh(stage_labels, stage_vals, color='#4c78a8')
    axes[0].invert_yaxis()
    axes[0].set_title('Total Step Breakdown (%)')
    axes[0].set_xlabel('% of total step')
    for i, row in enumerate(stage_rows):
        axes[0].text(row['pct_total'] + 0.5, i, f'{row["seconds"]:.3f}s', va='center', fontsize=9)

    forward_labels = [short_label(row['name']) for row in forward_rows]
    forward_vals = [row['pct_forward'] for row in forward_rows]
    axes[1].barh(forward_labels, forward_vals, color='#f58518')
    axes[1].invert_yaxis()
    axes[1].set_title('Forward Breakdown (%)')
    axes[1].set_xlabel('% of forward')
    for i, row in enumerate(forward_rows):
        axes[1].text(row['pct_forward'] + 0.5, i, f'{row["ms"]:.0f}ms', va='center', fontsize=9)

    cuda_axis = axes[-1]
    if scope_rows:
        scope_labels = [short_label(row['scope']) for row in scope_rows]
        scope_vals = [row['pct_scoped_device'] for row in scope_rows]
        axes[2].barh(scope_labels, scope_vals, color='#72b7b2')
        axes[2].invert_yaxis()
        axes[2].set_title('Source Scope Breakdown (%)')
        axes[2].set_xlabel('% of scoped device time')
        for i, row in enumerate(scope_rows):
            axes[2].text(
                row['pct_scoped_device'] + 0.4,
                i,
                f'{row["self_device_us"] / 1000.0:.0f}ms',
                va='center',
                fontsize=9,
            )

    cuda_labels = [short_label(row['name']) for row in top_cuda_rows]
    cuda_vals = [row['self_cuda_pct'] for row in top_cuda_rows]
    cuda_axis.barh(cuda_labels, cuda_vals, color='#54a24b')
    cuda_axis.invert_yaxis()
    cuda_axis.set_title('Top CUDA Ops (%)')
    cuda_axis.set_xlabel('% of self CUDA time')
    for i, row in enumerate(top_cuda_rows):
        cuda_axis.text(row['self_cuda_pct'] + 0.3, i, f'{row["self_cuda_pct"]:.2f}%', va='center', fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='profile_sweep_step_wikitext JSON artifact')
    parser.add_argument('--ops', required=True, help='matching ops table TXT artifact')
    parser.add_argument('--out', default='', help='output PNG path')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    json_path = Path(args.json)
    ops_path = Path(args.ops)
    output_path = Path(args.out) if args.out else _default_output_path(json_path)

    payload = load_json(json_path)
    ops_rows = parse_ops_table(ops_path)
    plot_chart(payload, ops_rows, output_path, args.top_n)

    print(f'Saved chart: {output_path}')


if __name__ == '__main__':
    main()
