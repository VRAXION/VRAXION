"""Analyze correlation between pointer freeze and gradient spikes"""

import re
from pathlib import Path
import statistics

def parse_log_line(line):
    """Extract metrics from log line"""
    match = re.search(r'step (\d+)', line)
    if not match:
        return None

    step = int(match.group(1))

    # Extract all numeric metrics
    metrics = {'step': step}
    for field in ['loss', 'acc', 'acc_ma', 'gnorm', 's_per_step',
                  'flip_rate', 'orbit', 'residual', 'anchor_clicks',
                  'inertia', 'deadzone', 'walk']:
        match = re.search(rf'{field}=([\d.]+)', line)
        if match:
            metrics[field] = float(match.group(1))

    return metrics

def main():
    log_path = Path("S:/AI/work/VRAXION_DEV/Golden Draft/logs/probe/probe_live.log")

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    print("=" * 70)
    print("POINTER-GRADIENT CORRELATION ANALYSIS")
    print("=" * 70)
    print()

    # Parse all steps
    steps = []
    with open(log_path) as f:
        for line in f:
            metrics = parse_log_line(line)
            if metrics:
                steps.append(metrics)

    print(f"Total steps analyzed: {len(steps)}")
    print()

    # Identify gradient spikes (gnorm > 20)
    spike_threshold = 20.0
    spikes = [s for s in steps if s.get('gnorm', 0) > spike_threshold]

    print(f"Gradient spikes (gnorm > {spike_threshold}): {len(spikes)}")
    print()

    if spikes:
        print("Top 10 gradient spikes:")
        print("-" * 70)
        print(f"{'Step':<6} {'gnorm':<8} {'flip_rate':<11} {'acc':<8} {'loss':<8}")
        print("-" * 70)

        for spike in sorted(spikes, key=lambda x: x.get('gnorm', 0), reverse=True)[:10]:
            print(f"{spike['step']:<6} {spike.get('gnorm', 0):<8.2f} "
                  f"{spike.get('flip_rate', 0):<11.4f} "
                  f"{spike.get('acc', 0):<8.2%} "
                  f"{spike.get('loss', 0):<8.4f}")
        print()

    # Analyze correlation
    freeze_threshold = 0.15
    frozen_steps = [s for s in steps if s.get('flip_rate', 1.0) < freeze_threshold]
    active_steps = [s for s in steps if s.get('flip_rate', 1.0) >= freeze_threshold]

    print(f"Pointer states:")
    print(f"  Frozen (flip_rate < {freeze_threshold}): {len(frozen_steps)} steps")
    print(f"  Active (flip_rate >= {freeze_threshold}): {len(active_steps)} steps")
    print()

    if frozen_steps and active_steps:
        frozen_gnorms = [s.get('gnorm', 0) for s in frozen_steps]
        active_gnorms = [s.get('gnorm', 0) for s in active_steps]

        print("Gradient statistics:")
        print(f"  Frozen pointer - mean gnorm: {statistics.mean(frozen_gnorms):.2f}")
        print(f"  Active pointer - mean gnorm: {statistics.mean(active_gnorms):.2f}")
        print()
        print(f"  Frozen pointer - max gnorm:  {max(frozen_gnorms):.2f}")
        print(f"  Active pointer - max gnorm:  {max(active_gnorms):.2f}")
        print()

        # Correlation verdict
        ratio = statistics.mean(frozen_gnorms) / statistics.mean(active_gnorms)
        print(f"Gradient amplification during freeze: {ratio:.2f}x")
        print()

        if ratio > 1.5:
            print("CONFIRMED: Pointer freeze strongly correlates with gradient spikes")
            print("   Recommendation: Increase inertia/deadzone to prevent pointer lock")
        elif ratio > 1.2:
            print("MODERATE: Pointer freeze shows correlation with higher gradients")
            print("   Recommendation: Monitor longer, consider control tuning")
        else:
            print("WEAK: No strong correlation detected")

    print("=" * 70)

if __name__ == "__main__":
    main()
