"""Analyze gradient spikes vs pointer telemetry."""

import re
from pathlib import Path

def main():
    log_path = Path("C:/Users/kenes/AppData/Local/Temp/claude/C--Users-kenes/tasks/b471208.output")

    print("=" * 70)
    print("GRADIENT SPIKE ANALYSIS")
    print("=" * 70)
    print()

    # Parse all steps - gnorm is on previous line
    steps = []
    last_gnorm = None

    with open(log_path) as f:
        for line in f:
            # Check for grad_norm line
            gnorm_match = re.search(r'grad_norm\(total\)=([\d.]+e[+-]?\d+|[\d.]+)', line)
            if gnorm_match:
                last_gnorm = float(gnorm_match.group(1))
                continue

            # Check for step line
            if "step" not in line or "loss" not in line:
                continue

            step_match = re.search(r'step (\d+)/\d+', line)
            if not step_match:
                continue

            metrics = {'step': int(step_match.group(1))}

            # Associate gnorm from previous line
            if last_gnorm is not None:
                metrics['gnorm'] = last_gnorm
                last_gnorm = None  # Reset

            # Main metrics
            loss_match = re.search(r'loss\s+([\d.]+)', line)
            if loss_match:
                metrics['loss'] = float(loss_match.group(1))

            # V_COG telemetry
            vcog_match = re.search(r'V_COG\[(.*?)\]', line)
            if vcog_match:
                vcog = vcog_match.group(1)
                for field in ['ORB', 'AC', 'IDENT']:
                    match = re.search(rf'{field}:(\d+(?:\.\d+)?)', vcog)
                    if match:
                        metrics[field] = float(match.group(1))
                # RD can be in scientific notation
                rd_match = re.search(r'RD:([\d.]+(?:e[+-]?\d+)?)', vcog)
                if rd_match:
                    metrics['RD'] = float(rd_match.group(1))

            # RAW telemetry
            raw_match = re.search(r'RAW\[(.*?)\]', line)
            if raw_match:
                raw = raw_match.group(1)
                # INR format: 0.48->0.48/F:0.00
                inr_match = re.search(r'INR:([\d.]+)->([\d.]+)/F:([\d.]+)', raw)
                if inr_match:
                    metrics['INR_from'] = float(inr_match.group(1))
                    metrics['INR_to'] = float(inr_match.group(2))
                    metrics['FREEZE'] = float(inr_match.group(3))

                for field in ['DZN', 'WLK', 'SCA']:
                    match = re.search(rf'{field}:([\d.]+)', raw)
                    if match:
                        metrics[field] = float(match.group(1))

            steps.append(metrics)

    print(f"Total steps parsed: {len(steps)}")
    print()

    # Find gradient spikes (top 20%)
    if not steps:
        print("No steps found!")
        return

    gnorms = [s.get('gnorm', 0) for s in steps]
    gnorms_sorted = sorted(gnorms, reverse=True)
    spike_threshold = gnorms_sorted[len(gnorms_sorted) // 5] if len(gnorms_sorted) > 5 else 20.0

    spikes = [s for s in steps if s.get('gnorm', 0) >= spike_threshold]
    normal = [s for s in steps if s.get('gnorm', 0) < spike_threshold]

    print(f"Spike threshold (top 20%): gnorm >= {spike_threshold:.2f}")
    print(f"Spike steps: {len(spikes)}")
    print(f"Normal steps: {len(normal)}")
    print()

    # Top 10 worst spikes
    print("TOP 10 WORST GRADIENT SPIKES:")
    print("-" * 70)
    print(f"{'Step':<6} {'gnorm':<8} {'loss':<8} {'ORB':<5} {'AC':<4} {'RD':<10} {'IDENT':<8}")
    print("-" * 70)

    top_spikes = sorted(spikes, key=lambda x: x.get('gnorm', 0), reverse=True)[:10]
    for s in top_spikes:
        print(f"{s.get('step', 0):<6} "
              f"{s.get('gnorm', 0):<8.2f} "
              f"{s.get('loss', 0):<8.4f} "
              f"{s.get('ORB', 0):<5.0f} "
              f"{s.get('AC', 0):<4.0f} "
              f"{s.get('RD', 0):<10.2e} "
              f"{s.get('IDENT', 0):<8.3f}")
    print()

    # Correlation analysis
    print("CORRELATION ANALYSIS:")
    print("-" * 70)

    def avg(items, field):
        vals = [x.get(field, 0) for x in items if field in x]
        return sum(vals) / len(vals) if vals else 0

    print("Average values during spikes vs normal:")
    print()
    fields = ['ORB', 'AC', 'RD', 'IDENT', 'FREEZE', 'DZN', 'WLK']
    for field in fields:
        spike_avg = avg(spikes, field)
        normal_avg = avg(normal, field)
        ratio = spike_avg / normal_avg if normal_avg > 0 else 0

        indicator = "[!]" if ratio > 1.2 or ratio < 0.8 else "[=]"
        print(f"{field:<10} Spike: {spike_avg:>8.4f}  Normal: {normal_avg:>8.4f}  "
              f"Ratio: {ratio:>6.2f}x  {indicator}")

    print()
    print("=" * 70)
    print("INTERPRETATION:")
    print()

    # Find the most correlated metric
    correlations = []
    for field in fields:
        spike_avg = avg(spikes, field)
        normal_avg = avg(normal, field)
        if normal_avg > 0:
            ratio = abs(spike_avg / normal_avg - 1.0)  # deviation from 1.0
            correlations.append((field, ratio, spike_avg, normal_avg))

    correlations.sort(key=lambda x: x[1], reverse=True)

    if correlations:
        top_field, deviation, spike_val, normal_val = correlations[0]
        print(f"Strongest correlation: {top_field}")
        print(f"  During spikes: {spike_val:.4f}")
        print(f"  During normal: {normal_val:.4f}")
        print()

        if spike_val > normal_val:
            print(f"=> Gradient spikes correlate with HIGH {top_field}")
        else:
            print(f"=> Gradient spikes correlate with LOW {top_field}")

    print()

if __name__ == "__main__":
    main()
