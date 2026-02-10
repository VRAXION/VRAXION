"""Wrapper script to run probe with proper output/error capture"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    # Paths
    probe_script = Path("S:/AI/work/VRAXION_DEV/Golden Draft/tools/_scratch/long_run_probe_agc_off.py")
    log_dir = Path("S:/AI/work/VRAXION_DEV/Golden Draft/logs/probe")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_log = log_dir / f"probe_stdout_{timestamp}.log"
    stderr_log = log_dir / f"probe_stderr_{timestamp}.log"
    combined_log = log_dir / f"probe_console_{timestamp}.log"

    print("=" * 70)
    print("PROBE LAUNCHER - 2000 Step Run with Full Error Capture")
    print("=" * 70)
    print(f"Probe script:   {probe_script}")
    print(f"STDOUT log:     {stdout_log}")
    print(f"STDERR log:     {stderr_log}")
    print(f"Combined log:   {combined_log}")
    print(f"Dashboard log:  logs/probe/probe_live.log")
    print()
    print("Starting probe... (Ctrl+C to stop)")
    print("=" * 70)
    print()

    # Run probe with output capture
    try:
        with open(stdout_log, 'w') as out_f, \
             open(stderr_log, 'w') as err_f, \
             open(combined_log, 'w') as combined_f:

            process = subprocess.Popen(
                [sys.executable, str(probe_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=probe_script.parent.parent.parent
            )

            # Stream output to both console and files
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if stdout_line:
                    print(stdout_line, end='')
                    out_f.write(stdout_line)
                    combined_f.write(stdout_line)
                    out_f.flush()
                    combined_f.flush()

                if stderr_line:
                    print(f"[STDERR] {stderr_line}", end='', file=sys.stderr)
                    err_f.write(stderr_line)
                    combined_f.write(f"[STDERR] {stderr_line}")
                    err_f.flush()
                    combined_f.flush()

                # Check if process finished
                if process.poll() is not None and not stdout_line and not stderr_line:
                    break

            # Get return code
            returncode = process.wait()

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Probe interrupted by user (Ctrl+C)")
        print("=" * 70)
        process.terminate()
        returncode = -1

    # Summary
    print()
    print("=" * 70)
    print("PROBE RUN COMPLETE")
    print("=" * 70)
    print(f"Return code: {returncode}")
    print(f"Logs saved to: {log_dir}")
    print()

    if returncode != 0:
        print("WARNING: Non-zero exit code detected")
        print(f"   Check error log: {stderr_log}")
        print()

    return returncode

if __name__ == "__main__":
    sys.exit(main())
