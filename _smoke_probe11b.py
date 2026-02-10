"""Smoke test: run probe11 byte_waveform for 50 steps on CPU."""
import sys, os
os.chdir(r"S:\AI\work\VRAXION_DEV\Golden Draft")
sys.argv = [
    "probe11_fib_volume_weight.py",
    "--task", "byte_waveform",
    "--steps", "50",
    "--solo-ant", "4",
    "--batch-size", "16",
    "--seq-len", "64",
    "--device", "cpu",
    "--no-sync",
    "--no-dashboard",
    "--log-every", "10",
    "--checkpoint-every", "0",
    "--telemetry", "probe11b_smoke_telemetry.jsonl",
]
sys.path.insert(0, "tools")
from probe11_fib_volume_weight import main
main()
