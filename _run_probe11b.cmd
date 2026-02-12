@echo off
S:
cd "S:\AI\work\VRAXION_DEV\Golden Draft"
C:\Users\kenes\AppData\Local\Programs\Python\Python311\python.exe tools\probe11_fib_volume_weight.py --task byte_waveform --steps 2500 --solo-ant 4 --batch-size 64 --seq-len 128 --checkpoint-every 10 --device cuda --no-sync --telemetry probe11b_telemetry.jsonl --checkpoint-dir logs/probe/checkpoints_11b
echo EXIT_CODE=%ERRORLEVEL%
pause
