@echo off
S:
cd "S:\AI\work\VRAXION_DEV\Golden Draft"
C:\Users\kenes\AppData\Local\Programs\Python\Python311\python.exe tools\probe11_fib_volume_weight.py --steps 2500 --active-ants 2 --checkpoint-every 10 --device cuda --no-sync --no-dashboard --resume logs\probe\checkpoints\probe11_step_01480.pt > "S:\AI\work\VRAXION_DEV\probe11_debug.log" 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> "S:\AI\work\VRAXION_DEV\probe11_debug.log"
