@echo off
S:
cd "S:\AI\work\VRAXION_DEV\Golden Draft"
C:\Users\kenes\AppData\Local\Programs\Python\Python311\python.exe tools\probe11_fib_volume_weight.py --steps 2500 --active-ants 2 --solo-ant 1 --checkpoint-every 10 --device cuda --no-sync --no-dashboard
