@echo off
S:
cd "S:\AI\work\VRAXION_DEV\Golden Draft"
C:\Users\kenes\AppData\Local\Programs\Python\Python311\python.exe tools\probe11_fib_volume_weight.py --steps 2500 --solo-ant 4 --checkpoint-every 10 --device cuda --no-sync
echo EXIT_CODE=%ERRORLEVEL%
pause
