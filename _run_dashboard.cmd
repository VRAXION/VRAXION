@echo off
S:
cd "S:\AI\work\VRAXION_DEV\Golden Draft"
set PROBE11_TELEMETRY=S:\AI\work\VRAXION_DEV\Golden Draft\probe11_telemetry.jsonl
C:\Users\kenes\AppData\Local\Programs\Python\Python311\python.exe -m streamlit run tools\probe11_dashboard.py --server.port 8511 --server.headless true --browser.gatherUsageStats false
pause
