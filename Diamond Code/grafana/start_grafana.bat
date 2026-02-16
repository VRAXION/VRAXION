@echo off
REM Diamond Code - Grafana Launcher
REM Grafana runs as a Windows service (auto-start)
REM Install: C:\Program Files\GrafanaLabs\grafana
REM Provisioning: S:\AI\work\VRAXION_DEV\Diamond Code\grafana\provisioning
REM Dashboards:   S:\AI\work\VRAXION_DEV\Diamond Code\grafana\dashboards

REM Restart the service (requires admin - will trigger UAC)
powershell -Command "Start-Process powershell -ArgumentList '-Command','Restart-Service Grafana' -Verb RunAs -Wait"

echo Grafana restarted. Dashboard: http://localhost:3000
pause
