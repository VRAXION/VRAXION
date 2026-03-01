@echo off
title VRAXION Dashboard
cd /d "%~dp0"
python training\dashboard.py %*
if errorlevel 1 pause
