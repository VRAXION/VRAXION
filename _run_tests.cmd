@echo off
S:
cd "S:\AI\work\VRAXION_DEV\Golden Draft"
C:\Users\kenes\AppData\Local\Programs\Python\Python311\python.exe -m unittest discover -s tests -v 2>&1 | findstr /C:"Ran " /C:"FAILED" /C:"OK" /C:"FAIL:"
echo EXIT_CODE=%ERRORLEVEL%
