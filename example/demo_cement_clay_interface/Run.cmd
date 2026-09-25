@echo off
rem Run this demo with the four modules two folders up.
cd /d "%~dp0"
set "CASE_FILE=%~1"
if not defined CASE_FILE set "CASE_FILE=case.yaml"
wsl /home/xi_b/miniforge3/envs/fenicsx/bin/python ../../coupling.py "%CASE_FILE%"
set "RUN_RESULT=%ERRORLEVEL%"
pause
exit /b %RUN_RESULT%
