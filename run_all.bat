@echo off
REM ==========================================================
REM  Real-Time Systems - Unit 5 Lab (Mini Projects 5.1-5.3)
REM  Runs every simulator and saves the console output to outputs\
REM  Requirements: Python 3.8+ (standard library only;
REM  matplotlib is optional and only used for the PNG plots)
REM ==========================================================
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=python
where python >nul 2>nul || set PY=py
if not exist outputs mkdir outputs

echo.
echo [1/3] Mini Project 5.1 - TMR voting, reliability metrics, watchdog
%PY% mp5_1_tmr_watchdog\reliability_sim.py > outputs\mp5_1_output.txt
type outputs\mp5_1_output.txt

echo.
echo [2/3] Mini Project 5.2 - Monte Carlo simplex vs TMR vs 5MR/7MR (about 1 minute)
%PY% mp5_2_monte_carlo_nmr\monte_carlo_nmr.py > outputs\mp5_2_output.txt
type outputs\mp5_2_output.txt

echo.
echo [3/3] Mini Project 5.3 - checkpoint and rollback-recovery
%PY% mp5_3_checkpoint_rollback\checkpoint_sim.py > outputs\mp5_3_output.txt
type outputs\mp5_3_output.txt

echo.
echo All done. Console outputs, CSV files and plots are in the outputs folder.
pause
