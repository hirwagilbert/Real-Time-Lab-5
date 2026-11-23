#!/usr/bin/env sh
# Linux/macOS equivalent of run_all.bat
cd "$(dirname "$0")" && mkdir -p outputs
python3 mp5_1_tmr_watchdog/reliability_sim.py          | tee outputs/mp5_1_output.txt
python3 mp5_2_monte_carlo_nmr/monte_carlo_nmr.py       | tee outputs/mp5_2_output.txt
python3 mp5_3_checkpoint_rollback/checkpoint_sim.py    | tee outputs/mp5_3_output.txt
