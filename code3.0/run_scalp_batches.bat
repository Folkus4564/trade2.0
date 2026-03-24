@echo off
REM ============================================================
REM  run_scalp_batches.bat
REM  Launches 20 parallel scalp_research instances, each with
REM  its own seed batch, output dir, and isolated model file.
REM
REM  Usage: double-click or run from any terminal in code3.0/
REM  Stop:  close the individual batch windows
REM ============================================================

cd /d "%~dp0"

echo [launcher] Starting 20 parallel scalp research batches...

start "scalp_batch_01" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_01.yaml --output-dir artefacts/scalp_research/batch_01 --base-model-id batch_01"
start "scalp_batch_02" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_02.yaml --output-dir artefacts/scalp_research/batch_02 --base-model-id batch_02"
start "scalp_batch_03" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_03.yaml --output-dir artefacts/scalp_research/batch_03 --base-model-id batch_03"
start "scalp_batch_04" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_04.yaml --output-dir artefacts/scalp_research/batch_04 --base-model-id batch_04"
start "scalp_batch_05" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_05.yaml --output-dir artefacts/scalp_research/batch_05 --base-model-id batch_05"
start "scalp_batch_06" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_06.yaml --output-dir artefacts/scalp_research/batch_06 --base-model-id batch_06"
start "scalp_batch_07" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_07.yaml --output-dir artefacts/scalp_research/batch_07 --base-model-id batch_07"
start "scalp_batch_08" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_08.yaml --output-dir artefacts/scalp_research/batch_08 --base-model-id batch_08"
start "scalp_batch_09" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_09.yaml --output-dir artefacts/scalp_research/batch_09 --base-model-id batch_09"
start "scalp_batch_10" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_10.yaml --output-dir artefacts/scalp_research/batch_10 --base-model-id batch_10"
start "scalp_batch_11" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_11.yaml --output-dir artefacts/scalp_research/batch_11 --base-model-id batch_11"
start "scalp_batch_12" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_12.yaml --output-dir artefacts/scalp_research/batch_12 --base-model-id batch_12"
start "scalp_batch_13" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_13.yaml --output-dir artefacts/scalp_research/batch_13 --base-model-id batch_13"
start "scalp_batch_14" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_14.yaml --output-dir artefacts/scalp_research/batch_14 --base-model-id batch_14"
start "scalp_batch_15" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_15.yaml --output-dir artefacts/scalp_research/batch_15 --base-model-id batch_15"
start "scalp_batch_16" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_16.yaml --output-dir artefacts/scalp_research/batch_16 --base-model-id batch_16"
start "scalp_batch_17" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_17.yaml --output-dir artefacts/scalp_research/batch_17 --base-model-id batch_17"
start "scalp_batch_18" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_18.yaml --output-dir artefacts/scalp_research/batch_18 --base-model-id batch_18"
start "scalp_batch_19" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_19.yaml --output-dir artefacts/scalp_research/batch_19 --base-model-id batch_19"
start "scalp_batch_20" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_20.yaml --output-dir artefacts/scalp_research/batch_20 --base-model-id batch_20"

echo [launcher] All 20 batches launched. Each runs in its own window.
echo [launcher] When all finish, run: python merge_scalp_batches.py
