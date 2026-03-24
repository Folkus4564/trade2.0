@echo off
REM ============================================================
REM  run_scalp_batches_deepseek.bat
REM  Same as run_scalp_batches.bat but uses DeepSeek for
REM  indicator translation. Modules prefixed with ds_ to
REM  avoid colliding with the Claude seed run.
REM
REM  Usage: double-click or run from any terminal in code3.0/
REM  Stop:  close the individual batch windows
REM ============================================================

cd /d "%~dp0"

echo [launcher] Starting 20 parallel scalp research batches (DeepSeek)...

start "scalp_ds_01" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_01.yaml --output-dir artefacts/scalp_research/ds_batch_01 --base-model-id ds_batch_01 --provider deepseek --module-prefix ds_"
start "scalp_ds_02" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_02.yaml --output-dir artefacts/scalp_research/ds_batch_02 --base-model-id ds_batch_02 --provider deepseek --module-prefix ds_"
start "scalp_ds_03" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_03.yaml --output-dir artefacts/scalp_research/ds_batch_03 --base-model-id ds_batch_03 --provider deepseek --module-prefix ds_"
start "scalp_ds_04" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_04.yaml --output-dir artefacts/scalp_research/ds_batch_04 --base-model-id ds_batch_04 --provider deepseek --module-prefix ds_"
start "scalp_ds_05" cmd /k "scalp_research --source seed --max-ideas 11 --seed-path configs/scalp_seeds/batch_05.yaml --output-dir artefacts/scalp_research/ds_batch_05 --base-model-id ds_batch_05 --provider deepseek --module-prefix ds_"
start "scalp_ds_06" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_06.yaml --output-dir artefacts/scalp_research/ds_batch_06 --base-model-id ds_batch_06 --provider deepseek --module-prefix ds_"
start "scalp_ds_07" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_07.yaml --output-dir artefacts/scalp_research/ds_batch_07 --base-model-id ds_batch_07 --provider deepseek --module-prefix ds_"
start "scalp_ds_08" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_08.yaml --output-dir artefacts/scalp_research/ds_batch_08 --base-model-id ds_batch_08 --provider deepseek --module-prefix ds_"
start "scalp_ds_09" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_09.yaml --output-dir artefacts/scalp_research/ds_batch_09 --base-model-id ds_batch_09 --provider deepseek --module-prefix ds_"
start "scalp_ds_10" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_10.yaml --output-dir artefacts/scalp_research/ds_batch_10 --base-model-id ds_batch_10 --provider deepseek --module-prefix ds_"
start "scalp_ds_11" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_11.yaml --output-dir artefacts/scalp_research/ds_batch_11 --base-model-id ds_batch_11 --provider deepseek --module-prefix ds_"
start "scalp_ds_12" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_12.yaml --output-dir artefacts/scalp_research/ds_batch_12 --base-model-id ds_batch_12 --provider deepseek --module-prefix ds_"
start "scalp_ds_13" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_13.yaml --output-dir artefacts/scalp_research/ds_batch_13 --base-model-id ds_batch_13 --provider deepseek --module-prefix ds_"
start "scalp_ds_14" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_14.yaml --output-dir artefacts/scalp_research/ds_batch_14 --base-model-id ds_batch_14 --provider deepseek --module-prefix ds_"
start "scalp_ds_15" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_15.yaml --output-dir artefacts/scalp_research/ds_batch_15 --base-model-id ds_batch_15 --provider deepseek --module-prefix ds_"
start "scalp_ds_16" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_16.yaml --output-dir artefacts/scalp_research/ds_batch_16 --base-model-id ds_batch_16 --provider deepseek --module-prefix ds_"
start "scalp_ds_17" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_17.yaml --output-dir artefacts/scalp_research/ds_batch_17 --base-model-id ds_batch_17 --provider deepseek --module-prefix ds_"
start "scalp_ds_18" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_18.yaml --output-dir artefacts/scalp_research/ds_batch_18 --base-model-id ds_batch_18 --provider deepseek --module-prefix ds_"
start "scalp_ds_19" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_19.yaml --output-dir artefacts/scalp_research/ds_batch_19 --base-model-id ds_batch_19 --provider deepseek --module-prefix ds_"
start "scalp_ds_20" cmd /k "scalp_research --source seed --max-ideas 10 --seed-path configs/scalp_seeds/batch_20.yaml --output-dir artefacts/scalp_research/ds_batch_20 --base-model-id ds_batch_20 --provider deepseek --module-prefix ds_"

echo [launcher] All 20 DeepSeek batches launched. Each runs in its own window.
echo [launcher] When all finish, run: python merge_scalp_batches.py --prefix ds_batch
