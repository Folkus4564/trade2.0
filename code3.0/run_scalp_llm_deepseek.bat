@echo off
REM ============================================================
REM  run_scalp_llm_deepseek.bat
REM  LLM-driven indicator discovery using DeepSeek.
REM  No seed list -- DeepSeek suggests all 200 indicators.
REM  Modules prefixed with llm_ to avoid collisions.
REM  Single window, runs until max-ideas reached.
REM
REM  Usage: double-click or run from any terminal in code3.0/
REM  Stop:  close the window
REM ============================================================

cd /d "%~dp0"

echo [launcher] Starting LLM discovery run (DeepSeek, 200 ideas)...

start "scalp_llm_deepseek" cmd /k "scalp_research --source llm --max-ideas 200 --provider deepseek --output-dir artefacts/scalp_research/llm_deepseek --base-model-id llm_deepseek --module-prefix llm_"

echo [launcher] LLM DeepSeek run launched.
