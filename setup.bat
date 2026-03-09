@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo   trade2.0 - XAUUSD Quant Research System Setup
echo ============================================================
echo.

:: ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER% found

:: ── Create virtual environment ────────────────────────────────
if exist venv (
    echo [SKIP] Virtual environment already exists
) else (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

:: ── Activate venv ─────────────────────────────────────────────
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

:: ── Upgrade pip ───────────────────────────────────────────────
echo [SETUP] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: ── Install TA-Lib (Windows pre-compiled wheel) ───────────────
echo [SETUP] Installing TA-Lib (Windows binary)...
python -c "import talib" >nul 2>&1
if errorlevel 1 (
    echo   Trying pre-compiled wheel from cgohlke/talib-build...
    pip install TA-Lib --find-links https://github.com/cgohlke/talib-build/releases/latest/download/ --quiet 2>nul
    if errorlevel 1 (
        echo   Trying PyPI TA-Lib-precompiled...
        pip install ta-lib-precompiled --quiet 2>nul
        if errorlevel 1 (
            echo   Trying TA_Lib wheel from unofficial Windows binaries...
            pip install --upgrade TA-Lib --quiet 2>nul
            if errorlevel 1 (
                echo.
                echo [WARN] Automatic TA-Lib install failed.
                echo   Manual fix options:
                echo   1. Download wheel from: https://github.com/cgohlke/talib-build/releases
                echo      Then run: pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
                echo   2. Or install via conda: conda install -c conda-forge ta-lib
                echo.
                echo   Continuing with other packages...
            )
        )
    )
) else (
    echo [SKIP] TA-Lib already installed
)

:: ── Install remaining requirements ───────────────────────────
echo [SETUP] Installing remaining packages from requirements.txt...
pip install numpy pandas scikit-learn hmmlearn optuna joblib pyarrow matplotlib PyYAML --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install core packages
    pause
    exit /b 1
)
echo [OK] Core packages installed

echo [SETUP] Installing vectorbt...
pip install vectorbt --quiet
if errorlevel 1 (
    echo [WARN] vectorbt install failed - try: pip install vectorbt manually
) else (
    echo [OK] vectorbt installed
)

:: ── Create required directories ───────────────────────────────
if not exist "data\processed" mkdir "data\processed"
if not exist "models"         mkdir "models"
if not exist "backtests"      mkdir "backtests"
if not exist "reports"        mkdir "reports"
echo [OK] Directories verified

:: ── Verify data file ─────────────────────────────────────────
if exist "data\XAUUSD_1H_2019_2024.csv" (
    echo [OK] Raw data file found: data\XAUUSD_1H_2019_2024.csv
) else (
    echo [WARN] Raw data file missing: data\XAUUSD_1H_2019_2024.csv
    echo   Copy it from Claude-execute\data\ or re-download with download_xauusd.py
)

:: ── Run data preparation ─────────────────────────────────────
echo.
echo [SETUP] Preparing processed datasets...
python src\data\prepare_data.py
if errorlevel 1 (
    echo [WARN] Data preparation had errors - check TA-Lib install
) else (
    echo [OK] Processed datasets ready in data\processed\
)

:: ── Done ─────────────────────────────────────────────────────
echo.
echo ============================================================
echo   SETUP COMPLETE
echo ============================================================
echo.
echo   Next steps:
echo.
echo   1. Run the pipeline (default params):
echo      python src\pipeline.py
echo.
echo   2. Run with Optuna optimization:
echo      python src\pipeline.py --optimize --trials 100
echo.
echo   3. Run with walk-forward validation:
echo      python src\pipeline.py --walk-forward
echo.
echo   4. Use Claude agent skill (in Claude Code chat):
echo      /run-trading-pipeline Build a systematic XAUUSD strategy
echo.
echo   Results will appear in:
echo      reports\final_verdict.json
echo      backtests\*_test_results.json
echo ============================================================
echo.
pause
