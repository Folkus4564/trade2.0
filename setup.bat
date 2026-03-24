@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  trade2.0 - XAUUSD Quant Research System - Setup Script
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found

:: Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found. Run: python -m ensurepip
    pause
    exit /b 1
)
echo [OK] pip found

echo.
echo [1/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo [2/5] Installing TA-Lib (pre-built wheel for Windows)...
pip show TA-Lib >nul 2>&1
if errorlevel 1 (
    echo Attempting TA-Lib install via pip...
    pip install TA-Lib
    if errorlevel 1 (
        echo.
        echo [WARN] TA-Lib pip install failed. Trying unofficial Windows wheel...
        python -c "import struct; print(struct.calcsize('P')*8)" > tmp_bits.txt
        set /p BITS=<tmp_bits.txt
        del tmp_bits.txt
        if "!BITS!"=="64" (
            pip install --find-links https://github.com/cgohlke/talib-build/releases/latest TA-Lib
        )
        if errorlevel 1 (
            echo.
            echo [ACTION REQUIRED] Auto-install of TA-Lib failed.
            echo Download the .whl manually from:
            echo   https://github.com/cgohlke/talib-build/releases
            echo Then run: pip install TA_Lib-0.4.XX-cpXXX-cpXXX-win_amd64.whl
            echo After that, re-run this script.
            pause
            exit /b 1
        )
    )
) else (
    echo [OK] TA-Lib already installed, skipping.
)

echo.
echo [3/5] Installing trade2 package (editable)...
pip install -e "code3.0/[ai]"
if errorlevel 1 (
    echo [ERROR] Package install failed. Check error above.
    pause
    exit /b 1
)
echo [OK] trade2 package installed

echo.
echo [4/5] Verifying CLI entry points...
trade2 --help >nul 2>&1
if errorlevel 1 (
    echo [WARN] trade2 CLI not found in PATH. Try running in a new terminal.
) else (
    echo [OK] trade2 CLI works
)

echo.
echo [5/5] Checking data files...
set MISSING_DATA=0
if not exist "code3.0\data\raw\XAUUSD_1H_2019_2025.csv" (
    echo [MISSING] code3.0\data\raw\XAUUSD_1H_2019_2025.csv
    set MISSING_DATA=1
)
if not exist "code3.0\data\raw\XAUUSD_5M_2019_2025.csv" (
    echo [MISSING] code3.0\data\raw\XAUUSD_5M_2019_2025.csv
    set MISSING_DATA=1
)
if "!MISSING_DATA!"=="1" (
    echo.
    echo [ACTION REQUIRED] Copy the missing CSV files to code3.0\data\raw\
    echo These are large files excluded from git. Transfer them from the original machine.
) else (
    echo [OK] Data files found
)

echo.
echo [OPTIONAL] For live MT5 trading, create code3.0\.env with:
echo   MT5_LOGIN=your_login
echo   MT5_PASSWORD=your_password
echo   MT5_SERVER=Exness-MT5Trial
echo Then run: pip install MetaTrader5

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Quick start:
echo    trade2 --config code3.0\configs\xauusd_mtf.yaml --retrain-model
echo    tv_research --max-ideas 5 --source seed
echo    scalp_research --max-ideas 10 --source seed
echo ============================================================
pause
