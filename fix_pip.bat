@echo off
echo ============================================================
echo  Fix Python PATH - trade2.0
echo ============================================================
echo.

:: Find Python installation
echo Searching for Python...
for /f "tokens=*" %%p in ('where python 2^>nul') do (
    set PYTHON_EXE=%%p
    goto :found_python
)

:: Try common install locations
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python310"
    "C:\Python313"
    "C:\Python312"
    "C:\Python311"
    "C:\Python310"
) do (
    if exist "%%~d\python.exe" (
        set PYTHON_DIR=%%~d
        goto :fix_path
    )
)

echo [ERROR] Python not found anywhere on this machine.
echo Please install Python 3.10+ from https://python.org
echo During install, CHECK "Add Python to PATH"
pause
exit /b 1

:found_python
echo [OK] Python found at: %PYTHON_EXE%
python --version
pip --version >nul 2>&1
if not errorlevel 1 (
    echo [OK] pip already works! You can run setup.bat now.
    pause
    exit /b 0
)
for %%f in ("%PYTHON_EXE%") do set PYTHON_DIR=%%~dpf
set PYTHON_DIR=%PYTHON_DIR:~0,-1%

:fix_path
echo Python directory: %PYTHON_DIR%
echo.
echo [1/2] Adding Python to system PATH...
setx PATH "%PATH%;%PYTHON_DIR%;%PYTHON_DIR%\Scripts" /M >nul 2>&1
if errorlevel 1 (
    echo Trying user PATH instead (no admin rights)...
    setx PATH "%PATH%;%PYTHON_DIR%;%PYTHON_DIR%\Scripts"
)

echo [2/2] Ensuring pip is installed...
"%PYTHON_DIR%\python.exe" -m ensurepip --upgrade

echo.
echo ============================================================
echo  Done! CLOSE THIS WINDOW and open a NEW terminal, then run:
echo    python --version
echo    pip --version
echo    setup.bat
echo ============================================================
pause
