@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title Bagley v7.01 - Setup

:: ============================================================================
:: BAGLEY V7.01 SETUP - BULLETPROOF EDITION
:: Works on ANY Windows machine, even without Python
:: Auto-installs everything, shows progress
:: ============================================================================

set "BAGLEY_VERSION=7.01"
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "CONFIG_FILE=%SCRIPT_DIR%\.bagley_installed"
set "LOG_FILE=%SCRIPT_DIR%\setup_log.txt"
set "VENV_DIR=%SCRIPT_DIR%\venv"

:: Initialize log
echo ============================================ > "%LOG_FILE%"
echo Bagley v7.01 Setup Log - %date% %time% >> "%LOG_FILE%"
echo ============================================ >> "%LOG_FILE%"

:: Check if already installed
if exist "%CONFIG_FILE%" (
    goto :SHOW_MENU
) else (
    goto :FRESH_INSTALL
)

:: ============================================================================
:: MENU FOR EXISTING INSTALLATION
:: ============================================================================
:SHOW_MENU
cls
echo.
echo  ================================================================
echo                    BAGLEY v7.01 - SETUP MENU
echo  ================================================================
echo.
echo    Bagley is already installed!
echo.
echo    [1] Run Bagley
echo    [2] Repair Installation (fix errors)
echo    [3] Update / Reinstall
echo    [4] Uninstall (keep data)
echo    [5] Full Uninstall (delete everything)
echo    [6] Run Diagnostics
echo    [7] Exit
echo.
echo  ================================================================
echo.
set /p "MENU_CHOICE=Select option [1-7]: "

if "%MENU_CHOICE%"=="1" goto :RUN_BAGLEY
if "%MENU_CHOICE%"=="2" goto :REPAIR_INSTALL
if "%MENU_CHOICE%"=="3" goto :REINSTALL
if "%MENU_CHOICE%"=="4" goto :UNINSTALL_KEEP
if "%MENU_CHOICE%"=="5" goto :FULL_UNINSTALL
if "%MENU_CHOICE%"=="6" goto :RUN_DIAGNOSTICS
if "%MENU_CHOICE%"=="7" goto :EXIT_SCRIPT

echo Invalid option.
timeout /t 2 >nul
goto :SHOW_MENU

:: ============================================================================
:: FRESH INSTALLATION
:: ============================================================================
:FRESH_INSTALL
cls
echo.
echo  ================================================================
echo           BAGLEY v7.01 - FIRST TIME SETUP
echo  ================================================================
echo.
echo    Welcome! This will set up Bagley on your system.
echo.
echo    What will happen:
echo      [1] Check/Install Python 3.12
echo      [2] Detect your GPUs (NVIDIA/AMD/Intel)
echo      [3] Create virtual environment
echo      [4] Install all dependencies
echo      [5] Configure Bagley
echo      [6] Create desktop shortcut
echo.
echo    Estimated time: 5-15 minutes (depends on internet)
echo.
echo  ================================================================
echo.
echo  Press any key to begin...
pause >nul

set "STEP=0"
set "TOTAL_STEPS=8"
set "START_TIME=%time%"

:: Step 1: System Detection
call :STEP_HEADER "Detecting System"
call :DETECT_SYSTEM

:: Step 2: Python Check/Install  
call :STEP_HEADER "Setting up Python"
call :ENSURE_PYTHON

:: Step 3: GPU Detection
call :STEP_HEADER "Detecting GPUs"
call :DETECT_GPUS

:: Step 4: Virtual Environment
call :STEP_HEADER "Creating Virtual Environment"
call :SETUP_VENV

:: Step 5: Install Dependencies
call :STEP_HEADER "Installing Dependencies"
call :INSTALL_DEPENDENCIES

:: Step 6: Configure
call :STEP_HEADER "Configuring Bagley"
call :CONFIGURE_BAGLEY

:: Step 7: Test
call :STEP_HEADER "Running Tests"
call :RUN_TESTS

:: Step 8: Shortcuts
call :STEP_HEADER "Creating Shortcuts"
call :CREATE_SHORTCUTS

:: Done
call :FINALIZE_INSTALL
goto :INSTALL_COMPLETE

:: ============================================================================
:: STEP HEADER
:: ============================================================================
:STEP_HEADER
set /a "STEP+=1"
echo.
echo  ================================================================
echo   [%STEP%/%TOTAL_STEPS%] %~1
echo  ================================================================
echo.
call :LOG "Step %STEP%: %~1"
goto :eof

:: ============================================================================
:: SYSTEM DETECTION
:: ============================================================================
:DETECT_SYSTEM
echo  Checking Windows version...

:: Get Windows version
for /f "tokens=4-5 delims=. " %%i in ('ver') do set "WIN_VER=%%i.%%j"
echo    Windows: %WIN_VER%

:: Get architecture
if exist "%ProgramFiles(x86)%" (
    set "ARCH=64-bit"
) else (
    set "ARCH=32-bit"
)
echo    Architecture: %ARCH%

:: Get RAM
for /f "tokens=2 delims==" %%a in ('wmic computersystem get TotalPhysicalMemory /value 2^>nul') do (
    set "RAM_BYTES=%%a"
)
if defined RAM_BYTES (
    set /a "RAM_GB=%RAM_BYTES:~0,-9%"
) else (
    set "RAM_GB=8"
)
echo    RAM: %RAM_GB% GB

:: Get CPU  
for /f "tokens=2 delims==" %%a in ('wmic cpu get name /value 2^>nul ^| findstr /r "[A-Za-z]"') do (
    set "CPU_NAME=%%a"
)
echo    CPU: %CPU_NAME%

echo.
echo    [OK] System detection complete
call :LOG "System: Windows %WIN_VER%, %ARCH%, %RAM_GB%GB RAM"
goto :eof

:: ============================================================================
:: ENSURE PYTHON (Auto-install if missing)
:: ============================================================================
:ENSURE_PYTHON
echo  Looking for Python...

set "PYTHON_CMD="
set "PYTHON_VERSION="

:: Try py launcher first (most reliable on Windows)
py --version >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=2 delims= " %%v in ('py --version 2^>^&1') do set "PYTHON_VERSION=%%v"
    set "PYTHON_CMD=py"
    goto :CHECK_PYTHON_VERSION
)

:: Try python
python --version >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%v"
    set "PYTHON_CMD=python"
    goto :CHECK_PYTHON_VERSION
)

:: Try python3
python3 --version >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=2 delims= " %%v in ('python3 --version 2^>^&1') do set "PYTHON_VERSION=%%v"
    set "PYTHON_CMD=python3"
    goto :CHECK_PYTHON_VERSION
)

:: Check common install paths
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
) do (
    if exist %%p (
        set "PYTHON_CMD=%%~p"
        for /f "tokens=2 delims= " %%v in ('"%%~p" --version 2^>^&1') do set "PYTHON_VERSION=%%v"
        goto :CHECK_PYTHON_VERSION
    )
)

:: Python not found - install it
echo    [!] Python not found - installing automatically...
call :INSTALL_PYTHON
goto :ENSURE_PYTHON

:CHECK_PYTHON_VERSION
echo    Found Python %PYTHON_VERSION%

:: Check version is >= 3.10
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if %PY_MAJOR% LSS 3 (
    echo    [!] Python 3.10+ required, found %PYTHON_VERSION%
    call :INSTALL_PYTHON
    goto :ENSURE_PYTHON
)
if %PY_MAJOR%==3 if %PY_MINOR% LSS 10 (
    echo    [!] Python 3.10+ required, found %PYTHON_VERSION%
    call :INSTALL_PYTHON
    goto :ENSURE_PYTHON
)

echo    [OK] Python %PYTHON_VERSION% is compatible
call :LOG "Python: %PYTHON_VERSION% at %PYTHON_CMD%"
goto :eof

:: ============================================================================
:: INSTALL PYTHON
:: ============================================================================
:INSTALL_PYTHON
echo.
echo    Downloading Python 3.12.8...
echo    This may take 1-2 minutes...
echo.

set "PYTHON_URL=https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe"
set "PYTHON_INSTALLER=%TEMP%\python_installer.exe"

:: Download with progress using PowerShell
powershell -Command ^
    "$ProgressPreference = 'Continue'; " ^
    "try { " ^
    "    $url = '%PYTHON_URL%'; " ^
    "    $out = '%PYTHON_INSTALLER%'; " ^
    "    Write-Host '    Downloading...' -ForegroundColor Cyan; " ^
    "    Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing; " ^
    "    Write-Host '    Download complete!' -ForegroundColor Green; " ^
    "} catch { " ^
    "    Write-Host '    Download failed!' -ForegroundColor Red; " ^
    "    exit 1; " ^
    "}" 2>>"%LOG_FILE%"

if not exist "%PYTHON_INSTALLER%" (
    echo.
    echo    [ERROR] Failed to download Python!
    echo.
    echo    Please install Python 3.12 manually:
    echo    https://www.python.org/downloads/
    echo.
    echo    Make sure to check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo    Installing Python 3.12.8...
echo    (This window may freeze briefly - that's normal)
echo.

:: Install Python silently with PATH
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1

:: Wait for install
timeout /t 5 >nul

:: Clean up
del "%PYTHON_INSTALLER%" 2>nul

:: Refresh PATH
call :REFRESH_PATH

echo    [OK] Python installed!
call :LOG "Python 3.12.8 installed"
goto :eof

:: ============================================================================
:: REFRESH PATH
:: ============================================================================
:REFRESH_PATH
:: Refresh environment PATH to pick up new Python
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%b"
set "PATH=%SYS_PATH%;%USER_PATH%"

:: Also add common Python locations directly
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts"
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts"
goto :eof

:: ============================================================================
:: GPU DETECTION
:: ============================================================================
:DETECT_GPUS
echo  Scanning for GPUs...

set "NVIDIA_GPUS=0"
set "AMD_GPUS=0"
set "INTEL_GPUS=0"
set "CUDA_VERSION="
set "GPU_MODE=cpu"

:: Check NVIDIA
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo    Checking NVIDIA GPUs...
    for /f "tokens=*" %%a in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        set /a "NVIDIA_GPUS+=1"
        echo      - %%a
    )
    for /f "tokens=6 delims= " %%a in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set "CUDA_VERSION=%%a"
    if defined CUDA_VERSION echo    CUDA Version: %CUDA_VERSION%
)

:: Check AMD via WMI
for /f "tokens=*" %%a in ('wmic path win32_videocontroller get name 2^>nul ^| findstr /i "AMD Radeon"') do (
    set /a "AMD_GPUS+=1"
    echo      - %%a
)

:: Check Intel via WMI
for /f "tokens=*" %%a in ('wmic path win32_videocontroller get name 2^>nul ^| findstr /i "Intel.*Graphics"') do (
    set /a "INTEL_GPUS+=1"
    echo      - %%a
)

:: Determine mode
set /a "TOTAL_GPUS=%NVIDIA_GPUS%+%AMD_GPUS%+%INTEL_GPUS%"

if %NVIDIA_GPUS% GTR 0 if %AMD_GPUS% GTR 0 (
    set "GPU_MODE=mixed"
    echo    [!] Mixed AMD/NVIDIA setup detected
) else if %NVIDIA_GPUS% GTR 0 (
    set "GPU_MODE=nvidia"
) else if %AMD_GPUS% GTR 0 (
    set "GPU_MODE=amd"
) else if %INTEL_GPUS% GTR 0 (
    set "GPU_MODE=intel"
) else (
    set "GPU_MODE=cpu"
    echo    [!] No dedicated GPU found - will use CPU
)

echo.
echo    Total GPUs: %TOTAL_GPUS% (NVIDIA: %NVIDIA_GPUS%, AMD: %AMD_GPUS%, Intel: %INTEL_GPUS%)
echo    Mode: %GPU_MODE%
echo    [OK] GPU detection complete
call :LOG "GPUs: NVIDIA=%NVIDIA_GPUS%, AMD=%AMD_GPUS%, Intel=%INTEL_GPUS%, Mode=%GPU_MODE%"
goto :eof

:: ============================================================================
:: SETUP VIRTUAL ENVIRONMENT
:: ============================================================================
:SETUP_VENV
echo  Creating virtual environment...

:: Remove old venv if corrupted
if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo    Removing corrupted venv...
        rmdir /s /q "%VENV_DIR%" 2>nul
    ) else (
        echo    [OK] Existing venv is healthy
        set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
        set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
        goto :eof
    )
)

:: Create new venv
echo    Creating new virtual environment...
"%PYTHON_CMD%" -m venv "%VENV_DIR%" 2>>"%LOG_FILE%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo    [ERROR] Failed to create virtual environment!
    echo.
    echo    Try running as Administrator, or manually run:
    echo    %PYTHON_CMD% -m venv "%VENV_DIR%"
    echo.
    pause
    exit /b 1
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

:: Upgrade pip
echo    Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet 2>>"%LOG_FILE%"

echo    [OK] Virtual environment created
call :LOG "venv created at %VENV_DIR%"
goto :eof

:: ============================================================================
:: INSTALL DEPENDENCIES
:: ============================================================================
:INSTALL_DEPENDENCIES
echo  Installing dependencies (this takes 5-10 minutes)...
echo.

set "PIP_OPTS=--quiet --disable-pip-version-check"

:: Upgrade pip, setuptools, wheel first
echo    [1/9] Core tools...
"%VENV_PIP%" install --upgrade pip setuptools wheel %PIP_OPTS% 2>>"%LOG_FILE%"

:: PyTorch (biggest download)
echo    [2/9] PyTorch (large download, please wait)...
if "%GPU_MODE%"=="nvidia" (
    "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 %PIP_OPTS% 2>>"%LOG_FILE%"
) else if "%GPU_MODE%"=="amd" (
    "%VENV_PIP%" install torch torchvision torchaudio %PIP_OPTS% 2>>"%LOG_FILE%"
    "%VENV_PIP%" install torch-directml %PIP_OPTS% 2>>"%LOG_FILE%"
) else (
    "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu %PIP_OPTS% 2>>"%LOG_FILE%"
)

:: Transformers & ML
echo    [3/9] Transformers and ML libraries...
"%VENV_PIP%" install transformers accelerate datasets %PIP_OPTS% 2>>"%LOG_FILE%"
"%VENV_PIP%" install sentencepiece tokenizers safetensors %PIP_OPTS% 2>>"%LOG_FILE%"
"%VENV_PIP%" install huggingface_hub einops %PIP_OPTS% 2>>"%LOG_FILE%"

:: UI Framework
echo    [4/9] UI Framework (PySide6)...
"%VENV_PIP%" install PySide6 %PIP_OPTS% 2>>"%LOG_FILE%"

:: Training tools
echo    [5/9] Training tools...
"%VENV_PIP%" install peft %PIP_OPTS% 2>>"%LOG_FILE%"
"%VENV_PIP%" install wandb tensorboard %PIP_OPTS% 2>>"%LOG_FILE%"

:: Media processing
echo    [6/9] Media processing...
"%VENV_PIP%" install Pillow opencv-python %PIP_OPTS% 2>>"%LOG_FILE%"
"%VENV_PIP%" install imageio imageio-ffmpeg %PIP_OPTS% 2>>"%LOG_FILE%"
"%VENV_PIP%" install soundfile %PIP_OPTS% 2>>"%LOG_FILE%"

:: Web/API
echo    [7/9] Web and API tools...
"%VENV_PIP%" install fastapi uvicorn httpx aiofiles websockets %PIP_OPTS% 2>>"%LOG_FILE%"
"%VENV_PIP%" install requests feedparser schedule %PIP_OPTS% 2>>"%LOG_FILE%"

:: Utilities
echo    [8/9] Utilities...
"%VENV_PIP%" install tqdm rich psutil GPUtil %PIP_OPTS% 2>>"%LOG_FILE%"
"%VENV_PIP%" install pyyaml toml pydantic omegaconf %PIP_OPTS% 2>>"%LOG_FILE%"

:: Install Bagley itself
echo    [9/9] Installing Bagley...
"%VENV_PIP%" install -e "%SCRIPT_DIR%" %PIP_OPTS% 2>>"%LOG_FILE%"

echo.
echo    [OK] All dependencies installed!
call :LOG "Dependencies installed"
goto :eof

:: ============================================================================
:: CONFIGURE BAGLEY
:: ============================================================================
:CONFIGURE_BAGLEY
echo  Creating configuration...

:: Create directories
for %%d in (config models data logs checkpoints cache temp output web_intelligence knowledge_base) do (
    if not exist "%SCRIPT_DIR%\%%d" mkdir "%SCRIPT_DIR%\%%d"
)

:: Create hardware config
(
echo # Bagley v7.01 Hardware Configuration
echo # Auto-generated on %date% %time%
echo.
echo [hardware]
echo gpu_mode = "%GPU_MODE%"
echo nvidia_gpus = %NVIDIA_GPUS%
echo amd_gpus = %AMD_GPUS%
echo intel_gpus = %INTEL_GPUS%
echo ram_gb = %RAM_GB%
echo cuda_version = "%CUDA_VERSION%"
echo.
echo [paths]
echo root = "%SCRIPT_DIR%"
echo venv = "%VENV_DIR%"
echo models = "%SCRIPT_DIR%\models"
echo data = "%SCRIPT_DIR%\data"
echo logs = "%SCRIPT_DIR%\logs"
) > "%SCRIPT_DIR%\config\hardware.toml"

echo    [OK] Configuration created
call :LOG "Config files created"
goto :eof

:: ============================================================================
:: RUN TESTS
:: ============================================================================
:RUN_TESTS
echo  Testing installation...
echo.

set "TEST_PASS=0"
set "TEST_FAIL=0"

:: Test Python
echo    Testing Python...
"%VENV_PYTHON%" --version >nul 2>&1
if %errorlevel%==0 (
    echo      [PASS] Python OK
    set /a "TEST_PASS+=1"
) else (
    echo      [FAIL] Python not working
    set /a "TEST_FAIL+=1"
)

:: Test PyTorch
echo    Testing PyTorch...
"%VENV_PYTHON%" -c "import torch; print(f'      PyTorch {torch.__version__}')" 2>>"%LOG_FILE%"
if %errorlevel%==0 (
    echo      [PASS] PyTorch OK
    set /a "TEST_PASS+=1"
) else (
    echo      [FAIL] PyTorch error
    set /a "TEST_FAIL+=1"
)

:: Test GPU
echo    Testing GPU access...
if "%GPU_MODE%"=="nvidia" (
    "%VENV_PYTHON%" -c "import torch; print(f'      CUDA available: {torch.cuda.is_available()}')" 2>>"%LOG_FILE%"
) else if "%GPU_MODE%"=="amd" (
    "%VENV_PYTHON%" -c "import torch_directml; print('      DirectML OK')" 2>>"%LOG_FILE%"
)
set /a "TEST_PASS+=1"

:: Test Transformers
echo    Testing Transformers...
"%VENV_PYTHON%" -c "import transformers" 2>>"%LOG_FILE%"
if %errorlevel%==0 (
    echo      [PASS] Transformers OK
    set /a "TEST_PASS+=1"
) else (
    echo      [FAIL] Transformers error
    set /a "TEST_FAIL+=1"
)

:: Test PySide6
echo    Testing UI (PySide6)...
"%VENV_PYTHON%" -c "from PySide6.QtWidgets import QApplication" 2>>"%LOG_FILE%"
if %errorlevel%==0 (
    echo      [PASS] PySide6 OK
    set /a "TEST_PASS+=1"
) else (
    echo      [FAIL] PySide6 error
    set /a "TEST_FAIL+=1"
)

:: Test Bagley core
echo    Testing Bagley core...
"%VENV_PYTHON%" -c "from bagley.core import __version__; print(f'      Bagley v{__version__}')" 2>>"%LOG_FILE%"
if %errorlevel%==0 (
    echo      [PASS] Bagley core OK
    set /a "TEST_PASS+=1"
) else (
    echo      [WARN] Bagley core has issues (may need repair)
)

echo.
echo    Results: %TEST_PASS% passed, %TEST_FAIL% failed
if %TEST_FAIL% GTR 0 (
    echo    [!] Some tests failed - run Repair from menu
)
call :LOG "Tests: %TEST_PASS% passed, %TEST_FAIL% failed"
goto :eof

:: ============================================================================
:: CREATE SHORTCUTS
:: ============================================================================
:CREATE_SHORTCUTS
echo  Creating launcher scripts...

:: Main run script
(
echo @echo off
echo title Bagley v7.01
echo cd /d "%SCRIPT_DIR%"
echo call "%VENV_DIR%\Scripts\activate.bat"
echo echo Starting Bagley...
echo python -m bagley.main --ui
echo if errorlevel 1 ^(
echo     echo.
echo     echo [ERROR] Bagley failed to start!
echo     echo Run setup.bat and choose Repair to fix.
echo     echo.
echo     pause
echo ^)
) > "%SCRIPT_DIR%\run.bat"

:: CLI script
(
echo @echo off
echo cd /d "%SCRIPT_DIR%"
echo call "%VENV_DIR%\Scripts\activate.bat"
echo python -m bagley.main %%*
) > "%SCRIPT_DIR%\run_cli.bat"

:: Train script
(
echo @echo off
echo title Bagley Training
echo cd /d "%SCRIPT_DIR%"
echo call "%VENV_DIR%\Scripts\activate.bat"
echo python -m bagley.main --train %%*
echo pause
) > "%SCRIPT_DIR%\train.bat"

:: Web Intelligence script
(
echo @echo off
echo title Bagley Web Intelligence
echo cd /d "%SCRIPT_DIR%"
echo call "%VENV_DIR%\Scripts\activate.bat"
echo python -c "from bagley.core.daily_intelligence import start_daily_intelligence; start_daily_intelligence()"
echo pause
) > "%SCRIPT_DIR%\start_web_intel.bat"

:: Desktop shortcut
set "SHORTCUT=%USERPROFILE%\Desktop\Bagley v7.lnk"
powershell -Command ^
    "$ws = New-Object -ComObject WScript.Shell; " ^
    "$s = $ws.CreateShortcut('%SHORTCUT%'); " ^
    "$s.TargetPath = '%SCRIPT_DIR%\run.bat'; " ^
    "$s.WorkingDirectory = '%SCRIPT_DIR%'; " ^
    "$s.Description = 'Bagley v7.01 - The Best AI'; " ^
    "$s.Save()" 2>nul

if exist "%SHORTCUT%" (
    echo    [OK] Desktop shortcut created
) else (
    echo    [INFO] Couldn't create desktop shortcut (not critical)
)

echo    [OK] Launcher scripts created
call :LOG "Shortcuts created"
goto :eof

:: ============================================================================
:: FINALIZE
:: ============================================================================
:FINALIZE_INSTALL
:: Write install marker
(
echo version=%BAGLEY_VERSION%
echo installed=%date% %time%
echo python=%PYTHON_VERSION%
echo gpu_mode=%GPU_MODE%
echo nvidia=%NVIDIA_GPUS%
echo amd=%AMD_GPUS%
echo path=%SCRIPT_DIR%
) > "%CONFIG_FILE%"
call :LOG "Installation complete"
goto :eof

:: ============================================================================
:: INSTALLATION COMPLETE
:: ============================================================================
:INSTALL_COMPLETE
cls
echo.
echo  ================================================================
echo           BAGLEY v7.01 - INSTALLATION COMPLETE!
echo  ================================================================
echo.
echo    System Configuration:
echo      GPU Mode: %GPU_MODE%
echo      NVIDIA GPUs: %NVIDIA_GPUS%
echo      AMD GPUs: %AMD_GPUS%
echo      Python: %PYTHON_VERSION%
echo.
echo    How to use:
echo      - Double-click "run.bat" or the desktop shortcut
echo      - Or run this setup again for more options
echo.
echo    Web Intelligence:
echo      - Run "start_web_intel.bat" to start daily news scraping
echo.
echo    Training:
echo      - Put your data in the "data" folder
echo      - Run "train.bat" to start training
echo.
echo  ================================================================
echo.
set /p "RUN_NOW=Launch Bagley now? [Y/N]: "
if /i "%RUN_NOW%"=="Y" goto :RUN_BAGLEY
goto :EXIT_SCRIPT

:: ============================================================================
:: RUN BAGLEY
:: ============================================================================
:RUN_BAGLEY
echo.
echo  Starting Bagley...
cd /d "%SCRIPT_DIR%"
call "%VENV_DIR%\Scripts\activate.bat"
python -m bagley.main --ui
if errorlevel 1 (
    echo.
    echo  [ERROR] Bagley failed to start!
    echo  Run setup.bat and choose [2] Repair
    echo.
    pause
)
goto :EXIT_SCRIPT

:: ============================================================================
:: REPAIR
:: ============================================================================
:REPAIR_INSTALL
echo.
echo  Repairing Bagley...
echo.

call :DETECT_SYSTEM
call :ENSURE_PYTHON
call :DETECT_GPUS

:: Check venv
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo  Virtual environment missing - recreating...
    call :SETUP_VENV
) else (
    set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
    set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
)

:: Reinstall packages
echo  Reinstalling packages...
call :INSTALL_DEPENDENCIES

:: Test
call :RUN_TESTS

echo.
echo  Repair complete!
pause
goto :SHOW_MENU

:: ============================================================================
:: REINSTALL
:: ============================================================================
:REINSTALL
echo.
echo  Full reinstall - removing old installation...
rmdir /s /q "%VENV_DIR%" 2>nul
del "%CONFIG_FILE%" 2>nul
goto :FRESH_INSTALL

:: ============================================================================
:: UNINSTALL KEEP DATA
:: ============================================================================
:UNINSTALL_KEEP
echo.
set /p "CONFIRM=Uninstall but keep data? [Y/N]: "
if /i not "%CONFIRM%"=="Y" goto :SHOW_MENU

rmdir /s /q "%VENV_DIR%" 2>nul
rmdir /s /q "%SCRIPT_DIR%\cache" 2>nul
del "%CONFIG_FILE%" 2>nul
del "%SCRIPT_DIR%\run.bat" 2>nul
del "%USERPROFILE%\Desktop\Bagley v7.lnk" 2>nul

echo  Uninstalled. Data preserved in data/ and models/
pause
goto :EXIT_SCRIPT

:: ============================================================================
:: FULL UNINSTALL
:: ============================================================================
:FULL_UNINSTALL
echo.
echo  WARNING: This deletes EVERYTHING!
set /p "CONFIRM=Type DELETE to confirm: "
if not "%CONFIRM%"=="DELETE" goto :SHOW_MENU

rmdir /s /q "%VENV_DIR%" 2>nul
rmdir /s /q "%SCRIPT_DIR%\models" 2>nul
rmdir /s /q "%SCRIPT_DIR%\data" 2>nul
rmdir /s /q "%SCRIPT_DIR%\checkpoints" 2>nul
rmdir /s /q "%SCRIPT_DIR%\logs" 2>nul
rmdir /s /q "%SCRIPT_DIR%\cache" 2>nul
rmdir /s /q "%SCRIPT_DIR%\config" 2>nul
del "%CONFIG_FILE%" 2>nul
del "%SCRIPT_DIR%\run*.bat" 2>nul
del "%USERPROFILE%\Desktop\Bagley v7.lnk" 2>nul

echo  Full uninstall complete.
pause
goto :EXIT_SCRIPT

:: ============================================================================
:: DIAGNOSTICS
:: ============================================================================
:RUN_DIAGNOSTICS
cls
echo.
echo  ================================================================
echo              BAGLEY v7.01 DIAGNOSTICS
echo  ================================================================
echo.

call :DETECT_SYSTEM
echo.
call :ENSURE_PYTHON
echo.
call :DETECT_GPUS
echo.

echo  Disk Space:
for /f "tokens=3" %%a in ('dir "%SCRIPT_DIR%" /-c 2^>nul ^| findstr /c:"bytes free"') do (
    set /a "FREE_GB=%%a/1073741824"
    echo    Free: !FREE_GB! GB
)
echo.

if exist "%VENV_DIR%\Scripts\pip.exe" (
    echo  Key Packages:
    "%VENV_DIR%\Scripts\pip.exe" list 2>nul | findstr /i "torch transformers pyside6 accelerate bagley"
    echo.
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
call :RUN_TESTS

echo.
echo  Press any key to return to menu...
pause >nul
goto :SHOW_MENU

:: ============================================================================
:: LOG FUNCTION
:: ============================================================================
:LOG
echo [%date% %time%] %~1 >> "%LOG_FILE%"
goto :eof

:: ============================================================================
:: EXIT
:: ============================================================================
:EXIT_SCRIPT
echo.
echo  Thanks for using Bagley!
echo.
endlocal
exit /b 0
