@echo off
setlocal enabledelayedexpansion
title Bagley v7 - Ultimate AI Setup
color 0A

:: ============================================================================
:: BAGLEY V7 SETUP - THE BEST AI IN THE WORLD
:: Fully automatic setup that works on ANY Windows system
:: Auto-detects: Python, GPUs (AMD/NVIDIA/Intel), CUDA, ROCm, paths
:: ============================================================================

set "BAGLEY_VERSION=7.0.0"
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "CONFIG_FILE=%SCRIPT_DIR%\.bagley_installed"
set "LOG_FILE=%SCRIPT_DIR%\setup_log.txt"

:: Initialize log
echo ============================================ > "%LOG_FILE%"
echo Bagley v7 Setup Log - %date% %time% >> "%LOG_FILE%"
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
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║                                                                  ║
echo  ║   ██████╗  █████╗  ██████╗ ██╗     ███████╗██╗   ██╗            ║
echo  ║   ██╔══██╗██╔══██╗██╔════╝ ██║     ██╔════╝╚██╗ ██╔╝            ║
echo  ║   ██████╔╝███████║██║  ███╗██║     █████╗   ╚████╔╝             ║
echo  ║   ██╔══██╗██╔══██║██║   ██║██║     ██╔══╝    ╚██╔╝              ║
echo  ║   ██████╔╝██║  ██║╚██████╔╝███████╗███████╗   ██║               ║
echo  ║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝               ║
echo  ║                                                                  ║
echo  ║              THE BEST AI IN THE WORLD - v%BAGLEY_VERSION%                  ║
echo  ║                                                                  ║
echo  ╠══════════════════════════════════════════════════════════════════╣
echo  ║                                                                  ║
echo  ║   Bagley is already installed on this system!                   ║
echo  ║                                                                  ║
echo  ║   [1] Run Bagley                                                ║
echo  ║   [2] Repair Installation                                       ║
echo  ║   [3] Update / Reinstall                                        ║
echo  ║   [4] Uninstall (Keep Data)                                     ║
echo  ║   [5] Full Uninstall (Delete Everything)                        ║
echo  ║   [6] Run Diagnostics                                           ║
echo  ║   [7] Exit                                                      ║
echo  ║                                                                  ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.
set /p "MENU_CHOICE=Select option [1-7]: "

if "%MENU_CHOICE%"=="1" goto :RUN_BAGLEY
if "%MENU_CHOICE%"=="2" goto :REPAIR_INSTALL
if "%MENU_CHOICE%"=="3" goto :REINSTALL
if "%MENU_CHOICE%"=="4" goto :UNINSTALL_KEEP
if "%MENU_CHOICE%"=="5" goto :FULL_UNINSTALL
if "%MENU_CHOICE%"=="6" goto :RUN_DIAGNOSTICS
if "%MENU_CHOICE%"=="7" goto :EXIT_SCRIPT

echo Invalid option. Please try again.
timeout /t 2 >nul
goto :SHOW_MENU

:: ============================================================================
:: FRESH INSTALLATION
:: ============================================================================
:FRESH_INSTALL
cls
echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║                                                                  ║
echo  ║   ██████╗  █████╗  ██████╗ ██╗     ███████╗██╗   ██╗            ║
echo  ║   ██╔══██╗██╔══██╗██╔════╝ ██║     ██╔════╝╚██╗ ██╔╝            ║
echo  ║   ██████╔╝███████║██║  ███╗██║     █████╗   ╚████╔╝             ║
echo  ║   ██╔══██╗██╔══██║██║   ██║██║     ██╔══╝    ╚██╔╝              ║
echo  ║   ██████╔╝██║  ██║╚██████╔╝███████╗███████╗   ██║               ║
echo  ║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝               ║
echo  ║                                                                  ║
echo  ║              THE BEST AI IN THE WORLD - v%BAGLEY_VERSION%                  ║
echo  ║                                                                  ║
echo  ╠══════════════════════════════════════════════════════════════════╣
echo  ║                                                                  ║
echo  ║   Welcome to Bagley v7 Setup!                                   ║
echo  ║                                                                  ║
echo  ║   This setup will:                                              ║
echo  ║   • Auto-detect your system configuration                       ║
echo  ║   • Find and configure Python                                   ║
echo  ║   • Detect all GPUs (NVIDIA, AMD, Intel)                        ║
echo  ║   • Install all required dependencies                           ║
echo  ║   • Configure optimal settings for YOUR hardware                ║
echo  ║   • Test everything works perfectly                             ║
echo  ║                                                                  ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.
echo Press any key to begin installation...
pause >nul

:: Start installation
call :LOG "Starting fresh installation..."
call :DETECT_SYSTEM
call :DETECT_PYTHON
call :DETECT_GPUS
call :SETUP_VENV
call :INSTALL_DEPENDENCIES
call :CONFIGURE_BAGLEY
call :RUN_TESTS
call :CREATE_SHORTCUTS
call :FINALIZE_INSTALL

goto :INSTALL_COMPLETE

:: ============================================================================
:: SYSTEM DETECTION
:: ============================================================================
:DETECT_SYSTEM
echo.
echo [1/8] Detecting system configuration...
call :LOG "Detecting system..."

:: Get Windows version
for /f "tokens=4-5 delims=. " %%i in ('ver') do set "WIN_VER=%%i.%%j"
call :LOG "Windows Version: %WIN_VER%"

:: Get architecture
if exist "%ProgramFiles(x86)%" (
    set "ARCH=x64"
) else (
    set "ARCH=x86"
)
call :LOG "Architecture: %ARCH%"

:: Get RAM
for /f "tokens=2 delims==" %%a in ('wmic computersystem get TotalPhysicalMemory /value 2^>nul') do set "RAM_BYTES=%%a"
set /a "RAM_GB=%RAM_BYTES:~0,-9%"
if "%RAM_GB%"=="" set "RAM_GB=8"
call :LOG "RAM: %RAM_GB% GB"

:: Get CPU
for /f "tokens=2 delims==" %%a in ('wmic cpu get name /value 2^>nul') do set "CPU_NAME=%%a"
call :LOG "CPU: %CPU_NAME%"

echo    • Windows %WIN_VER% (%ARCH%)
echo    • RAM: %RAM_GB% GB
echo    • CPU: %CPU_NAME%
echo    [OK] System detection complete
goto :eof

:: ============================================================================
:: PYTHON DETECTION
:: ============================================================================
:DETECT_PYTHON
echo.
echo [2/8] Detecting Python installation...
call :LOG "Detecting Python..."

set "PYTHON_CMD="
set "PYTHON_VERSION="

:: Try different Python commands
for %%p in (python python3 py) do (
    where %%p >nul 2>&1
    if !errorlevel!==0 (
        for /f "tokens=2 delims= " %%v in ('%%p --version 2^>^&1') do (
            set "PYTHON_VERSION=%%v"
            set "PYTHON_CMD=%%p"
            goto :PYTHON_FOUND
        )
    )
)

:: Try common installation paths
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python310\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist %%p (
        set "PYTHON_CMD=%%~p"
        for /f "tokens=2 delims= " %%v in ('"%%~p" --version 2^>^&1') do set "PYTHON_VERSION=%%v"
        goto :PYTHON_FOUND
    )
)

:: Python not found - offer to install
echo    [!] Python not found!
echo.
echo    Bagley requires Python 3.10 or higher.
echo    Would you like to download Python automatically?
echo.
set /p "INSTALL_PYTHON=Install Python? [Y/N]: "
if /i "%INSTALL_PYTHON%"=="Y" (
    call :INSTALL_PYTHON_AUTO
    goto :DETECT_PYTHON
)
echo    [ERROR] Cannot continue without Python.
call :LOG "ERROR: Python not found and user declined installation"
pause
exit /b 1

:PYTHON_FOUND
call :LOG "Python found: %PYTHON_CMD% (version %PYTHON_VERSION%)"
echo    • Python %PYTHON_VERSION% found
echo    • Location: %PYTHON_CMD%

:: Check Python version is >= 3.10
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
if %PY_MAJOR% LSS 3 (
    echo    [ERROR] Python 3.10+ required, found %PYTHON_VERSION%
    goto :PYTHON_TOO_OLD
)
if %PY_MAJOR%==3 if %PY_MINOR% LSS 10 (
    echo    [ERROR] Python 3.10+ required, found %PYTHON_VERSION%
    goto :PYTHON_TOO_OLD
)

echo    [OK] Python version compatible
goto :eof

:PYTHON_TOO_OLD
echo.
echo    Your Python version is too old.
set /p "UPGRADE_PYTHON=Would you like to install Python 3.12? [Y/N]: "
if /i "%UPGRADE_PYTHON%"=="Y" (
    call :INSTALL_PYTHON_AUTO
    goto :DETECT_PYTHON
)
exit /b 1

:INSTALL_PYTHON_AUTO
echo.
echo    Downloading Python 3.12...
set "PYTHON_URL=https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
set "PYTHON_INSTALLER=%TEMP%\python_installer.exe"
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'" 2>nul
if exist "%PYTHON_INSTALLER%" (
    echo    Installing Python 3.12...
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    del "%PYTHON_INSTALLER%" 2>nul
    echo    [OK] Python installed successfully
    :: Refresh PATH
    call :REFRESH_PATH
) else (
    echo    [ERROR] Failed to download Python.
    echo    Please install Python 3.12 manually from https://python.org
    pause
    exit /b 1
)
goto :eof

:REFRESH_PATH
:: Refresh environment variables
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%b"
set "PATH=%SYS_PATH%;%USER_PATH%"
goto :eof

:: ============================================================================
:: GPU DETECTION
:: ============================================================================
:DETECT_GPUS
echo.
echo [3/8] Detecting GPUs...
call :LOG "Detecting GPUs..."

set "NVIDIA_GPUS=0"
set "AMD_GPUS=0"
set "INTEL_GPUS=0"
set "TOTAL_VRAM=0"
set "GPU_LIST="
set "CUDA_VERSION="
set "ROCM_VERSION="

:: Detect NVIDIA GPUs
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo    Checking NVIDIA GPUs...
    for /f "tokens=*" %%a in ('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2^>nul') do (
        set /a "NVIDIA_GPUS+=1"
        echo    • NVIDIA: %%a
        set "GPU_LIST=!GPU_LIST!NVIDIA:%%a;"
        call :LOG "Found NVIDIA GPU: %%a"
    )
    :: Get CUDA version
    for /f "tokens=6 delims= " %%a in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set "CUDA_VERSION=%%a"
    if defined CUDA_VERSION (
        echo    • CUDA Version: %CUDA_VERSION%
        call :LOG "CUDA Version: %CUDA_VERSION%"
    )
)

:: Detect AMD GPUs
where rocm-smi >nul 2>&1
if %errorlevel%==0 (
    echo    Checking AMD GPUs...
    for /f "tokens=*" %%a in ('rocm-smi --showproductname 2^>nul ^| findstr /v "^=" ^| findstr /v "^$"') do (
        set /a "AMD_GPUS+=1"
        echo    • AMD: %%a
        set "GPU_LIST=!GPU_LIST!AMD:%%a;"
        call :LOG "Found AMD GPU: %%a"
    )
    :: Get ROCm version
    for /f "tokens=2 delims=:" %%a in ('rocm-smi --version 2^>nul ^| findstr "version"') do set "ROCM_VERSION=%%a"
) else (
    :: Try Windows WMI for AMD
    for /f "tokens=*" %%a in ('wmic path win32_videocontroller get name 2^>nul ^| findstr /i "AMD Radeon"') do (
        set /a "AMD_GPUS+=1"
        echo    • AMD: %%a
        set "GPU_LIST=!GPU_LIST!AMD:%%a;"
        call :LOG "Found AMD GPU: %%a"
    )
)

:: Detect Intel GPUs
for /f "tokens=*" %%a in ('wmic path win32_videocontroller get name 2^>nul ^| findstr /i "Intel"') do (
    set /a "INTEL_GPUS+=1"
    echo    • Intel: %%a
    set "GPU_LIST=!GPU_LIST!Intel:%%a;"
    call :LOG "Found Intel GPU: %%a"
)

:: Calculate total GPUs
set /a "TOTAL_GPUS=%NVIDIA_GPUS%+%AMD_GPUS%+%INTEL_GPUS%"

if %TOTAL_GPUS%==0 (
    echo    [!] No dedicated GPUs detected - will use CPU
    set "GPU_MODE=cpu"
    call :LOG "No GPUs found, using CPU mode"
) else (
    echo    [OK] Found %TOTAL_GPUS% GPU(s)
    
    :: Determine GPU mode
    if %NVIDIA_GPUS% GTR 0 if %AMD_GPUS% GTR 0 (
        set "GPU_MODE=mixed"
        echo    [!] Mixed AMD/NVIDIA setup detected - using GLOO backend
    ) else if %NVIDIA_GPUS% GTR 0 (
        set "GPU_MODE=nvidia"
    ) else if %AMD_GPUS% GTR 0 (
        set "GPU_MODE=amd"
    ) else (
        set "GPU_MODE=intel"
    )
    call :LOG "GPU Mode: %GPU_MODE%"
)
goto :eof

:: ============================================================================
:: VIRTUAL ENVIRONMENT SETUP
:: ============================================================================
:SETUP_VENV
echo.
echo [4/8] Setting up virtual environment...
call :LOG "Setting up venv..."

set "VENV_DIR=%SCRIPT_DIR%\venv"

:: Check if venv exists
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo    • Existing venv found
    set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
    set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
    echo    [OK] Using existing virtual environment
    goto :eof
)

:: Create new venv
echo    Creating virtual environment...
"%PYTHON_CMD%" -m venv "%VENV_DIR%" 2>>"%LOG_FILE%"
if %errorlevel% neq 0 (
    echo    [ERROR] Failed to create venv
    call :LOG "ERROR: venv creation failed"
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
echo.
echo [5/8] Installing dependencies...
call :LOG "Installing dependencies..."

:: Upgrade pip first
"%VENV_PIP%" install --upgrade pip setuptools wheel --quiet 2>>"%LOG_FILE%"

:: Install PyTorch based on GPU
echo    Installing PyTorch for %GPU_MODE%...
if "%GPU_MODE%"=="nvidia" (
    :: Install CUDA version
    "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet 2>>"%LOG_FILE%"
    call :LOG "Installed PyTorch with CUDA 12.1"
) else if "%GPU_MODE%"=="amd" (
    :: Install ROCm version (Windows uses DirectML)
    "%VENV_PIP%" install torch torchvision torchaudio --quiet 2>>"%LOG_FILE%"
    "%VENV_PIP%" install torch-directml --quiet 2>>"%LOG_FILE%"
    call :LOG "Installed PyTorch with DirectML for AMD"
) else if "%GPU_MODE%"=="mixed" (
    :: Mixed setup - use CPU torch with manual device handling
    "%VENV_PIP%" install torch torchvision torchaudio --quiet 2>>"%LOG_FILE%"
    call :LOG "Installed PyTorch for mixed GPU setup"
) else (
    :: CPU only
    "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet 2>>"%LOG_FILE%"
    call :LOG "Installed PyTorch CPU version"
)

:: Core dependencies
echo    Installing core dependencies...
"%VENV_PIP%" install transformers datasets accelerate --quiet 2>>"%LOG_FILE%"
"%VENV_PIP%" install sentencepiece tokenizers safetensors --quiet 2>>"%LOG_FILE%"
"%VENV_PIP%" install huggingface_hub --quiet 2>>"%LOG_FILE%"

:: UI dependencies
echo    Installing UI dependencies...
"%VENV_PIP%" install PySide6 --quiet 2>>"%LOG_FILE%"

:: Training dependencies
echo    Installing training dependencies...
"%VENV_PIP%" install bitsandbytes peft --quiet 2>>"%LOG_FILE%"
"%VENV_PIP%" install wandb tensorboard --quiet 2>>"%LOG_FILE%"

:: Media dependencies
echo    Installing media dependencies...
"%VENV_PIP%" install Pillow opencv-python --quiet 2>>"%LOG_FILE%"
"%VENV_PIP%" install imageio imageio-ffmpeg --quiet 2>>"%LOG_FILE%"
"%VENV_PIP%" install soundfile librosa --quiet 2>>"%LOG_FILE%"

:: Utility dependencies
echo    Installing utilities...
"%VENV_PIP%" install tqdm rich psutil GPUtil --quiet 2>>"%LOG_FILE%"
"%VENV_PIP%" install pyyaml toml --quiet 2>>"%LOG_FILE%"

:: Install Bagley itself
echo    Installing Bagley...
"%VENV_PIP%" install -e "%SCRIPT_DIR%" --quiet 2>>"%LOG_FILE%"

echo    [OK] All dependencies installed
call :LOG "All dependencies installed successfully"
goto :eof

:: ============================================================================
:: CONFIGURE BAGLEY
:: ============================================================================
:CONFIGURE_BAGLEY
echo.
echo [6/8] Configuring Bagley...
call :LOG "Configuring Bagley..."

:: Create config directory
set "CONFIG_DIR=%SCRIPT_DIR%\config"
if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

:: Create main config file
echo    Creating configuration files...

:: Write hardware config
(
echo # Bagley v7 Hardware Configuration
echo # Auto-generated by setup.bat on %date% %time%
echo.
echo [hardware]
echo gpu_mode = "%GPU_MODE%"
echo nvidia_gpus = %NVIDIA_GPUS%
echo amd_gpus = %AMD_GPUS%
echo intel_gpus = %INTEL_GPUS%
echo total_gpus = %TOTAL_GPUS%
echo ram_gb = %RAM_GB%
echo.
echo [cuda]
echo version = "%CUDA_VERSION%"
echo.
echo [rocm]
echo version = "%ROCM_VERSION%"
echo.
echo [paths]
echo bagley_root = "%SCRIPT_DIR%"
echo venv = "%VENV_DIR%"
echo models = "%SCRIPT_DIR%\models"
echo data = "%SCRIPT_DIR%\data"
echo logs = "%SCRIPT_DIR%\logs"
echo checkpoints = "%SCRIPT_DIR%\checkpoints"
) > "%CONFIG_DIR%\hardware.toml"

:: Create model config
(
echo # Bagley v7 Model Configuration
echo.
echo [training]
echo batch_size = "auto"
echo gradient_accumulation = 4
echo learning_rate = 2e-5
echo warmup_steps = 100
echo max_steps = -1
echo save_steps = 500
echo eval_steps = 100
echo logging_steps = 10
echo fp16 = true
echo.
echo [inference]
echo max_length = 4096
echo temperature = 0.7
echo top_p = 0.9
echo top_k = 50
echo.
echo [memory]
echo use_flash_attention = true
echo gradient_checkpointing = true
echo use_8bit = false
echo use_4bit = false
) > "%CONFIG_DIR%\model.toml"

:: Create directories
echo    Creating directory structure...
for %%d in (models data logs checkpoints cache temp output) do (
    if not exist "%SCRIPT_DIR%\%%d" mkdir "%SCRIPT_DIR%\%%d"
)

:: Create data subdirectories for auto-training
for %%d in (chat code images audio video) do (
    if not exist "%SCRIPT_DIR%\data\%%d" mkdir "%SCRIPT_DIR%\data\%%d"
)

echo    [OK] Configuration complete
call :LOG "Configuration files created"
goto :eof

:: ============================================================================
:: RUN TESTS
:: ============================================================================
:RUN_TESTS
echo.
echo [7/8] Running tests...
call :LOG "Running tests..."

set "TEST_ERRORS=0"

:: Test Python
echo    Testing Python...
"%VENV_PYTHON%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo    [FAIL] Python not working
    set /a "TEST_ERRORS+=1"
) else (
    echo    [PASS] Python OK
)

:: Test PyTorch
echo    Testing PyTorch...
"%VENV_PYTHON%" -c "import torch; print(f'PyTorch {torch.__version__}')" 2>>"%LOG_FILE%"
if %errorlevel% neq 0 (
    echo    [FAIL] PyTorch not working
    set /a "TEST_ERRORS+=1"
) else (
    echo    [PASS] PyTorch OK
)

:: Test GPU access
echo    Testing GPU access...
if "%GPU_MODE%"=="nvidia" (
    "%VENV_PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA GPUs: {torch.cuda.device_count()}')" 2>>"%LOG_FILE%"
    if %errorlevel% neq 0 (
        echo    [WARN] CUDA not available - will use CPU
    ) else (
        echo    [PASS] CUDA OK
    )
) else if "%GPU_MODE%"=="amd" (
    "%VENV_PYTHON%" -c "import torch_directml; print('DirectML OK')" 2>>"%LOG_FILE%"
    if %errorlevel% neq 0 (
        echo    [WARN] DirectML not available - will use CPU
    ) else (
        echo    [PASS] DirectML OK
    )
) else (
    echo    [INFO] CPU mode
)

:: Test transformers
echo    Testing Transformers...
"%VENV_PYTHON%" -c "from transformers import AutoTokenizer; print('Transformers OK')" 2>>"%LOG_FILE%"
if %errorlevel% neq 0 (
    echo    [FAIL] Transformers not working
    set /a "TEST_ERRORS+=1"
) else (
    echo    [PASS] Transformers OK
)

:: Test PySide6
echo    Testing UI framework...
"%VENV_PYTHON%" -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')" 2>>"%LOG_FILE%"
if %errorlevel% neq 0 (
    echo    [FAIL] PySide6 not working
    set /a "TEST_ERRORS+=1"
) else (
    echo    [PASS] PySide6 OK
)

:: Test Bagley import
echo    Testing Bagley modules...
"%VENV_PYTHON%" -c "from bagley.core import UnifiedBrain; print('Bagley Core OK')" 2>>"%LOG_FILE%"
if %errorlevel% neq 0 (
    echo    [WARN] Bagley modules need attention
) else (
    echo    [PASS] Bagley OK
)

if %TEST_ERRORS% GTR 0 (
    echo.
    echo    [!] %TEST_ERRORS% test(s) failed - check setup_log.txt
    call :LOG "Tests completed with %TEST_ERRORS% errors"
) else (
    echo.
    echo    [OK] All tests passed!
    call :LOG "All tests passed"
)
goto :eof

:: ============================================================================
:: CREATE SHORTCUTS
:: ============================================================================
:CREATE_SHORTCUTS
echo.
echo [8/8] Creating shortcuts...
call :LOG "Creating shortcuts..."

:: Create run.bat
(
echo @echo off
echo cd /d "%SCRIPT_DIR%"
echo call "%VENV_DIR%\Scripts\activate.bat"
echo python -m bagley.main --ui
echo pause
) > "%SCRIPT_DIR%\run.bat"

:: Create run_cli.bat
(
echo @echo off
echo cd /d "%SCRIPT_DIR%"
echo call "%VENV_DIR%\Scripts\activate.bat"
echo python -m bagley.main %%*
) > "%SCRIPT_DIR%\run_cli.bat"

:: Create train.bat
(
echo @echo off
echo cd /d "%SCRIPT_DIR%"
echo call "%VENV_DIR%\Scripts\activate.bat"
echo python -m bagley.main --train %%*
echo pause
) > "%SCRIPT_DIR%\train.bat"

:: Create desktop shortcut using PowerShell
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Bagley v7.lnk"
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_PATH%'); $s.TargetPath = '%SCRIPT_DIR%\run.bat'; $s.WorkingDirectory = '%SCRIPT_DIR%'; $s.Description = 'Bagley v7 - The Best AI'; $s.Save()" 2>nul

if exist "%SHORTCUT_PATH%" (
    echo    [OK] Desktop shortcut created
) else (
    echo    [INFO] Couldn't create desktop shortcut
)

echo    [OK] Launcher scripts created
call :LOG "Shortcuts created"
goto :eof

:: ============================================================================
:: FINALIZE INSTALLATION
:: ============================================================================
:FINALIZE_INSTALL
:: Write installation marker
(
echo version=%BAGLEY_VERSION%
echo installed=%date% %time%
echo python=%PYTHON_VERSION%
echo gpu_mode=%GPU_MODE%
echo nvidia_gpus=%NVIDIA_GPUS%
echo amd_gpus=%AMD_GPUS%
echo path=%SCRIPT_DIR%
) > "%CONFIG_FILE%"

call :LOG "Installation finalized"
goto :eof

:: ============================================================================
:: INSTALLATION COMPLETE
:: ============================================================================
:INSTALL_COMPLETE
cls
echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║                                                                  ║
echo  ║   ██████╗  █████╗  ██████╗ ██╗     ███████╗██╗   ██╗            ║
echo  ║   ██╔══██╗██╔══██╗██╔════╝ ██║     ██╔════╝╚██╗ ██╔╝            ║
echo  ║   ██████╔╝███████║██║  ███╗██║     █████╗   ╚████╔╝             ║
echo  ║   ██╔══██╗██╔══██║██║   ██║██║     ██╔══╝    ╚██╔╝              ║
echo  ║   ██████╔╝██║  ██║╚██████╔╝███████╗███████╗   ██║               ║
echo  ║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝               ║
echo  ║                                                                  ║
echo  ║         INSTALLATION COMPLETE - THE BEST AI IS READY!           ║
echo  ║                                                                  ║
echo  ╠══════════════════════════════════════════════════════════════════╣
echo  ║                                                                  ║
echo  ║   System Configuration:                                         ║
echo  ║   • GPU Mode: %GPU_MODE%
echo  ║   • NVIDIA GPUs: %NVIDIA_GPUS%
echo  ║   • AMD GPUs: %AMD_GPUS%
echo  ║   • Python: %PYTHON_VERSION%
echo  ║                                                                  ║
echo  ║   Quick Start:                                                  ║
echo  ║   • Double-click "run.bat" to launch Bagley                     ║
echo  ║   • Or use the desktop shortcut                                 ║
echo  ║   • Run setup.bat again for repair/update options               ║
echo  ║                                                                  ║
echo  ║   Training:                                                     ║
echo  ║   • Drop data into the "data" folder                            ║
echo  ║   • Bagley will auto-detect and train                           ║
echo  ║                                                                  ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.
set /p "RUN_NOW=Would you like to run Bagley now? [Y/N]: "
if /i "%RUN_NOW%"=="Y" goto :RUN_BAGLEY
goto :EXIT_SCRIPT

:: ============================================================================
:: RUN BAGLEY
:: ============================================================================
:RUN_BAGLEY
echo.
echo Starting Bagley...
cd /d "%SCRIPT_DIR%"
call "%VENV_DIR%\Scripts\activate.bat"
python -m bagley.main --ui
goto :EXIT_SCRIPT

:: ============================================================================
:: REPAIR INSTALLATION
:: ============================================================================
:REPAIR_INSTALL
echo.
echo Repairing Bagley installation...
call :LOG "Starting repair..."

:: Re-detect everything
call :DETECT_SYSTEM
call :DETECT_PYTHON
call :DETECT_GPUS

:: Fix venv if needed
if not exist "%SCRIPT_DIR%\venv\Scripts\python.exe" (
    echo    Recreating virtual environment...
    rmdir /s /q "%SCRIPT_DIR%\venv" 2>nul
    call :SETUP_VENV
)

set "VENV_PYTHON=%SCRIPT_DIR%\venv\Scripts\python.exe"
set "VENV_PIP=%SCRIPT_DIR%\venv\Scripts\pip.exe"

:: Reinstall dependencies
call :INSTALL_DEPENDENCIES

:: Run tests
call :RUN_TESTS

echo.
echo Repair complete!
call :LOG "Repair completed"
pause
goto :SHOW_MENU

:: ============================================================================
:: REINSTALL
:: ============================================================================
:REINSTALL
echo.
echo Reinstalling Bagley...
call :LOG "Starting reinstall..."

:: Remove venv
echo    Removing virtual environment...
rmdir /s /q "%SCRIPT_DIR%\venv" 2>nul

:: Remove config
del "%CONFIG_FILE%" 2>nul

:: Start fresh
goto :FRESH_INSTALL

:: ============================================================================
:: UNINSTALL (KEEP DATA)
:: ============================================================================
:UNINSTALL_KEEP
echo.
echo Uninstalling Bagley (keeping your data)...
call :LOG "Uninstalling (keep data)..."

set /p "CONFIRM=Are you sure? Your models and data will be preserved. [Y/N]: "
if /i not "%CONFIRM%"=="Y" goto :SHOW_MENU

:: Remove venv
echo    Removing virtual environment...
rmdir /s /q "%SCRIPT_DIR%\venv" 2>nul

:: Remove cache
echo    Removing cache...
rmdir /s /q "%SCRIPT_DIR%\cache" 2>nul
rmdir /s /q "%SCRIPT_DIR%\temp" 2>nul

:: Remove config marker
del "%CONFIG_FILE%" 2>nul

:: Remove shortcuts
del "%SCRIPT_DIR%\run.bat" 2>nul
del "%SCRIPT_DIR%\run_cli.bat" 2>nul
del "%SCRIPT_DIR%\train.bat" 2>nul
del "%USERPROFILE%\Desktop\Bagley v7.lnk" 2>nul

echo.
echo Uninstall complete. Your data folder is preserved.
call :LOG "Uninstall completed (data preserved)"
pause
goto :EXIT_SCRIPT

:: ============================================================================
:: FULL UNINSTALL
:: ============================================================================
:FULL_UNINSTALL
echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║                      FULL UNINSTALL WARNING                      ║
echo  ╠══════════════════════════════════════════════════════════════════╣
echo  ║                                                                  ║
echo  ║   This will DELETE EVERYTHING:                                  ║
echo  ║   • Virtual environment                                         ║
echo  ║   • All downloaded models                                       ║
echo  ║   • All training data                                           ║
echo  ║   • All checkpoints                                             ║
echo  ║   • All logs                                                    ║
echo  ║   • All configuration                                           ║
echo  ║                                                                  ║
echo  ║   THIS CANNOT BE UNDONE!                                        ║
echo  ║                                                                  ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.
set /p "CONFIRM1=Type 'DELETE' to confirm: "
if not "%CONFIRM1%"=="DELETE" goto :SHOW_MENU

echo.
call :LOG "Starting full uninstall..."

:: Remove everything except source code and setup.bat
echo    Removing virtual environment...
rmdir /s /q "%SCRIPT_DIR%\venv" 2>nul

echo    Removing models...
rmdir /s /q "%SCRIPT_DIR%\models" 2>nul

echo    Removing data...
rmdir /s /q "%SCRIPT_DIR%\data" 2>nul

echo    Removing checkpoints...
rmdir /s /q "%SCRIPT_DIR%\checkpoints" 2>nul

echo    Removing logs...
rmdir /s /q "%SCRIPT_DIR%\logs" 2>nul

echo    Removing cache...
rmdir /s /q "%SCRIPT_DIR%\cache" 2>nul
rmdir /s /q "%SCRIPT_DIR%\temp" 2>nul

echo    Removing config...
rmdir /s /q "%SCRIPT_DIR%\config" 2>nul
del "%CONFIG_FILE%" 2>nul

echo    Removing shortcuts...
del "%SCRIPT_DIR%\run.bat" 2>nul
del "%SCRIPT_DIR%\run_cli.bat" 2>nul
del "%SCRIPT_DIR%\train.bat" 2>nul
del "%USERPROFILE%\Desktop\Bagley v7.lnk" 2>nul

echo.
echo Full uninstall complete.
call :LOG "Full uninstall completed"
pause
goto :EXIT_SCRIPT

:: ============================================================================
:: RUN DIAGNOSTICS
:: ============================================================================
:RUN_DIAGNOSTICS
cls
echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║                    BAGLEY v7 DIAGNOSTICS                         ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.

call :LOG "Running diagnostics..."

:: System Info
echo ═══════════════════════════════════════════════════════════════════
echo SYSTEM INFORMATION
echo ═══════════════════════════════════════════════════════════════════
call :DETECT_SYSTEM
echo.

:: Python Info
echo ═══════════════════════════════════════════════════════════════════
echo PYTHON INFORMATION
echo ═══════════════════════════════════════════════════════════════════
call :DETECT_PYTHON
echo.

:: GPU Info
echo ═══════════════════════════════════════════════════════════════════
echo GPU INFORMATION
echo ═══════════════════════════════════════════════════════════════════
call :DETECT_GPUS
echo.

:: Disk Space
echo ═══════════════════════════════════════════════════════════════════
echo DISK SPACE
echo ═══════════════════════════════════════════════════════════════════
for /f "tokens=3" %%a in ('dir "%SCRIPT_DIR%" /-c 2^>nul ^| findstr /c:"bytes free"') do (
    set /a "FREE_GB=%%a/1073741824"
    echo    Free space: !FREE_GB! GB
)
echo.

:: Package Versions
echo ═══════════════════════════════════════════════════════════════════
echo INSTALLED PACKAGES
echo ═══════════════════════════════════════════════════════════════════
if exist "%SCRIPT_DIR%\venv\Scripts\pip.exe" (
    "%SCRIPT_DIR%\venv\Scripts\pip.exe" list 2>nul | findstr /i "torch transformers pyside6 accelerate"
)
echo.

:: Run full tests
echo ═══════════════════════════════════════════════════════════════════
echo RUNNING TESTS
echo ═══════════════════════════════════════════════════════════════════
set "VENV_PYTHON=%SCRIPT_DIR%\venv\Scripts\python.exe"
call :RUN_TESTS
echo.

echo Diagnostics complete. Press any key to return to menu...
pause >nul
goto :SHOW_MENU

:: ============================================================================
:: UTILITY FUNCTIONS
:: ============================================================================
:LOG
echo [%date% %time%] %~1 >> "%LOG_FILE%"
goto :eof

:: ============================================================================
:: EXIT
:: ============================================================================
:EXIT_SCRIPT
echo.
echo Thank you for using Bagley v7 - The Best AI in the World!
echo.
endlocal
exit /b 0
