@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "MAIN_PY=%SCRIPT_DIR%main.py"
set "FOUND_RUNTIME="
set "LAUNCH_EXITCODE=1"
set "PROBE_ONLY=0"
if /i "%~1"=="--probe-only" (
    set "PROBE_ONLY=1"
    shift
)
set "APP_ARGS=%*"

if not exist "%MAIN_PY%" (
    echo [PyKaboo] main.py was not found next to this launcher.
    exit /b 1
)

if defined PYKABOO_PYTHON call :try_interpreter "%PYKABOO_PYTHON%" "PYKABOO_PYTHON"
if defined FOUND_RUNTIME exit /b !LAUNCH_EXITCODE!
if defined CONDA_PREFIX call :try_interpreter "%CONDA_PREFIX%\python.exe" "active conda environment"
if defined FOUND_RUNTIME exit /b !LAUNCH_EXITCODE!

for %%P in (
    "%USERPROFILE%\.conda\envs\CamApp\python.exe"
    "%ProgramData%\anaconda3\envs\CamApp\python.exe"
    "%USERPROFILE%\anaconda3\envs\CamApp\python.exe"
) do (
    call :try_interpreter "%%~fP" "CamApp environment"
    if defined FOUND_RUNTIME exit /b !LAUNCH_EXITCODE!
)

call :try_interpreter "python" "PATH python"
if defined FOUND_RUNTIME exit /b !LAUNCH_EXITCODE!

echo [PyKaboo] No usable Python runtime was found.
echo [PyKaboo] Expected a Python that can import PySide6, cv2, and PySpin.
echo [PyKaboo] Create or activate the Conda environment from environment.yaml, then rerun this launcher.
exit /b 1

:try_interpreter
set "CANDIDATE=%~1"
set "SOURCE=%~2"
if "%CANDIDATE%"=="" exit /b 0

if /i not "%CANDIDATE%"=="python" (
    if not exist "%CANDIDATE%" exit /b 0
)

"%CANDIDATE%" -c "import PySide6, cv2, PySpin" >nul 2>nul
if errorlevel 1 exit /b 0

echo [PyKaboo] Launching with %SOURCE%: %CANDIDATE%
set "FOUND_RUNTIME=1"
if "%PROBE_ONLY%"=="1" (
    set "LAUNCH_EXITCODE=0"
    exit /b 0
)
"%CANDIDATE%" "%MAIN_PY%" %APP_ARGS%
set "LAUNCH_EXITCODE=%errorlevel%"
exit /b 0
