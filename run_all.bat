@echo off
setlocal EnableExtensions

echo Starting Sortiskeos Project Services...
echo.

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%" >nul

set "API_PYTHON="
if exist "venv\Scripts\python.exe" set "API_PYTHON=%ROOT_DIR%venv\Scripts\python.exe"
if not defined API_PYTHON if exist "ml\venv\Scripts\python.exe" set "API_PYTHON=%ROOT_DIR%ml\venv\Scripts\python.exe"
if not defined API_PYTHON set "API_PYTHON=python"

set "AGENT_COMMAND="
if exist "ml\dist\sortiskeos_core.exe" (
    set "AGENT_COMMAND=%ROOT_DIR%ml\dist\sortiskeos_core.exe"
) else (
    set "AGENT_COMMAND=%API_PYTHON% %ROOT_DIR%ml\sortiskeos_core.py"
)

set "HAS_ERRORS=0"

echo [Preflight] Checking local prerequisites...

docker compose version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Compose is not available.
    set "HAS_ERRORS=1"
)

"%API_PYTHON%" -c "import fastapi, uvicorn, elasticsearch, aiohttp, dotenv" >nul 2>&1
if errorlevel 1 (
    echo ERROR: API Python dependencies are missing for %API_PYTHON%.
    echo        Run: python -m pip install -r api\requirements.txt
    set "HAS_ERRORS=1"
)

"%API_PYTHON%" -c "import pandas, numpy, sklearn, elasticsearch, psutil, win32evtlog, win32evtlogutil" >nul 2>&1
if errorlevel 1 (
    echo ERROR: ML agent Python dependencies are missing for %API_PYTHON%.
    echo        Run: python -m pip install -r ml\requirements.txt psutil
    set "HAS_ERRORS=1"
)

if not exist "ui\node_modules" (
    echo ERROR: UI dependencies are missing.
    echo        Run: cd ui ^&^& npm install
    set "HAS_ERRORS=1"
)

where node >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not available on PATH.
    set "HAS_ERRORS=1"
)

if "%HAS_ERRORS%"=="1" (
    echo.
    echo Startup aborted because this laptop is missing required local setup.
    echo Fix the errors above and run this file again.
    popd >nul
    pause
    exit /b 1
)

echo.
echo [1/4] Starting ELK Stack (Docker Compose)...
pushd "elk" >nul
start "Sortiskeos - ELK Stack" cmd /k "docker compose up"
popd >nul

echo.
echo [2/4] Starting Sortiskeos Edge Agent...
pushd "ml" >nul
start "Sortiskeos - Agent Background Process" cmd /k "%AGENT_COMMAND%"
popd >nul

echo.
echo [3/4] Starting Sortiskeos API (FastAPI)...
pushd "api" >nul
start "Sortiskeos - API" cmd /k "%API_PYTHON% -m uvicorn main:app --host 0.0.0.0 --port 8000"
popd >nul

echo.
echo [4/4] Starting Sortiskeos UI (React)...
pushd "ui" >nul
start "Sortiskeos - UI" cmd /k "npm.cmd start"
popd >nul

echo.
echo =========================================================================
echo All 4 services have been actively launched. Keep all terminals open!
echo.
echo Dashboard UI: http://localhost:3000
echo Backend API:  http://localhost:8000
echo Kibana ELK:   http://localhost:5601
echo.
echo Background Agent: sortiskeos_core is active and reading telemetry.
echo =========================================================================

popd >nul
pause
