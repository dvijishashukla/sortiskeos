@echo off
echo Starting Sortiskeos Project Services...

echo.
echo [1/4] Starting ELK Stack (Docker Compose)...
cd elk
start "Sortiskeos - ELK Stack" cmd /k "docker-compose up"
cd ..

echo.
echo [2/4] Starting Sortiskeos Edge Agent...
cd ml\dist
start "Sortiskeos - Agent Background Process" cmd /k "sortiskeos_core.exe"
cd ..\..

echo.
echo [3/4] Starting Sortiskeos API (FastAPI)...
cd api
start "Sortiskeos - API" cmd /k "..\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
cd ..

echo.
echo [4/4] Starting Sortiskeos UI (React)...
cd ui
start "Sortiskeos - UI" cmd /k "npm start"
cd ..

echo.
echo =========================================================================
echo All 4 services have been actively launched. Keep all terminals open!
echo.
echo Dashboard UI: http://localhost:3000
echo Backend API:  http://localhost:8000
echo Kibana ELK:   http://localhost:5601
echo.
echo Background Agent: "sortiskeos_core.exe" is active and reading telemetry.
echo =========================================================================
pause
