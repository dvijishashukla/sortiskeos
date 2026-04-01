@echo off
echo Stopping all Sortiskeos Project Services...

echo.
echo [1/4] Killing Sortiskeos Background Agent...
taskkill /FI "WINDOWTITLE eq Sortiskeos - Agent Background Process" /T /F >nul 2>&1
taskkill /IM sortiskeos_core.exe /F >nul 2>&1

echo.
echo [2/4] Killing Sortiskeos API...
taskkill /FI "WINDOWTITLE eq Sortiskeos - API" /T /F >nul 2>&1

echo.
echo [3/4] Killing Sortiskeos UI...
taskkill /FI "WINDOWTITLE eq Sortiskeos - UI" /T /F >nul 2>&1

echo.
echo [4/4] Stopping ELK Stack (Docker Compose)...
cd elk
docker-compose down
echo (Docker containers safely stopped)
cd ..

echo.
echo =========================================================================
echo All Sortiskeos services and background processes have been shut down cleanly.
echo =========================================================================
pause
