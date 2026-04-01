@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ================================================
echo SortiskeOS Auto Trigger Task Registration
echo Run this script as Administrator.
echo This will recreate the Task Scheduler task:
echo   SortiskeOS-AutoTrigger
echo ================================================
echo.

net session >nul 2>&1
if not "%errorlevel%"=="0" (
    echo ERROR: Please right-click this file and choose Run as administrator.
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%") do set "ML_DIR=%%~fI"
set "TASK_NAME=SortiskeOS-AutoTrigger"
set "XML_FILE=%TEMP%\SortiskeOS-AutoTrigger.xml"
set "PYTHON_EXE="

for /f "delims=" %%I in ('where python 2^>nul') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
)

if not defined PYTHON_EXE (
    echo ERROR: Could not resolve python.exe using WHERE PYTHON.
    exit /b 1
)

set "TASK_COMMAND=%PYTHON_EXE%"
set "TASK_ARGS=%ML_DIR%startup_trigger.py"

schtasks /Query /TN "%TASK_NAME%" >nul 2>&1
if "%errorlevel%"=="0" (
    echo Existing task found. Deleting it first...
    schtasks /Delete /TN "%TASK_NAME%" /F >nul
)

> "%XML_FILE%" echo ^<?xml version="1.0"?^>
>> "%XML_FILE%" echo ^<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^>
>> "%XML_FILE%" echo   ^<RegistrationInfo^>
>> "%XML_FILE%" echo     ^<Author^>SortiskeOS^</Author^>
>> "%XML_FILE%" echo     ^<Description^>Runs startup_trigger.py at startup and after Kernel-Power Event ID 41.^</Description^>
>> "%XML_FILE%" echo   ^</RegistrationInfo^>
>> "%XML_FILE%" echo   ^<Triggers^>
>> "%XML_FILE%" echo     ^<BootTrigger^>
>> "%XML_FILE%" echo       ^<Enabled^>true^</Enabled^>
>> "%XML_FILE%" echo     ^</BootTrigger^>
>> "%XML_FILE%" echo     ^<EventTrigger^>
>> "%XML_FILE%" echo       ^<Enabled^>true^</Enabled^>
>> "%XML_FILE%" echo       ^<Subscription^>^&lt;QueryList^&gt;^&lt;Query Id="0" Path="System"^&gt;^&lt;Select Path="System"^&gt;*[System[Provider[@Name='Microsoft-Windows-Kernel-Power'] and (EventID=41)]]^&lt;/Select^&gt;^&lt;/Query^&gt;^&lt;/QueryList^&gt;^</Subscription^>
>> "%XML_FILE%" echo     ^</EventTrigger^>
>> "%XML_FILE%" echo   ^</Triggers^>
>> "%XML_FILE%" echo   ^<Principals^>
>> "%XML_FILE%" echo     ^<Principal id="Author"^>
>> "%XML_FILE%" echo       ^<UserId^>SYSTEM^</UserId^>
>> "%XML_FILE%" echo       ^<LogonType^>ServiceAccount^</LogonType^>
>> "%XML_FILE%" echo       ^<RunLevel^>HighestAvailable^</RunLevel^>
>> "%XML_FILE%" echo     ^</Principal^>
>> "%XML_FILE%" echo   ^</Principals^>
>> "%XML_FILE%" echo   ^<Settings^>
>> "%XML_FILE%" echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^>
>> "%XML_FILE%" echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^>
>> "%XML_FILE%" echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^>
>> "%XML_FILE%" echo     ^<AllowHardTerminate^>true^</AllowHardTerminate^>
>> "%XML_FILE%" echo     ^<StartWhenAvailable^>true^</StartWhenAvailable^>
>> "%XML_FILE%" echo     ^<RunOnlyIfNetworkAvailable^>false^</RunOnlyIfNetworkAvailable^>
>> "%XML_FILE%" echo     ^<IdleSettings^>
>> "%XML_FILE%" echo       ^<StopOnIdleEnd^>false^</StopOnIdleEnd^>
>> "%XML_FILE%" echo       ^<RestartOnIdle^>false^</RestartOnIdle^>
>> "%XML_FILE%" echo     ^</IdleSettings^>
>> "%XML_FILE%" echo     ^<AllowStartOnDemand^>true^</AllowStartOnDemand^>
>> "%XML_FILE%" echo     ^<Enabled^>true^</Enabled^>
>> "%XML_FILE%" echo     ^<Hidden^>false^</Hidden^>
>> "%XML_FILE%" echo     ^<RunOnlyIfIdle^>false^</RunOnlyIfIdle^>
>> "%XML_FILE%" echo     ^<WakeToRun^>false^</WakeToRun^>
>> "%XML_FILE%" echo     ^<ExecutionTimeLimit^>PT1H^</ExecutionTimeLimit^>
>> "%XML_FILE%" echo     ^<Priority^>7^</Priority^>
>> "%XML_FILE%" echo   ^</Settings^>
>> "%XML_FILE%" echo   ^<Actions Context="Author"^>
>> "%XML_FILE%" echo     ^<Exec^>
>> "%XML_FILE%" echo       ^<Command^>%TASK_COMMAND%^</Command^>
>> "%XML_FILE%" echo       ^<Arguments^>"%TASK_ARGS%"^</Arguments^>
>> "%XML_FILE%" echo       ^<WorkingDirectory^>%ML_DIR:~0,-1%^</WorkingDirectory^>
>> "%XML_FILE%" echo     ^</Exec^>
>> "%XML_FILE%" echo   ^</Actions^>
>> "%XML_FILE%" echo ^</Task^>

echo Registering task...
schtasks /Create /TN "%TASK_NAME%" /XML "%XML_FILE%" /RU SYSTEM /F
set "RESULT=%errorlevel%"

del "%XML_FILE%" >nul 2>&1

if not "%RESULT%"=="0" (
    echo ERROR: Task registration failed.
    exit /b %RESULT%
)

echo.
echo Task registered successfully.
echo Open Task Scheduler and confirm "%TASK_NAME%" appears.
exit /b 0
