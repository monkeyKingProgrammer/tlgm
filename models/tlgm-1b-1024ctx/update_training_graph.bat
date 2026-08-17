@echo off
setlocal
cd /d "%~dp0"
if not defined TLGM_SSH_KEY (
  echo Set TLGM_SSH_KEY to your PuTTY private-key path.
  exit /b 2
)
if not defined TLGM_REMOTE (
  echo Set TLGM_REMOTE to user@hostname.
  exit /b 2
)
if not defined TLGM_REMOTE_DIR set "TLGM_REMOTE_DIR=/home/user/minimind/tlgm_1b_1024ctx"
if not defined PUTTY_DIR set "PUTTY_DIR=C:\Program Files\PuTTY"
set "KEY=%TLGM_SSH_KEY%"
set "REMOTE=%TLGM_REMOTE%"
set "REMOTE_DIR=%TLGM_REMOTE_DIR%"

"%PUTTY_DIR%\plink.exe" -batch -no-antispoof -i "%KEY%" "%REMOTE%" "cd %REMOTE_DIR% && /home/user/venvs/tlgm/bin/python scripts/plot_training_progress.py"
if errorlevel 1 exit /b 1

if not exist outputs mkdir outputs
"%PUTTY_DIR%\pscp.exe" -batch -i "%KEY%" "%REMOTE%:%REMOTE_DIR%/outputs/training_progress.png" "outputs\training_progress.png"
if errorlevel 1 exit /b 1
"%PUTTY_DIR%\pscp.exe" -batch -i "%KEY%" "%REMOTE%:%REMOTE_DIR%/outputs/training_progress.json" "outputs\training_progress.json"
if errorlevel 1 exit /b 1

echo.
echo Updated: %CD%\outputs\training_progress.png
start "" "%CD%\outputs\training_progress.png"
