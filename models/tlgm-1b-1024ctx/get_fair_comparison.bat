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

echo ============================================================
echo TLGM fair perplexity comparison status
echo ============================================================
"%PUTTY_DIR%\plink.exe" -batch -no-antispoof -i "%KEY%" "%REMOTE%" "systemctl status tlgm1b-fair-ppl.service --no-pager -l | head -25; test -f %REMOTE_DIR%/outputs/fair_perplexity_1b/FAIR_PERPLEXITY_REPORT.md"
if errorlevel 1 (
  echo.
  echo The final report is not available yet. Training or comparison is still pending.
  exit /b 1
)

if not exist outputs\fair_perplexity_1b mkdir outputs\fair_perplexity_1b
for %%F in (FAIR_PERPLEXITY_REPORT.md results.json fair_comparison.png) do (
  "%PUTTY_DIR%\pscp.exe" -batch -i "%KEY%" "%REMOTE%:%REMOTE_DIR%/outputs/fair_perplexity_1b/%%F" "outputs\fair_perplexity_1b\%%F"
  if errorlevel 1 exit /b 1
)

echo.
echo Report downloaded to:
echo %CD%\outputs\fair_perplexity_1b\FAIR_PERPLEXITY_REPORT.md
start "" "%CD%\outputs\fair_perplexity_1b\fair_comparison.png"
