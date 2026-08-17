@echo off
setlocal

cd /d C:\Users\ADMIN\minimind\tlgm_120m

echo === TLGM masked broad SFT ===
python scripts\train.py --config configs\sft_tlgm_120m_masked.yaml
if errorlevel 1 (
  echo.
  echo ERROR: masked broad SFT failed.
  pause
  exit /b 1
)

echo.
echo === TLGM masked repair SFT ===
python scripts\train.py --config configs\sft_tlgm_120m_repair_masked.yaml
if errorlevel 1 (
  echo.
  echo ERROR: masked repair SFT failed.
  pause
  exit /b 1
)

echo.
echo === TLGM prompt test ===
python scripts\test_prompts.py --checkpoint checkpoints\tlgm_120m_sft_final.pth
if errorlevel 1 (
  echo.
  echo ERROR: prompt test failed.
  pause
  exit /b 1
)

echo.
echo Done. Final checkpoint: checkpoints\tlgm_120m_sft_final.pth
pause
