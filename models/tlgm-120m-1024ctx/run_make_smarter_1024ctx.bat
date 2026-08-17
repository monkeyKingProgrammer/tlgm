@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo TLGM 120M 1024-context continued SFT improvement
echo Folder: %CD%
echo.
echo This does NOT overwrite the current polished checkpoint.
echo It creates:
echo   checkpoints\tlgm_120m_1024ctx_sft_smarter.pth
echo.
echo Stages:
echo   1. Continue broad masked SFT on sft_chat_mix, 8000 steps
echo   2. Repair SFT on sft_chat_repair, 1500 steps
echo   3. Short polish SFT, 300 steps
echo   4. Run prompt test
echo ============================================================

python scripts\train.py --config configs\sft_tlgm_120m_continue_chatmix.yaml
if errorlevel 1 goto fail

python scripts\train.py --config configs\sft_tlgm_120m_continue_repair.yaml
if errorlevel 1 goto fail

if not exist data\processed\sft_polish.jsonl (
  python scripts\prepare_polish_sft.py --output data\processed\sft_polish.jsonl --repeats 500
  if errorlevel 1 goto fail
)

python scripts\train.py --config configs\sft_tlgm_120m_continue_polish.yaml
if errorlevel 1 goto fail

python scripts\test_prompts.py --config configs\sft_tlgm_120m_continue_polish.yaml --checkpoint checkpoints\tlgm_120m_1024ctx_sft_smarter.pth --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 160
if errorlevel 1 goto fail

echo ============================================================
echo COMPLETE
echo Final improved checkpoint:
echo   checkpoints\tlgm_120m_1024ctx_sft_smarter.pth
echo.
echo Chat command:
echo   python chat_cli.py --config configs\sft_tlgm_120m_continue_polish.yaml --checkpoint checkpoints\tlgm_120m_1024ctx_sft_smarter.pth --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 160
echo ============================================================
pause
exit /b 0

:fail
echo ============================================================
echo FAILED. Check the last error above.
echo You can rerun this BAT; each stage auto-resumes from its checkpoint.
echo ============================================================
pause
exit /b 1
