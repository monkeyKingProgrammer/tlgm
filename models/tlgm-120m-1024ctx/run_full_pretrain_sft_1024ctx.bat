@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo TLGM 120M 1024-context full training pipeline
echo Folder: %CD%
echo Stages:
echo   1. Pretrain from random init, auto-resume if checkpoint exists
echo   2. Masked SFT from pretrain checkpoint
echo   3. Repair masked SFT
echo   4. Prepare polish SFT data
echo   5. Polish SFT
echo   6. Run simple prompt test
echo ============================================================

python scripts\train.py --config configs\pretrain_tlgm_120m_2b.yaml
if errorlevel 1 goto fail

python scripts\train.py --config configs\sft_tlgm_120m_masked.yaml
if errorlevel 1 goto fail

python scripts\train.py --config configs\sft_tlgm_120m_repair_masked.yaml
if errorlevel 1 goto fail

if not exist data\processed\sft_polish.jsonl (
  python scripts\prepare_polish_sft.py --output data\processed\sft_polish.jsonl --repeats 500
  if errorlevel 1 goto fail
)

python scripts\train.py --config configs\sft_tlgm_120m_polish.yaml
if errorlevel 1 goto fail

python scripts\test_prompts.py --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 120
if errorlevel 1 goto fail

echo ============================================================
echo COMPLETE
echo Final checkpoint:
echo   checkpoints\tlgm_120m_1024ctx_sft_polished.pth
echo Chat command:
echo   python chat_cli.py --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 120
echo ============================================================
pause
exit /b 0

:fail
echo ============================================================
echo FAILED. Check the last error above.
echo You can rerun this BAT; completed checkpoints will auto-resume.
echo ============================================================
pause
exit /b 1
