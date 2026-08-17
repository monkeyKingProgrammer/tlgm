@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo TLGM 120M 1024-context Knowledge 3B Pipeline
echo Folder: %CD%
echo.
echo This does NOT overwrite existing 1024ctx checkpoints.
echo Final output:
echo   checkpoints\tlgm_120m_1024ctx_knowledge3b_sft_final.pth
echo.
echo Data mix:
echo   40%% SmolLM FineWeb-Edu-Dedup
echo   30%% SmolLM Cosmopedia-v2
echo   30%% Wikimedia Wikipedia 20231101.en
echo.
echo Stages:
echo   1. Stream/download/tokenize 3B knowledge tokens if missing
echo   2. Continue pretraining from 1024ctx pretrain checkpoint
echo   3. Broad SFT on sft_chat_mix
echo   4. Repair SFT on sft_chat_repair
echo   5. Polish SFT
echo   6. Run prompt test
echo.
echo Windows note:
echo   Training configs use num_workers=0 to avoid multiprocessing
echo   MemoryError when spawning workers for large datasets.
echo ============================================================

if not exist data\processed\knowledge3b_tokens.bin (
  python scripts\prepare_knowledge_pretrain_tokens.py --target_tokens 3000000000 --output_bin data\processed\knowledge3b_tokens.bin --meta data\processed\knowledge3b_tokens_meta.json
  if errorlevel 1 goto fail
) else (
  echo Found existing data\processed\knowledge3b_tokens.bin, skipping tokenization.
)

python scripts\train.py --config configs\pretrain_tlgm_120m_knowledge3b.yaml
if errorlevel 1 goto fail

python scripts\train.py --config configs\sft_tlgm_120m_knowledge3b_chatmix.yaml
if errorlevel 1 goto fail

python scripts\train.py --config configs\sft_tlgm_120m_knowledge3b_repair.yaml
if errorlevel 1 goto fail

if not exist data\processed\sft_polish.jsonl (
  python scripts\prepare_polish_sft.py --output data\processed\sft_polish.jsonl --repeats 500
  if errorlevel 1 goto fail
)

python scripts\train.py --config configs\sft_tlgm_120m_knowledge3b_polish.yaml
if errorlevel 1 goto fail

python scripts\test_prompts.py --config configs\sft_tlgm_120m_knowledge3b_polish.yaml --checkpoint checkpoints\tlgm_120m_1024ctx_knowledge3b_sft_final.pth --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 160
if errorlevel 1 goto fail

echo ============================================================
echo COMPLETE
echo Final knowledge-improved checkpoint:
echo   checkpoints\tlgm_120m_1024ctx_knowledge3b_sft_final.pth
echo.
echo Chat command:
echo   python chat_cli.py --config configs\sft_tlgm_120m_knowledge3b_polish.yaml --checkpoint checkpoints\tlgm_120m_1024ctx_knowledge3b_sft_final.pth --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 160
echo ============================================================
pause
exit /b 0

:fail
echo ============================================================
echo FAILED. Check the last error above.
echo You can rerun this BAT; training stages auto-resume from their own checkpoints.
echo If tokenization failed partway through, delete data\processed\knowledge3b_tokens.bin before rerunning.
echo ============================================================
pause
exit /b 1
