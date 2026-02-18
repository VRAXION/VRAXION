@echo off
REM ================================================================
REM  D=128 MINI GOLDILOCKS — Proven Architecture Validation
REM  Architecture: D=128 depth=4 ring=16 seq=16 bits=8 beings=1
REM  LCX: single L0, hash-bucketed, key_dim=32, top_k=3, 100 slots
REM  CPU + FP64 full precision, batch=32, binary-bits mode
REM  Proven: 73%% bit_acc in 200 steps at tt=1 (binary bits)
REM ================================================================

cd /d "S:\AI\work\VRAXION_DEV\Diamond Code"

REM Kill any stale control panel on :7777
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7777.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
REM Auto-start control panel
echo Starting control panel on :7777...
start "CTRL_PANEL" /MIN python -u tools/control_panel.py
timeout /t 2 /nobreak >nul

python -u test_swarm_config.py ^
    --embedding 128 ^
    --depth 4 ^
    --binary-bits ^
    --num_beings 1 ^
    --num_bits 8 ^
    --batch_size 32 ^
    --seq_len 16 ^
    --memory_size 16 ^
    --think_ticks 1 ^
    --attention_radius 3 ^
    --lcx_mode hash ^
    --lcx_num_levels 1 ^
    --lcx_level_slots "100" ^
    --lcx_key_dim 32 ^
    --lcx_top_k 3 ^
    --num_pointers 1 ^
    --fp64 ^
    --device cpu ^
    --data_dir "data/traindat/" ^
    --checkpoint_dir "checkpoints/d128_goldilocks" ^
    --checkpoint_every 100 ^
    --eval_every 10 ^
    --steps 1000000 ^
    --lr 0.0003 ^
    --warmup_steps 100 ^
    --lr_min 1e-5 ^
    --controls_every 1 ^
    --effort Beta

REM Kill control panel when training ends
taskkill /F /FI "WINDOWTITLE eq CTRL_PANEL" >nul 2>&1
echo Training ended. Control panel stopped.
pause
