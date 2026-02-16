@echo off
REM ================================================================
REM  GOLDILOCKS ANT v4 — Single-Level L0 Hash LCX
REM  Architecture: LOCKED (do not change these params)
REM ================================================================
REM
REM  D=6180  depth=12  ring=62  seq=62  bits=8  beings=1
REM  LCX: single L0, hash-bucketed, key_dim=618, top_k=6
REM  Starts INFANT (Alpha): tt=0, batch=10, LCX=OFF, AMP=ON
REM  Advance via Grafana controls or controls.json
REM
REM  EFFORT TIERS (via Grafana or controls.json):
REM  ------------------------------------------------------------------
REM  Alpha(Reflex):     tt=0  batch=10  lcx=OFF  <-- START (INFANT)
REM  Beta(Recall):      tt=1  batch=10  lcx=ON   (CHILD)
REM  Gamma(Reason):     tt=2  batch=5   lcx=ON
REM  Delta(Depth):      tt=4  batch=3   lcx=ON
REM  Epsilon(Emerge):   tt=8  batch=2   lcx=ON
REM  Zeta(Zenith):      tt=16 batch=1   lcx=ON
REM  ------------------------------------------------------------------
REM  PROGRESSIVE STAGES (via Advance Phase button):
REM  INFANT(no LCX) > CHILD(L0=2K) > TEEN(L0=20K) > RECALL(L0=100K)
REM  > DEPTH(L0=200K) > SAGE(L0=200K,tt=4)
REM  ------------------------------------------------------------------
REM  Data mix: controlled from Grafana Data Sources panel
REM  Delete logs/swarm/controls.json before launch for fresh defaults.
REM ================================================================

cd /d "S:\AI\work\VRAXION_DEV\Diamond Code"

REM Keep controls.json if it exists (preserves data_weights, LR, effort tier)
REM Delete it manually if you want fresh defaults.

REM Kill any stale control panel on :7777
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7777.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
REM Auto-start control panel (Grafana data mix, training controls)
echo Starting control panel on :7777...
start "CTRL_PANEL" /MIN python -u tools/control_panel.py
timeout /t 2 /nobreak >nul

python -u test_swarm_config.py ^
    --embedding 6180 ^
    --depth 12 ^
    --num_beings 1 ^
    --num_bits 8 ^
    --batch_size 10 ^
    --seq_len 62 ^
    --memory_size 62 ^
    --think_ticks 1 ^
    --attention_radius 6 ^
    --lcx_mode hash ^
    --lcx_num_levels 1 ^
    --lcx_level_slots "2000" ^
    --lcx_key_dim 618 ^
    --lcx_top_k 6 ^
    --amp ^
    --data_dir "data/traindat/" ^
    --checkpoint_dir "checkpoints/goldilocks_v4" ^
    --checkpoint_every 25 ^
    --eval_every 5 ^
    --steps 1000000 ^
    --lr 0.0003 ^
    --warmup_steps 100 ^
    --lr_min 1e-5 ^
    --controls_every 1 ^
    --effort Alpha ^
    --start_lcx_off ^
    --resume "checkpoints/goldilocks_v4/checkpoint_step_5000.pt"

REM Kill control panel when training ends
taskkill /F /FI "WINDOWTITLE eq CTRL_PANEL" >nul 2>&1
echo Training ended. Control panel stopped.
pause
