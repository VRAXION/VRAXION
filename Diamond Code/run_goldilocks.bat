@echo off
REM ================================================================
REM  GOLDILOCKS ANT v4 — Single-Level L0 Hash LCX
REM  Architecture: LOCKED (do not change these params)
REM ================================================================
REM
REM  D=6180  depth=2  ring=192  seq=192  bits=8  beings=1
REM  LCX: single L0, hash-bucketed, key_dim=618, top_k=2
REM  LR=1e-3 (probe-validated: 1e-4 kills echo256, see probe_lr_ablation.py)
REM  Starts INFANT (Alpha): tt=0, batch=10, LCX=OFF, AMP=ON
REM  Brain learns pattern first, then promote to Beta to enable LCX
REM  (LCX from random init adds -12.8% noise, see probe_real_model.py)
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
REM  ------------------------------------------------------------------
REM  CHECKPOINTS: Auto-resume ON by default (resumes from latest draft).
REM  Add --fresh to python call below to start from random init.
REM  Golden saves in checkpoints/curriculum_v2/golden/
REM  To resume from specific: --resume "checkpoints/.../golden/xxx.pt"
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
    --depth 2 ^
    --num_beings 1 ^
    --num_bits 8 ^
    --batch_size 5 ^
    --seq_len 192 ^
    --memory_size 192 ^
    --think_ticks 1 ^
    --attention_radius 8 ^
    --lcx_mode hash ^
    --lcx_num_levels 1 ^
    --lcx_level_slots "2000" ^
    --lcx_key_dim 618 ^
    --lcx_top_k 2 ^
    --num_pointers 1 ^
    --amp ^
    --binary-bits ^
    --data_dir "data/traindat/" ^
    --data_weights "{\"fineweb_edu.traindat\":1}" ^
    --checkpoint_dir "checkpoints/curriculum_v2" ^
    --checkpoint_every 25 ^
    --eval_every 5 ^
    --steps 1000000 ^
    --lr 0.0004 ^
    --warmup_steps 50 ^
    --lr_min 1e-5 ^
    --controls_every 1 ^
    --zoom_gate_init -4.0 ^
    --effort Beta

REM Kill control panel when training ends
taskkill /F /FI "WINDOWTITLE eq CTRL_PANEL" >nul 2>&1
echo Training ended. Control panel stopped.
pause
