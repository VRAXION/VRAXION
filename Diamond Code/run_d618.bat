@echo off
REM ================================================================
REM  DIAMOND EDGE — D=618, seq=6, Golden Ratio Fractal (one level down)
REM  Architecture: LOCKED (same as v3.2.000, scaled to D=618)
REM ================================================================
REM
REM  D=618  depth=12  ring=6  seq=6  bits=8  beings=1
REM  LCX: single L0, hash-bucketed, 2000 slots (grow on plateau), key_dim=62, top_k=6
REM  Bottleneck: 618→62→C19→62→C19→618
REM  Starts Beta: tt=1, batch=500, LCX=ON, CPU (3x faster than GPU at D=618)
REM
REM  Params: ~4.35M   RAM: ~1.6GB   Step: ~7.7s   Throughput: ~65 sps
REM
REM  EFFORT TIERS (via Grafana or controls.json):
REM  ------------------------------------------------------------------
REM  Alpha(Reflex):     tt=0  batch=500  lcx=OFF
REM  Beta(Recall):      tt=1  batch=500  lcx=ON   <-- START
REM  Gamma(Reason):     tt=2  batch=500  lcx=ON
REM  Delta(Depth):      tt=4  batch=500  lcx=ON
REM  Epsilon(Emerge):   tt=8  batch=500  lcx=ON
REM  Zeta(Zenith):      tt=16 batch=250  lcx=ON
REM  ------------------------------------------------------------------
REM  Data mix: controlled from Grafana Data Sources panel
REM  Delete logs/swarm/controls.json before launch for fresh defaults.
REM ================================================================

cd /d "S:\AI\work\VRAXION_DEV\Diamond Code"

REM Kill any stale control panel on :7777
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7777.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
REM Auto-start control panel (Grafana data mix, training controls)
echo Starting control panel on :7777...
start "CTRL_PANEL" /MIN python -u tools/control_panel.py
timeout /t 2 /nobreak >nul

set OMP_NUM_THREADS=20
set MKL_NUM_THREADS=20

python -u test_swarm_config.py ^
    --embedding 618 ^
    --depth 12 ^
    --num_beings 1 ^
    --num_bits 8 ^
    --batch_size 500 ^
    --seq_len 6 ^
    --memory_size 6 ^
    --think_ticks 1 ^
    --attention_radius 6 ^
    --lcx_mode hash ^
    --lcx_num_levels 1 ^
    --lcx_level_slots "2000" ^
    --lcx_key_dim 62 ^
    --lcx_top_k 6 ^
    --num_pointers 1 ^
    --device cpu ^
    --data_dir "data/traindat/" ^
    --checkpoint_dir "checkpoints/edge_d618" ^
    --checkpoint_every 25 ^
    --eval_every 5 ^
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
