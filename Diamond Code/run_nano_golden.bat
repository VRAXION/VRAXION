@echo off
REM ================================================================
REM  GOLDILOCKS NANO — Full Golden Ratio Fractal (CPU Edge Model)
REM  Architecture: LOCKED (do not change these params)
REM ================================================================
REM
REM  D=618  depth=62  ring=6  seq=6  bits=6184  beings=1
REM  LCX: single L0, hash-bucketed, key_dim=61, top_k=6
REM  CPU fp64 training, ~1.7s/step at batch=10
REM  Deploy: 31.2M params, 59.5MB fp16, ~0.7 GB RAM
REM
REM  Golden Ratio Fractal Stack:
REM    seq_len  = 6    (phi x 10)
REM    key_dim  = 61   (phi x 100)
REM    D        = 618  (phi x 1000)
REM    num_bits = 6184 (phi x 10000, padded to x8)
REM    depth    = 62   (phi x 100)
REM    top_k    = 6    (phi x 10)
REM
REM  Context window: 4,638 bytes per forward pass
REM  + LCX: 2000 slots long-term memory
REM
REM  EFFORT TIERS (via controls.json):
REM  ------------------------------------------------------------------
REM  Alpha(Reflex):     tt=0  batch=10  lcx=OFF  <-- START
REM  Beta(Recall):      tt=1  batch=10  lcx=ON
REM  Gamma(Reason):     tt=2  batch=5   lcx=ON
REM  Delta(Depth):      tt=4  batch=3   lcx=ON
REM  Epsilon(Emerge):   tt=8  batch=2   lcx=ON
REM  Zeta(Zenith):      tt=16 batch=1   lcx=ON
REM  ------------------------------------------------------------------
REM ================================================================

cd /d "S:\AI\work\VRAXION_DEV\Diamond Code"

python -u test_swarm_config.py ^
    --embedding 618 ^
    --depth 62 ^
    --num_beings 1 ^
    --num_bits 6184 ^
    --binary-bits ^
    --batch_size 10 ^
    --seq_len 6 ^
    --memory_size 6 ^
    --think_ticks 1 ^
    --attention_radius 3 ^
    --lcx_mode hash ^
    --lcx_num_levels 1 ^
    --lcx_level_slots "2000" ^
    --lcx_key_dim 61 ^
    --lcx_top_k 6 ^
    --num_pointers 1 ^
    --fp64 ^
    --device cpu ^
    --data_dir "data/traindat/" ^
    --checkpoint_dir "checkpoints/nano_golden" ^
    --checkpoint_every 25 ^
    --eval_every 5 ^
    --steps 1000000 ^
    --lr 0.0003 ^
    --warmup_steps 100 ^
    --lr_min 1e-5 ^
    --controls_every 1 ^
    --effort Beta ^
    --resume "checkpoints/nano_golden/checkpoint_latest.pt"

echo Training ended.
pause
