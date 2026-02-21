@echo off
REM ================================================================
REM  GOLDILOCKS WIDE — D=6180 GPU Model (v4.1 — shallow)
REM  Architecture: LOCKED (do not change these params)
REM ================================================================
REM
REM  D=6180  depth=2  ring=62  seq=62  bits=616  beings=1
REM  LCX: single L0, hash-bucketed, key_dim=618, top_k=2
REM  GPU fp32 training, ~1s/step Alpha, ~5s/step Beta
REM  Deploy: 206M params, ~0.4GB fp16, ~9 GB VRAM training (Beta)
REM
REM  Golden Ratio Fractal Stack:
REM    top_k    = 2    (empirical optimum, probe #91)
REM    seq_len  = 62   (phi x 100,   Scale 2)
REM    ring     = 62   (phi x 100,   Scale 2)
REM    bits     = 616  (phi x 1000,  Scale 3, padded to x8)
REM    key_dim  = 618  (phi x 1000,  Scale 3)
REM    BN_dim   = 618  (phi x 1000,  Scale 3)
REM    D        = 6180 (phi x 10000, Scale 4)
REM    depth    = 2    (probe-validated: shallower is better, see probe_depth_v2.py)
REM
REM  Context window: 4,774 bytes per forward pass (62 pos x 77 bytes)
REM  + LCX: 2000 slots long-term memory
REM
REM  EFFORT TIERS (via controls.json):
REM  ------------------------------------------------------------------
REM  Alpha(Reflex):     tt=0  batch=16  lcx=OFF  <-- START
REM  Beta(Recall):      tt=1  batch=16  lcx=ON
REM  Gamma(Reason):     tt=2  batch=8   lcx=ON
REM  Delta(Depth):      tt=4  batch=4   lcx=ON
REM  Epsilon(Emerge):   tt=8  batch=2   lcx=ON
REM  Zeta(Zenith):      tt=16 batch=1   lcx=ON
REM  ------------------------------------------------------------------
REM ================================================================

cd /d "S:\AI\work\VRAXION_DEV\Diamond Code"

python -u test_swarm_config.py ^
    --embedding 6180 ^
    --depth 2 ^
    --num_beings 1 ^
    --num_bits 616 ^
    --binary-bits ^
    --batch_size 16 ^
    --seq_len 62 ^
    --memory_size 62 ^
    --think_ticks 0 ^
    --attention_radius 6 ^
    --lcx_mode hash ^
    --lcx_num_levels 1 ^
    --lcx_level_slots "2000" ^
    --lcx_key_dim 618 ^
    --lcx_top_k 2 ^
    --num_pointers 1 ^
    --device cuda ^
    --themes_dir "data/themes/" ^
    --active_theme arithmetic_mul ^
    --gist_ratio 0.10 ^
    --checkpoint_dir "checkpoints/goldilocks_d6" ^
    --checkpoint_every 25 ^
    --eval_every 5 ^
    --steps 1000000 ^
    --lr 0.0003 ^
    --warmup_steps 100 ^
    --lr_min 1e-5 ^
    --controls_every 1 ^
    --effort Alpha

echo Training ended.
pause
