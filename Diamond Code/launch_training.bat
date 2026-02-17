@echo off
cd /d "S:\AI\work\VRAXION_DEV\Diamond Code"

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
    --num_pointers 1 ^
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
    --effort Beta ^
    --resume "checkpoints/goldilocks_v4/checkpoint_latest.pt"

pause
