@echo off
REM Heterogeneous 2-being (Scanner+Thinker) D=128 training run
REM Scanner: depth=1, tt=0, no_grad (pre-fills LCX + ring)
REM Thinker: depth=2, tt=1, with grad (reads from scanner context)
python -u test_swarm_config.py ^
    --embedding 128 --depth "1,2" --num_beings 2 ^
    --think_ticks "0,1" --being_modes "scanner,thinker" ^
    --binary-bits --num_bits 8 --batch_size 32 --seq_len 16 ^
    --memory_size 16 --attention_radius 3 --lcx_mode hash ^
    --lcx_num_levels 1 --lcx_level_slots "100" --lcx_key_dim 32 ^
    --lcx_top_k 3 --num_pointers 1 --fp64 --device cpu ^
    --data_dir "data/traindat/" --checkpoint_dir "checkpoints/d128_hetero" ^
    --data_weights "{\"copy_echo256.traindat\":1,\"constant256.traindat\":1,\"add256.traindat\":0,\"count256.traindat\":0,\"delay_echo256.traindat\":0,\"denoise256.traindat\":0,\"echo256.traindat\":0,\"fib256.traindat\":0,\"gold_origin_echo.traindat\":0,\"not256.traindat\":0,\"shift256.traindat\":0}" ^
    --steps 100 --lr 0.0003 --effort Beta
