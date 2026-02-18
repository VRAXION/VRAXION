"""Launcher for Goldilocks Nano training — avoids shell escaping issues."""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_WEIGHTS = (
    '{"copy_echo256.traindat":1,'
    '"constant256.traindat":0,"add256.traindat":0,"count256.traindat":0,'
    '"delay_echo256.traindat":0,"denoise256.traindat":0,"echo256.traindat":0,'
    '"fib256.traindat":0,"gold_origin_echo.traindat":0,"not256.traindat":0,'
    '"shift256.traindat":0}'
)

cmd = [
    sys.executable, "-u", "test_swarm_config.py",
    "--embedding", "618",
    "--depth", "62",
    "--num_beings", "1",
    "--num_bits", "6184",
    "--binary-bits",
    "--batch_size", "10",
    "--seq_len", "6",
    "--memory_size", "6",
    "--think_ticks", "1",
    "--attention_radius", "3",
    "--lcx_mode", "hash",
    "--lcx_num_levels", "1",
    "--lcx_level_slots", "2000",
    "--lcx_key_dim", "61",
    "--lcx_top_k", "6",
    "--num_pointers", "1",
    "--fp64",
    "--device", "cpu",
    "--data_dir", "data/traindat/",
    "--data_weights", DATA_WEIGHTS,
    "--checkpoint_dir", "checkpoints/nano_golden",
    "--checkpoint_every", "25",
    "--eval_every", "5",
    "--steps", "1000000",
    "--lr", "0.0003",
    "--warmup_steps", "100",
    "--lr_min", "1e-5",
    "--controls_every", "1",
    "--effort", "Beta",
    "--resume", "checkpoints/nano_golden/checkpoint_latest.pt",
]

subprocess.run(cmd)
