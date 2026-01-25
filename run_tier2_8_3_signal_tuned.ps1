# run_tier2_8_3_signal_tuned.ps1
# Tier 2.8.x: reward/pain inertia signal (ratchet disabled) - tuned strength
$root = "S:\AI\mirror\VRAXION"
Set-Location $root

# Ensure no manual override sticks
Remove-Item Env:TP6_PTR_INERTIA_OVERRIDE -ErrorAction SilentlyContinue
Remove-Item Env:TP6_IGNORE_MAX_STEPS -ErrorAction SilentlyContinue

# CPU + f64
$env:VAR_COMPUTE_DEVICE = "cpu"
$env:CUDA_VISIBLE_DEVICES = ""
$env:TP6_PRECISION = "fp64"
$env:TP6_PTR_DTYPE = "fp64"
$env:OMP_NUM_THREADS = "24"
$env:MKL_NUM_THREADS = "24"

# Synthetic assoc_clean treadmill
$env:TP6_SYNTH = "1"
$env:TP6_SYNTH_MODE = "assoc_clean"
$env:TP6_SYNTH_LEN = "16"
$env:TP6_ASSOC_KEYS = "4"
$env:TP6_ASSOC_PAIRS = "1"
$env:TP6_OFFLINE_ONLY = "1"

# Ring + experts
$env:TP6_RING_LEN = "32"
$env:TP6_EXPERT_HEADS = "16"

# Eval + checkpoint cadence
$env:TP6_EVAL_EVERY_STEPS = "10"
$env:TP6_EVAL_AT_CHECKPOINT = "0"
$env:TP6_SAVE_EVERY_STEPS = "50"
$env:TP6_MAX_STEPS = "300"
$env:TP6_VCOG_PRG_FROM_ACC = "1"
$env:TP6_VCOG_PRG_ACC_TARGET = "1.0"

# Checkpoint + log paths
$env:TP6_RESUME = "1"
$env:TP6_CKPT = "checkpoints/sanity_tier2_8_3_signal/ckpt.pt"
$env:VAR_LOGGING_PATH = "logs/sanity_tier2_8_3_signal_tuned.log"

# Replace ratchet with reward/pain signal (tuned)
$env:TP6_INERTIA_RATCHET = "0"
$env:TP6_INERTIA_SIGNAL = "1"
$env:TP6_INERTIA_SIGNAL_ACC_MIN = "0.95"
$env:TP6_INERTIA_SIGNAL_STREAK = "1"
$env:TP6_INERTIA_SIGNAL_TARGET = "0.95"
$env:TP6_INERTIA_SIGNAL_FLOOR = "0.70"
$env:TP6_INERTIA_SIGNAL_REWARD = "0.50"
$env:TP6_INERTIA_SIGNAL_PAIN = "0.25"

python tournament_phase6.py
