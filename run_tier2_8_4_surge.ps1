$workDir = "S:\AI\mirror\VRAXION"
if (-not (Test-Path $workDir)) {
    Write-Error "Workdir not found: $workDir"
    exit 1
}

Set-Location $workDir
New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "$workDir\checkpoints\tier2_8_4_surge" | Out-Null

# CPU f64 foundry
$env:VAR_COMPUTE_DEVICE = "cpu"
$env:CUDA_VISIBLE_DEVICES = ""
$env:TP6_PRECISION = "fp64"
$env:TP6_PTR_DTYPE = "fp64"
$env:OMP_NUM_THREADS = "24"
$env:MKL_NUM_THREADS = "24"

# Tier 2.8.4 (Adrenaline surge) settings
$env:TP6_SYNTH = "1"
$env:TP6_SYNTH_MODE = "assoc_clean"
$env:TP6_ASSOC_KEYS = "4"
$env:TP6_ASSOC_PAIRS = "1"
$env:TP6_SYNTH_LEN = "16"
$env:TP6_RING_LEN = "32"
$env:TP6_EXPERT_HEADS = "16"
$env:TP6_RESUME = "0"
$env:TP6_EVAL_EVERY_STEPS = "10"
$env:TP6_SAVE_EVERY_STEPS = "50"
$env:TP6_MAX_STEPS = "400"
$env:TP6_DISABLE_SYNC = "1"

# Inertia signal (adrenaline surge)
$env:TP6_INERTIA_SIGNAL = "1"
$env:TP6_INERTIA_SIGNAL_ACC_MIN = "0.98"
$env:TP6_INERTIA_SIGNAL_STREAK = "2"
$env:TP6_INERTIA_SIGNAL_TARGET = "0.95"
$env:TP6_INERTIA_SIGNAL_FLOOR = "0.5"
$env:TP6_INERTIA_SIGNAL_REWARD = "0.5"
$env:TP6_INERTIA_SIGNAL_PAIN = "0.25"

$env:TP6_CKPT = "checkpoints/tier2_8_4_surge/ckpt.pt"
$env:VAR_LOGGING_PATH = "logs/tier2_8_4_surge.log"

Write-Host ">>> Tier 2.8.4 Adrenaline Surge (CPU f64) starting..." -ForegroundColor Cyan
python "$workDir\tournament_phase6.py"
