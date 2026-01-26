import os
import time
import torch

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def configure_env():
    # Force CPU + fp32 weights + fp64 pointers
    os.environ["VAR_COMPUTE_DEVICE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TP6_PRECISION"] = "fp32"
    os.environ["TP6_PTR_DTYPE"] = "fp64"

    # Threading
    threads = "24"
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    torch.set_num_threads(int(threads))

    # Resume + checkpoints
    os.environ["TP6_RESUME"] = "1"
    os.environ["TP6_CKPT"] = os.path.join("checkpoints", "checkpoint.pt")
    os.environ["TP6_SAVE_EVERY_STEPS"] = "50"
    os.environ["TP6_EVAL_EVERY_STEPS"] = "10"
    os.environ["TP6_EVAL_AT_CHECKPOINT"] = "0"

    # Synth config (assoc_byte)
    os.environ["TP6_SYNTH"] = "1"
    os.environ["TP6_SYNTH_MODE"] = "assoc_byte"
    os.environ["TP6_SYNTH_LEN"] = "512"
    os.environ["TP6_ASSOC_KEYS"] = "64"
    os.environ["TP6_ASSOC_PAIRS"] = "4"
    os.environ["TP6_MAX_SAMPLES"] = "8192"
    os.environ["TP6_BATCH_SIZE"] = "152"
    os.environ["TP6_OFFLINE_ONLY"] = "1"
    os.environ["TP6_EXPERT_HEADS"] = "16"

    # Logging path
    os.environ["VAR_LOGGING_PATH"] = os.path.join("logs", "current", "tournament_phase6.log")


def run_loop():
    while True:
        try:
            import tournament_phase6 as tp6
            tp6.main()
        except KeyboardInterrupt:
            print("[VRAXION] Manual stop requested.")
            break
        except Exception as exc:
            print(f"[VRAXION] Crash detected: {exc}")
            time.sleep(3)
            continue


if __name__ == "__main__":
    os.chdir(WORK_DIR)
    os.makedirs(os.path.join(WORK_DIR, "logs", "current"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "checkpoints"), exist_ok=True)
    configure_env()
    run_loop()
