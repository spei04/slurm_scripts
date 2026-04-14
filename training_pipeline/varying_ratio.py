import subprocess
import sys
from pathlib import Path
import numpy as np

# -------- CONFIG --------
N_RUNS = 5
# SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = Path("/data/vision/beery/scratch/serena/training_pipeline")
BASE_OUTDIR = SCRIPT_DIR
DATASET_SEED = 42 
CACHE_DATASET_DIR = SCRIPT_DIR / "tmp_augmented_small"

initial = 0
increment = 100
max_original = 1000
# n_orig_target = 200
# ------------------------

for n_gen in range(initial, max_original + 1, increment):
    exp_dir = BASE_OUTDIR / f"Siblings_Augmentation_{n_gen}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Creating dataset for {n_gen} additional images ===")

    # 1. Create dataset ONCE for this ratio
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "split_creation.py"),
        "--seed", str(DATASET_SEED),
        # "--target_classes", "Bombus lapidarius", "Bombus terrestris",
        # "--start_origs", str(n_orig_target), str(n_orig_target),
        "--increment", str(n_gen),
        "--out_dir", str(CACHE_DATASET_DIR),
        "--generated_base", "/data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_qwen_edit"
    ], check=True)

    # 2. Train ResNet18 on the temporary dataset
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "train_bjerge_val.py"),
        "--data_dir", str(CACHE_DATASET_DIR),
        "--seed", str(DATASET_SEED),
        "--out_dir", str(exp_dir),
        # "--val_data", "bjerge_10"
    ], check=True)
