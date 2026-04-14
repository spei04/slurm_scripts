import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import time
import copy
import os
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
import random
import argparse
from tqdm import tqdm
import pandas as pd

# =============================
# ARGUMENTS
# =============================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to real data (train/val)")
parser.add_argument("--generated_base", type=str, required=True, help="Path to synthetic images for curriculum")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument(
    "--reverse",
    action="store_true",
    help="Reverse curriculum order (hard → easy)"
)
parser.add_argument(
    "--val_data",
    type=str,
    default="default",
    choices=["default", "bjerge_10"],
    help="Which validation dataset to use"
)
args = parser.parse_args()

DATA_DIR = args.data_dir
SYNTHETIC_DIR = args.generated_base
SAVE_DIR = args.out_dir
SEED = args.seed
# Hardcoded validation path as requested
BJERGE_VAL_DIR = "/data/vision/beery/scratch/serena/Bjerge_dataset/10%_val"

if args.val_data == "default":
    val_dir = os.path.join(DATA_DIR, "val")
elif args.val_data == "bjerge_10":
    val_dir = BJERGE_VAL_DIR
else:
    raise ValueError(f"Unknown val_data option: {args.val_data}")

# =============================
# CONFIG
# =============================
NUM_CLASSES = 9
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20 
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else 
    "cuda" if torch.cuda.is_available() else 
    "cpu"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# TRANSFORMS
# =============================
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# =============================
# CURRICULUM SETTINGS
# =============================
synthetic_counts = [0, 100, 200, 500, 1000]

if args.reverse:
    synthetic_counts.reverse()
curriculum_epochs = np.linspace(0, EPOCHS-1, len(synthetic_counts), dtype=int)

def get_curriculum_count(epoch):
    for i, e in enumerate(curriculum_epochs):
        if epoch <= e: return synthetic_counts[i]
    return synthetic_counts[-1]

def get_train_dataloader(epoch):
    n_synth = get_curriculum_count(epoch)
    # Load real training data
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), data_transforms["train"])
    
    if n_synth > 0:
        # Load synthetic data from the --generated_base directory
        synth_ds = datasets.ImageFolder(SYNTHETIC_DIR, data_transforms["train"])
        indices = np.random.choice(len(synth_ds), min(n_synth, len(synth_ds)), replace=False)
        synth_ds = Subset(synth_ds, indices)
        train_ds = ConcatDataset([train_ds, synth_ds])
    
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Validation dataloader (Now using the specific 10%_val path)
val_ds = datasets.ImageFolder(val_dir, data_transforms["val"])
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =============================
# MODEL SETUP
# =============================
def setup_model(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, scheduler

# =============================
# TRAINING LOOP
# =============================
def train_model(model, val_loader, criterion, optimizer, scheduler, run_id, num_epochs=EPOCHS, patience=4):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        print(f"\n[Run {run_id}] Epoch {epoch+1}/{num_epochs}")
        
        train_loader = get_train_dataloader(epoch)

        for phase in ["train", "val"]:
            loader = train_loader if phase == "train" else val_loader
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            all_preds, all_labels = [], []

            pbar = tqdm(loader, desc=f"{phase.upper()}", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            print(f"{phase} Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f}")

            # Selection based on val_loss
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    print(f"⭐ New best Val Loss: {best_loss:.4f}")
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_model_wts, best_loss, best_epoch

# =============================
# EXECUTION
# =============================
n_runs = 5
results = []

for run in range(n_runs):
    print(f"\n{'='*30}\nRUN {run+1}/{n_runs}\n{'='*30}")
    seed = SEED + run
    model, criterion, optimizer, scheduler = setup_model(seed=seed)

    best_wts, best_loss, best_epoch = train_model(
        model, val_loader, criterion, optimizer, scheduler, run_id=run+1
    )

    save_path = os.path.join(SAVE_DIR, f"best_model_run{run+1}.pth")
    torch.save({
        'model_state_dict': best_wts,
        'val_loss': best_loss,
        'epoch': best_epoch,
        'seed': seed
    }, save_path)

    results.append({'run': run+1, 'val_loss': best_loss, 'epoch': best_epoch})

pd.DataFrame(results).to_csv(os.path.join(SAVE_DIR, 'summary.csv'), index=False)
print("\nDone. Models and summary saved to:", SAVE_DIR)