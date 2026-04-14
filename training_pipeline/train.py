import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy
import os
from sklearn.metrics import f1_score, average_precision_score
import numpy as np
from datetime import datetime
import random
import argparse
from tqdm import tqdm

# =============================
# CONFIG
# =============================
# =============================
# ARGUMENTS
# =============================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument(
    "--val_data",
    type=str,
    default="default",
    choices=["default", "bjerge_10"],
    help="Which validation dataset to use for val_loss"
)

args = parser.parse_args()

DATA_DIR = args.data_dir
SAVE_DIR = args.out_dir
SEED = args.seed
BJERGE_VAL_DIR = "/data/vision/beery/scratch/serena/Bjerge_dataset/10%_val"
# =============================

NUM_CLASSES = 9
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else 
    "cuda" if torch.cuda.is_available() else 
    "cpu"
)

print("Using device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# DATASETS & DATALOADERS
# =============================
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Select validation directory based on flag
if args.val_data == "default":
    val_dir = os.path.join(DATA_DIR, "val")
elif args.val_data == "bjerge_10":
    val_dir = BJERGE_VAL_DIR
else:
    raise ValueError(f"Unknown val_data option: {args.val_data}")

image_datasets = {
    "train": datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        data_transforms["train"]
    ),
    "val": datasets.ImageFolder(
        val_dir,
        data_transforms["val"]
    ),
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for x in ["train", "val"]
}

# =============================
# MODEL SETUP
# =============================

def setup_model(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, scheduler


def train_model(model, dataloaders, criterion, optimizer, scheduler, run_id, num_epochs=EPOCHS, patience=4, metric='val_loss', min_delta=0.001):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = -1
    epochs_no_improve = 0

    if metric in ('val_loss', 'loss'):
        best_score = float('inf')
    else:
        best_score = 0.0

    for epoch in range(num_epochs):
        print(f"\n[run {run_id}] Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []

            pbar = tqdm(dataloaders[phase], desc=f"Run {run_id} | {phase.capitalize()} Epoch {epoch+1}", leave=False)
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                
                if NUM_CLASSES == 2:
                    all_probs.extend(probs[:, 1].detach().cpu().numpy())
                else:
                    all_probs.extend(probs.detach().cpu().numpy())

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_prauc = float('nan')
            epoch_f1 = float('nan')

            try:
                if NUM_CLASSES == 2:
                    epoch_prauc = average_precision_score(all_labels, all_probs)
                else:
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
                    probs_mat = np.vstack(all_probs)
                    epoch_prauc = average_precision_score(y_true_bin, probs_mat, average='macro')
            except Exception as e:
                print(f"Could not compute PR-AUC for phase={phase}: {e}")

            try:
                epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            except Exception as e:
                print(f"Could not compute F1 for phase={phase}: {e}")

            print(f"[run {run_id}] {phase} Loss: {epoch_loss:.4f} | PR-AUC: {epoch_prauc:.4f} | F1(macro): {epoch_f1:.4f}")

            if phase == "val":
                if metric in ('val_loss', 'loss'):
                    current = epoch_loss
                elif metric == 'prauc':
                    current = epoch_prauc
                elif metric == 'f1_macro':
                    current = epoch_f1
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                if metric in ('val_loss', 'loss'):
                    is_improvement = np.isfinite(current) and (current < best_score - min_delta)
                else:
                    is_improvement = (not np.isnan(current)) and (current > best_score + min_delta)

                if is_improvement:
                    best_score = current
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    if metric in ('val_loss', 'loss'):
                        print(f"[run {run_id}] Validation {metric.upper()} decreased to {best_score:.4f} (epoch {best_epoch}) — weights updated in memory.")
                    else:
                        print(f"[run {run_id}] Validation {metric.upper()} improved to {best_score:.4f} (epoch {best_epoch}) — weights updated in memory.")
                else:
                    epochs_no_improve += 1
                    print(f"[run {run_id}] No improvement for {epochs_no_improve} epoch(s) (min_delta: {min_delta:.4f})")

        if epochs_no_improve >= patience:
            print(f"\n[run {run_id}] Early stopping triggered after {epoch+1} epochs")
            break

    if metric in ('val_loss', 'loss'):
        print(f"\n[run {run_id}] Best Validation {metric.upper()}: {best_score:.4f} (epoch {best_epoch})")
    else:
        print(f"\n[run {run_id}] Best Validation {metric.upper()}: {best_score:.4f} at epoch {best_epoch}")

    return best_model_wts, best_score, best_epoch


# =============================
# RUN MULTIPLE TRAINING ROUNDS
# =============================
n_runs = 5
results = []

selection_metric = 'val_loss' 
min_delta = 0.01            
patience = 3                 

os.makedirs(SAVE_DIR, exist_ok=True)

for run in range(n_runs):
    print(f"\n=== Starting Training Run {run+1}/{n_runs} ===")
    seed = SEED + run  
    model, criterion, optimizer, scheduler = setup_model(seed=seed)

    best_wts, best_score, best_epoch = train_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        run_id=run+1, 
        metric=selection_metric,
        patience=patience,
        min_delta=min_delta
    )

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_fname = f"best_model_run{run+1}.pth"
    save_path = os.path.join(SAVE_DIR, save_fname)

    checkpoint = {
        'model_state_dict': best_wts,
        'epoch': best_epoch,
        'selection_metric': selection_metric,
        'metric_value': best_score,
        'seed': seed,
        'timestamp': run_ts
    }
    torch.save(checkpoint, save_path)
    print(f"Saved best model for run {run+1} to: {save_path}")

    results.append({
        'run': run+1,
        'seed': seed,
        f'best_{selection_metric}': best_score,
        'best_epoch': best_epoch,
        'model_path': save_path
    })

print("\n=== Training Summary ===")
for run_result in results:
    print(f"Run {run_result['run']}: {selection_metric.upper()} = {run_result[f'best_{selection_metric}']:.4f} (epoch={run_result['best_epoch']}, seed={run_result['seed']}) -> {run_result['model_path']}")

import pandas as pd
df_results = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df_results.to_csv(os.path.join(SAVE_DIR, f'training_results_{timestamp}.csv'), index=False)