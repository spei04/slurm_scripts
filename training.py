import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy
import os
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import numpy as np
from datetime import datetime
import random

species_list = ["Coccinellidae septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris", "Eupeodes corollae", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris", "Eristalis tenax"]

# =============================
# CONFIG
# =============================
from torch.utils.data import Subset
from collections import defaultdict
import random

# =============================
# CONFIG
# =============================
DATA_DIR = "/data/vision/beery/scratch/serena/insect_analysis/BJERGE_NEW/prepared_GBIF_BJERGE"
NUM_CLASSES = 9
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")
# Where to save trained model files
SAVE_DIR = os.path.expanduser("/data/vision/beery/scratch/serena/insect_analysis/BJERGE_NEW/trained_model_GBIF_qwen_edit")
#SAVE_PATH = "gen_vs_orig.pth"

from torchvision import transforms

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

selection_metric = "prauc"   # options: "prauc", "f1_macro", "val_loss"
patience = 4
min_delta = 0.001

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
            # Note: MPS might not support all seed operations
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass

    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, scheduler


def train_model(model, dataloaders, criterion, optimizer, scheduler, run_id, num_epochs=EPOCHS, patience=4, metric='prauc', min_delta=0.001):
    """
    Train the model and track the best validation metric.

    Args:
        metric: 'prauc', 'f1_macro', or 'val_loss' (pick model with lowest val loss)
        min_delta: Minimum change in metric to qualify as improvement. For 'val_loss'
                   this means the loss must decrease by more than min_delta; for
                   other metrics it must increase by more than min_delta.
        patience: Number of validation epochs with no sufficient improvement
                  before early stopping triggers.

    Returns: best_model_wts (state_dict), best_score (float), best_epoch (int)
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = -1
    epochs_no_improve = 0

    # Initialize best_score depending on metric direction
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
            all_probs = []  # Store probabilities for PR-AUC

            for inputs, labels in dataloaders[phase]:
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
                # If NUM_CLASSES == 2 we use probs[:,1], otherwise we will compute multi-class AP later
                if NUM_CLASSES == 2:
                    all_probs.extend(probs[:, 1].detach().cpu().numpy())  # Prob of positive class
                else:
                    # Store full probability vectors for multi-class
                    all_probs.extend(probs.detach().cpu().numpy())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # Compute metrics safely
            epoch_prauc = float('nan')
            epoch_f1 = float('nan')

            try:
                if NUM_CLASSES == 2:
                    epoch_prauc = average_precision_score(all_labels, all_probs)
                else:
                    # multi-class macro average precision: binarize labels and compute macro-averaged AP
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
                # Decide which value to use for improvement checks
                if metric in ('val_loss', 'loss'):
                    current = epoch_loss
                elif metric == 'prauc':
                    current = epoch_prauc
                elif metric == 'f1_macro':
                    current = epoch_f1
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                # Determine improvement depending on whether higher is better or lower is better
                if metric in ('val_loss', 'loss'):
                    # Lower is better: improvement when current < best_score - min_delta
                    is_improvement = np.isfinite(current) and (current < best_score - min_delta)
                else:
                    # Higher is better: improvement when current > best_score + min_delta
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

        # EARLY STOPPING
        if epochs_no_improve >= patience:
            print(f"\n[run {run_id}] Early stopping triggered after {epoch+1} epochs")
            break

    # Final summary print
    if metric in ('val_loss', 'loss'):
        print(f"\n[run {run_id}] Best Validation {metric.upper()}: {best_score:.4f} (epoch {best_epoch})")
    else:
        print(f"\n[run {run_id}] Best Validation {metric.upper()}: {best_score:.4f} at epoch {best_epoch}")

    # Return the best weights & metrics; caller will save the single model file
    return best_model_wts, best_score, best_epoch



def make_subset_per_class(dataset, images_per_species, seed=0):
    """
    Returns a Subset of `dataset` with at most `images_per_species`
    samples per class.
    """
    rng = random.Random(seed)
    class_to_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) <= images_per_species:
            selected = indices
        else:
            selected = rng.sample(indices, images_per_species)
        selected_indices.extend(selected)

    return Subset(dataset, selected_indices)

from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset

DATA_DIR = "/data/vision/beery/scratch/serena/insect_analysis/BJERGE_NEW/prepared_GBIF_BJERGE"
QWEN_DATA_DIR = "/data/vision/beery/scratch/serena/diffusion/generated_full_dataset_qwen_edit"

def get_dataloaders(images_per_species, seed=0):
    """
    Training uses all GBIF images + `images_per_species` per class from Qwen edits.
    Validation is unchanged.
    """
    # GBIF training dataset (all images)
    gbif_train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        data_transforms["train"]
    )

    # Qwen dataset
    qwen_train_dataset = datasets.ImageFolder(
        os.path.join(QWEN_DATA_DIR),
        data_transforms["train"]
    )

    # Subsample Qwen images per class
    qwen_train_subset = make_subset_per_class(
        qwen_train_dataset,
        images_per_species,
        seed=seed
    )

    # Merge GBIF (all) + Qwen (subsampled)
    combined_train_dataset = ConcatDataset([gbif_train_dataset, qwen_train_subset])

    # Validation dataset (unchanged)
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        data_transforms["val"]
    )

    dataloaders = {
        "train": DataLoader(
            combined_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
    }

    return dataloaders, {"train": combined_train_dataset, "val": val_dataset}



images_per_species_list = [500]
n_runs = 5

all_results = []

for images_per_species in images_per_species_list:
    print(f"\n==============================")
    print(f"Training with {images_per_species} images per species")
    print(f"==============================")

    # Create dataloaders for this setting
    dataloaders, image_datasets = get_dataloaders(
        images_per_species=images_per_species,
        seed=42
    )

    SAVE_DIR_K = os.path.join(SAVE_DIR, f"{images_per_species}_per_class")
    os.makedirs(SAVE_DIR_K, exist_ok=True)

    results = []

    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} | {images_per_species} imgs/class ===")
        seed = 42 + run

        model, criterion, optimizer, scheduler = setup_model(seed=seed)

        best_wts, best_score, best_epoch = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            scheduler,
            run_id=f"{images_per_species}_run{run+1}",
            metric=selection_metric,
            patience=patience,
            min_delta=min_delta
        )

        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(
            SAVE_DIR_K,
            f"best_model_{images_per_species}_run{run+1}.pth"
        )

        checkpoint = {
            'model_state_dict': best_wts,
            'epoch': best_epoch,
            'selection_metric': selection_metric,
            'metric_value': best_score,
            'seed': seed,
            'images_per_species': images_per_species,
            'timestamp': run_ts
        }

        torch.save(checkpoint, save_path)

        results.append({
            'images_per_species': images_per_species,
            'run': run + 1,
            'seed': seed,
            f'best_{selection_metric}': best_score,
            'best_epoch': best_epoch,
            'model_path': save_path
        })

    all_results.extend(results)
