# import os
# import random
# import shutil
# from pathlib import Path
# import time
# import argparse

# # python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 0 --out_dir /data/vision/beery/scratch/serena/training_pipeline/splits_test

# # --- Arguments ---
# parser = argparse.ArgumentParser()
# parser.add_argument("--increment", type=int, required=True,
#                     help="Number of generated images to ADD to the training set for each class")
# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--out_dir", type=str, required=True, 
#                     help="Directory to save the new dataset splits")
# parser.add_argument("--target_classes", type=str, nargs='+', default=None,
#                     help="List of classes to limit original images for (space-separated)")
# parser.add_argument("--start_origs", type=int, nargs='+', default=None,
#                     help="List of initial original TRAIN image limits corresponding to target_classes (space-separated)")

# args = parser.parse_args()

# # --- Configuration ---
# original_base = Path('/data/vision/beery/scratch/serena/GBIF_prepared_downsampled') 
# generated_base = Path('/data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_qwen_edit')
# # generated_base = Path('/data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_flux')
# # generated_base = Path('/data/vision/beery/scratch/serena/diffusion/filtered_generated_full_dataset_flux')

# increment = args.increment
# RANDOM_SEED = args.seed
# train_val_ratio = 0.8
# output_dir = Path(args.out_dir,)

# # --- Validation & Setup for Multiple Targets ---
# target_classes = args.target_classes or []
# start_origs = args.start_origs or []

# if len(target_classes) != len(start_origs):
#     raise ValueError(f"Mismatch! You provided {len(target_classes)} target classes but {len(start_origs)} start_orig limits. They must match.")

# # Create a dictionary for easy lookup: {'Class A': 200, 'Class B': 150}
# target_mapping = dict(zip(target_classes, start_origs))

# random.seed(RANDOM_SEED)
# os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# # --- Step 1: Reset Output Directory ---
# if output_dir.exists():
#     print(f"Cleaning up: {output_dir}...")
#     shutil.rmtree(output_dir, ignore_errors=True)
#     time.sleep(0.5) #Added because my SSD sometimes needs a moment to release file locks after deletion

# output_dir.mkdir(parents=True, exist_ok=True)

# # --- Step 2: Identify Classes ---
# def _collect_original_classes(base_dir: Path, splits: list[str]) -> list[str]:
#     class_names = set()
#     for split in splits:
#         split_dir = base_dir / split
#         if split_dir.exists():
#             for entry in split_dir.iterdir():
#                 if entry.is_dir():
#                     class_names.add(entry.name)
#     return sorted(class_names)

# classes = _collect_original_classes(original_base, ['train', 'val'])

# if not classes:
#     raise ValueError(f"No classes found in {original_base}/train or {original_base}/val")

# stats = []
# warnings = []

# for cls in classes:
#     # 1. Gather ALL original images (Merge current Train + Val)
#     orig_sources = [original_base / 'train' / cls, original_base / 'val' / cls]
#     all_orig_images = []
#     for src in orig_sources:
#         if src.exists():
#             files = [f for f in src.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
#             all_orig_images.extend(files)
    
#     # 2. Gather ALL generated images
#     gen_source = generated_base / cls
#     all_gen_images = []
#     if gen_source.exists():
#         all_gen_images = [f for f in gen_source.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

#     random.shuffle(all_orig_images)
#     random.shuffle(all_gen_images)

#     # --- Step 3: Distribution Logic ---
#     # Check if the current class is in our target dictionary
#     is_target = cls in target_mapping

#     # Determine how many generated images we can actually use
#     actual_gen_count = min(len(all_gen_images), increment)
#     if actual_gen_count < increment:
#         warnings.append(f"[{cls}] Requested {increment} generated images, but only found {len(all_gen_images)}.")

#     train_gen = all_gen_images[:actual_gen_count]

#     if not is_target:
#         # =========================
#         # NORMAL CLASS (ALL ORIGINALS)
#         # =========================
#         total_orig_count = len(all_orig_images)
#         train_orig_cap = int(total_orig_count * train_val_ratio)

#         train_orig = all_orig_images[:train_orig_cap]
#         new_val_set = all_orig_images[train_orig_cap:]
        
#         final_train_set = train_orig + train_gen

#     else:
#         # =========================
#         # TARGET CLASS (CUSTOM CONTROL)
#         # =========================
#         # Retrieve the specific limit for this class
#         start_orig = target_mapping[cls]

#         if len(all_orig_images) < start_orig:
#             raise ValueError(f"[{cls}] Not enough ORIGINAL images. Required: {start_orig}, Available: {len(all_orig_images)}")

#         # Calculate proportional validation set size to maintain the 80/20 ratio 
#         # (e.g., if train is 80, val will be 20)
#         val_cap = int((start_orig / train_val_ratio) - start_orig)

#         if len(all_orig_images) < start_orig + val_cap:
#             raise ValueError(f"[{cls}] Not enough ORIGINAL images for both train ({start_orig}) and proportional val ({val_cap}).")

#         train_orig = all_orig_images[:start_orig]
#         new_val_set = all_orig_images[start_orig : start_orig + val_cap]

#         final_train_set = train_orig + train_gen

#     # Track stats
#     stats.append({
#         'class': cls,
#         'total_train': len(final_train_set),
#         'actual_orig': len(train_orig),
#         'actual_gen': len(train_gen),
#         'val_count': len(new_val_set)
#     })

#     # --- Step 4: Copying ---
#     for split_name, image_list in [('train', final_train_set), ('val', new_val_set)]:
#         dest_path = output_dir / split_name / cls
#         dest_path.mkdir(parents=True, exist_ok=True)
#         for img in image_list:
#             shutil.copy2(img, dest_path / img.name)

# # --- Step 5: Final Summary ---
# print("\n" + "="*85)
# print(f"{'Class Name':<30} | {'Total(T)':<8} | {'Orig(T)':<8} | {'Gen(T)':<8} | {'Val(V)':<8}")
# print("-" * 85)

# for s in stats:
#     print(f"{s['class'][:30]:<30} | {s['total_train']:<8} | {s['actual_orig']:<8} | {s['actual_gen']:<8} | {s['val_count']:<8}")

# print("-" * 85)
# print(f"RANDOM SEED: {RANDOM_SEED} | INCREMENT: +{increment} Gen Images")
# print("="*85)

# if warnings:
#     print("\n--- WARNINGS ---")
#     for w in warnings:
#         print(w)


# # Just added a final cleanup step to remove any macOS metadata files (._*) that might have been copied over during the shutil.copy2 operations, as these can sometimes cause issues in downstream processing.
# # I kinda broke my SSD and it is leaving behind these annoying files that I can't easily delete, so this is a workaround to clean them up from the new dataset splits after creation. If you don't have this issue, you can safely ignore this part.
# # --- Step 6: Remove macOS metadata files (._*) ---
# # removed = 0
# # for root, _, files in os.walk(output_dir):
# #     for f in files:
# #         if f.startswith("._"):
# #             try:
# #                 os.remove(os.path.join(root, f))
# #                 removed += 1
# #             except Exception as e:
# #                 print(f"Warning: could not remove {f}: {e}")

# # if removed > 0:
# #     print(f"\nRemoved {removed} macOS metadata files (._*) from dataset.")
import os
import random
import shutil
from pathlib import Path
import time
import argparse
from PIL import Image
import torchvision.transforms as transforms

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--increment", type=int, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--target_classes", type=str, nargs='+', default=None)
parser.add_argument("--start_origs", type=int, nargs='+', default=None)
parser.add_argument("--aug", action='store_true', help="Apply stochastic augmentations to generated images")
parser.add_argument("--use_all_generated", action='store_true', help="Include all generated images instead of increment")
parser.add_argument("--aug_name", type=str, default="full", help="Which augmentation config to use for generated images")
parser.add_argument("--generated_base", type=str, required=True)
    
args = parser.parse_args()

# --- Configuration ---
original_base = Path('/data/vision/beery/scratch/serena/GBIF_prepared_downsampled') 
generated_base = Path(args.generated_base)
increment = args.increment
RANDOM_SEED = args.seed
train_val_ratio = 0.8
output_dir = Path(args.out_dir)
apply_aug = args.aug
use_all_generated = args.use_all_generated
aug_name = args.aug_name

target_classes = args.target_classes or []
start_origs = args.start_origs or []

if len(target_classes) != len(start_origs):
    raise ValueError("Mismatch: target_classes and start_origs must have same length")

target_mapping = dict(zip(target_classes, start_origs))
random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# --- Reset Output Directory ---
if output_dir.exists():
    shutil.rmtree(output_dir, ignore_errors=True)
    time.sleep(0.5)
output_dir.mkdir(parents=True, exist_ok=True)

# --- Collect classes ---
def _collect_original_classes(base_dir: Path, splits: list[str]) -> list[str]:
    class_names = set()
    for split in splits:
        split_dir = base_dir / split
        if split_dir.exists():
            for entry in split_dir.iterdir():
                if entry.is_dir():
                    class_names.add(entry.name)
    return sorted(class_names)

classes = _collect_original_classes(original_base, ['train', 'val'])
if not classes:
    raise ValueError("No classes found in original dataset")

# --- Augmentation Configs ---
AUGMENTATION_CONFIGS = {
    "flip":                dict(flip=True,  rotation=False, color=False, blur=False, solarize=False),
    "rotation":            dict(flip=False, rotation=True,  color=False, blur=False, solarize=False),
    "color":               dict(flip=False, rotation=False, color=True,  blur=False, solarize=False),
    "flip_rotation":       dict(flip=True,  rotation=True,  color=False, blur=False, solarize=False),
    "flip_color":          dict(flip=True,  rotation=False, color=True,  blur=False, solarize=False),
    "rotation_color":      dict(flip=False, rotation=True,  color=True,  blur=False, solarize=False),
    "flip_rotation_color": dict(flip=True,  rotation=True,  color=True,  blur=False, solarize=False),
    "full":                dict(flip=True,  rotation=True,  color=True,  blur=True,  solarize=True),
}

NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def random_layered_augment(img: Image.Image, aug_name: str) -> Image.Image:
    """Apply stochastic augmentations from AUGMENTATION_CONFIGS with probabilities."""
    cfg = AUGMENTATION_CONFIGS.get(aug_name, AUGMENTATION_CONFIGS["full"])
    aug_ops = []

    if cfg["flip"]:
        aug_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg["rotation"]:
        aug_ops.append(transforms.RandomApply([transforms.RandomRotation(30)], p=0.5))
    if cfg["color"]:
        aug_ops.append(transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)], p=0.5
        ))
    if cfg["blur"]:
        aug_ops.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(1,3))], p=0.3))
    if cfg["solarize"]:
        aug_ops.append(transforms.RandomSolarize(threshold=100, p=0.2))

    aug_pipeline = transforms.Compose(aug_ops + [transforms.Resize((224,224))])
    img_aug = aug_pipeline(img)
    img_tensor = transforms.ToTensor()(img_aug)
    img_tensor = NORMALIZE(img_tensor)
    img_aug = transforms.ToPILImage()(img_tensor)
    return img_aug

# --- Dataset Creation ---
stats = []
warnings = []

for cls in classes:
    orig_sources = [original_base / 'train' / cls, original_base / 'val' / cls]
    all_orig_images = [f for src in orig_sources if src.exists() for f in src.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png')]
    gen_source = generated_base / cls
    all_gen_images = [f for f in gen_source.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png')] if gen_source.exists() else []

    random.shuffle(all_orig_images)
    random.shuffle(all_gen_images)
    is_target = cls in target_mapping

    # Determine generated images
    if use_all_generated:
        train_gen = all_gen_images
    else:
        actual_gen_count = min(len(all_gen_images), increment)
        train_gen = all_gen_images[:actual_gen_count]

    # Train/val split for originals
    if not is_target:
        train_orig_cap = int(len(all_orig_images) * train_val_ratio)
        train_orig = all_orig_images[:train_orig_cap]
        new_val_set = all_orig_images[train_orig_cap:]
    else:
        start_orig = target_mapping[cls]
        val_cap = int((start_orig / train_val_ratio) - start_orig)
        train_orig = all_orig_images[:start_orig]
        new_val_set = all_orig_images[start_orig:start_orig+val_cap]

    final_train_set = train_orig

    # Copy Originals
    for split_name, image_list in [('train', final_train_set), ('val', new_val_set)]:
        dest_path = output_dir / split_name / cls
        dest_path.mkdir(parents=True, exist_ok=True)
        for img in image_list:
            shutil.copy2(img, dest_path / img.name)

    # Copy Generated
    if (not use_all_generated and increment > 0) or use_all_generated:
        for img_path in train_gen:
            dest_path = output_dir / 'train' / cls
            dest_path.mkdir(parents=True, exist_ok=True)
            try:
                # Open image for augmentation if requested, otherwise direct copy
                if apply_aug:
                    img = Image.open(img_path).convert('RGB')
                    img_aug = random_layered_augment(img, aug_name)
                    img_aug.save(dest_path / img_path.name)
                else:
                    shutil.copy2(img_path, dest_path / img_path.name)
            except Exception as e:
                warnings.append(f"[{cls}] Failed {img_path.name}: {e}")
    else:
        # If increment is 0, ensure train_gen is empty for the stats printout
        train_gen = []

    stats.append({
        'class': cls,
        'total_train': len(final_train_set) + len(train_gen),
        'actual_orig': len(final_train_set),
        'actual_gen': len(train_gen),
        'val_count': len(new_val_set)
    })

# --- Summary ---
print("\n" + "="*85)
print(f"{'Class Name':<30} | {'Total(T)':<8} | {'Orig(T)':<8} | {'Gen(T)':<8} | {'Val(V)':<8}")
print("-"*85)
for s in stats:
    print(f"{s['class'][:30]:<30} | {s['total_train']:<8} | {s['actual_orig']:<8} | {s['actual_gen']:<8} | {s['val_count']:<8}")
print("-"*85)
print(f"RANDOM SEED: {RANDOM_SEED} | INCREMENT: +{increment} Gen Images | USE ALL GENERATED: {use_all_generated} | AUG: {apply_aug} | AUG_NAME: {aug_name}")
print("="*85)
if warnings:
    print("\n--- WARNINGS ---")
    for w in warnings:
        print(w)