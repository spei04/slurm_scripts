#!/usr/bin/env bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --mem=60GB
#SBATCH --time=36:00:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16

source /data/vision/beery/scratch/serena/.bashrc
conda activate bpp
# CUDA_VISIBLE_DEVICES=0 python -u /data/vision/beery/scratch/serena/slurm_job/training.py \
#     --generated-data-dir "/data/vision/beery/scratch/serena/diffusion/generated_full_dataset_qwen_edit"

CUDA_VISIBLE_DEVICES=0 python -u /data/vision/beery/scratch/serena/training_pipeline/varying_ratio.py

CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared --out_dir /data/vision/beery/scratch/serena/training_pipeline/baseline_no_downsample --seed 42
CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/training_pipeline/baseline --seed 42
CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/GBIF_histogram --out_dir /data/vision/beery/scratch/serena/training_pipeline/baseline_histogram --seed 42

CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/diffusion/full_GBIF_qwen_data --out_dir /data/vision/beery/scratch/serena/training_pipeline/curriculum_reverse_qwen --seed 42 --generated_base /data/vision/beery/scratch/serena/diffusion/full_GBIF_qwen_data --reverse
CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/diffusion/full_GBIF_qwen_data --out_dir /data/vision/beery/scratch/serena/training_pipeline/curriculum_forward_qwen --seed 42 --generated_base /data/vision/beery/scratch/serena/diffusion/full_GBIF_qwen_data

CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/full_aug_GBIF_qwen_data --out_dir /data/vision/beery/scratch/serena/training_pipeline/aug_qwen_train --seed 42

CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/filtered_qwen_data --out_dir /data/vision/beery/scratch/serena/training_pipeline/filtered_qwen_train --seed 42
CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/image_quality_GBIF_qwen_data --out_dir /data/vision/beery/scratch/serena/training_pipeline/image_quality_qwen_train --seed 42
CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/morph_fidelity_GBIF_qwen_data --out_dir /data/vision/beery/scratch/serena/training_pipeline/morph_fidelity_qwen_train --seed 42
CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/filtered_aug_qwen_data --out_dir /data/vision/beery/scratch/serena/training_pipeline/filtered_aug_qwen_train --seed 42