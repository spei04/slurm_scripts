#!/usr/bin/env bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH -w beery-a100-1

source /data/vision/beery/scratch/serena/.bashrc
conda activate bpp
CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/slurm_job/generation_qwen_edit.py