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


# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train.py \
#   --data_dir "/data/vision/beery/scratch/serena/diffusion/dreambooth_GBIF_data" \
#   --out_dir "/data/vision/beery/scratch/serena/dreambooth_new/dreambooth_b_loss_train" \
#   --seed 42

# CUDA_VISIBLE_DEVICES=0 python -u /data/vision/beery/scratch/serena/slurm_job/training.py \
#     --generated-data-dir "/data/vision/beery/scratch/serena/diffusion/generated_full_dataset_qwen_edit"

# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train.py --data_dir /data/vision/beery/scratch/serena/GBIF_downscaled_new --out_dir /data/vision/beery/scratch/serena/0424/baseline_simone_new --seed 42

########################################################################################
## all new results in 0424 directory

# sister augmentation qwen
# CUDA_VISIBLE_DEVICES=0 python -u /data/vision/beery/scratch/serena/training_pipeline/varying_ratio.py

# sister augmentation flux
# CUDA_VISIBLE_DEVICES=0 python -u /data/vision/beery/scratch/serena/training_pipeline/varying_ratio.py

# sister augmentation dreambooth
CUDA_VISIBLE_DEVICES=0 python -u /data/vision/beery/scratch/serena/training_pipeline/varying_ratio.py

# # baseline no downsample
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared --out_dir /data/vision/beery/scratch/serena/0424/baseline_no_downsample_bjerge_val_f1 --seed 42 --val_data "bjerge" --model_selection "f1"

# # baseline downsample
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/baseline_random_bjerge_val_f1 --seed 42 --val_data "bjerge" --model_selection "f1"

# # baseline histogram
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/GBIF_histogram --out_dir /data/vision/beery/scratch/serena/0424/baseline_histogram_bjerge_val_f1 --seed 42 --val_data "bjerge" --model_selection "f1"


# # QWEN
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/filtered_qwen_data --out_dir /data/vision/beery/scratch/serena/0424/filtered_qwen_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/image_quality_GBIF_qwen_data --out_dir /data/vision/beery/scratch/serena/0424/image_quality_qwen_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/morph_fidelity_GBIF_qwen_data --out_dir /data/vision/beery/scratch/serena/0424/morph_fidelity_qwen_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/filtered_aug_qwen_data --out_dir /data/vision/beery/scratch/serena/0424/filtered_aug_qwen_train --seed 42 --val_data "bjerge" --model_selection "f1"

# # curriculum learning
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_reverse_qwen_bjerge_val_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_qwen_edit --reverse --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_forward_qwen_bjerge_val_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_qwen_edit --val_data "bjerge" --model_selection "f1"


# FLUX
# filter data first /data/vision/beery/scratch/serena/diffusion/filter.ipynb
# create datasets using split creation
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/diffusion/full_GBIF_flux_data --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_flux
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/diffusion/full_aug_GBIF_flux_data --aug --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_flux
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/diffusion/morph_fidelity_GBIF_flux_data --use_all_generated --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_morph_fidelity_flux_filtered
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/diffusion/image_quality_GBIF_flux_data --use_all_generated --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_image_quality_flux_filtered
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/diffusion/filtered_flux_data --use_all_generated --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_flux_filtered

# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/filtered_flux_data --out_dir /data/vision/beery/scratch/serena/0424/filtered_flux_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/image_quality_GBIF_flux_data --out_dir /data/vision/beery/scratch/serena/0424/image_quality_flux_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/morph_fidelity_GBIF_flux_data --out_dir /data/vision/beery/scratch/serena/0424/morph_fidelity_flux_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/diffusion/full_aug_GBIF_flux_data --out_dir /data/vision/beery/scratch/serena/0424/filtered_aug_flux_train --seed 42 --val_data "bjerge" --model_selection "f1"

# # curriculum learning
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_reverse_flux_bjerge_val_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_flux --reverse --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_forward_flux_bjerge_val_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_generated_full_dataset_flux --val_data "bjerge" --model_selection "f1"


# # DREAMBOOTH (Choose 3-7 fixed?)
# call llm_judge_eval.sh
# run llm judge, filter for each metric /data/vision/beery/scratch/serena/diffusion/filter.ipynb
# create datasets using split creation
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/full_GBIF_dreambooth_data --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/full_aug_GBIF_dreambooth_data --aug --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/morph_fidelity_GBIF_dreambooth_data --use_all_generated --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed_morph_fidelity_filtered
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/image_quality_GBIF_dreambooth_data --use_all_generated --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed_image_quality_filtered
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/filtered_dreambooth_data --use_all_generated --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed_filtered

# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/filtered_dreambooth_data --out_dir /data/vision/beery/scratch/serena/0424/filtered_dreambooth_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/image_quality_GBIF_dreambooth_data --out_dir /data/vision/beery/scratch/serena/0424/image_quality_qwen_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/morph_fidelity_GBIF_dreambooth_data --out_dir /data/vision/beery/scratch/serena/0424/morph_fidelity_qwen_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/filtered_aug_dreambooth_data --out_dir /data/vision/beery/scratch/serena/0424/filtered_aug_qwen_train --seed 42 --val_data "bjerge" --model_selection "f1"

# # curriculum learning
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_reverse_dreambooth_3_7_vary_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_vary_prompt --reverse --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_forward_dreambooth_3_7_vary_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_vary_prompt --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_reverse_dreambooth_2_fixed_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_fixed --reverse --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_forward_dreambooth_2_fixed_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_fixed --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_reverse_dreambooth_2_vary_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_vary_prompt --reverse --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_forward_dreambooth_2_vary_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_vary_prompt --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_reverse_dreambooth_3_7_fixed_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed --reverse --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_forward_dreambooth_3_7_fixed_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_reverse_dreambooth_8_fixed_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_fixed --reverse --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_forward_dreambooth_8_fixed_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_fixed --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_reverse_dreambooth_8_vary_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_vary_prompt --reverse --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/curriculum.py --data_dir /data/vision/beery/scratch/serena/GBIF_prepared_downsampled --out_dir /data/vision/beery/scratch/serena/0424/curriculum_forward_dreambooth_8_vary_bjerge_f1 --seed 42 --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_vary_prompt --val_data "bjerge" --model_selection "f1"

# dreambooth cfg 2
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_GBIF_data --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_fixed
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_GBIF_data --out_dir /data/vision/beery/scratch/serena/0424/dreambooth_cfg_2_bjerge_f1 --seed 42 --val_data "bjerge" --model_selection "f1"
# # dreambooth cfg 8
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_GBIF_data --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_fixed
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_GBIF_data --out_dir /data/vision/beery/scratch/serena/0424/dreambooth_cfg_8_bjerge_f1 --seed 42 --val_data "bjerge" --model_selection "f1"
# # dreambooth cfg 3-7
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_data --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_data --out_dir /data/vision/beery/scratch/serena/0424/dreambooth_cfg_3_7_bjerge_f1 --seed 42 --val_data "bjerge" --model_selection "f1"

# # dreambooth cfg 2 with prompt variation
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_vary_prompt_GBIF_data --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_vary_prompt
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_2_vary_prompt_GBIF_data --out_dir /data/vision/beery/scratch/serena/0424/dreambooth_cfg_2_vary_prompt_bjerge_f1 --seed 42 --val_data "bjerge" --model_selection "f1"

# # dreambooth cfg 8 with prompt variation
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_vary_prompt_GBIF_data --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_vary_prompt
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_8_vary_prompt_GBIF_data --out_dir /data/vision/beery/scratch/serena/0424/dreambooth_cfg_8_vary_prompt_bjerge_f1 --seed 42 --val_data "bjerge" --model_selection "f1"

# # dreambooth cfg 3-7 with prompt variation
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_vary_prompt_GBIF_data --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_vary_prompt
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_vary_prompt_GBIF_data --out_dir /data/vision/beery/scratch/serena/0424/dreambooth_cfg_3_7_vary_prompt_bjerge_f1 --seed 42 --val_data "bjerge" --model_selection "f1"

# test dreambooth cfg 3-5 fixed prompt
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_full_GBIF_data --generated_base /data/vision/beery/scratch/serena/diffusion/downsampled_dreambooth_full
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_full_GBIF_data --out_dir /data/vision/beery/scratch/serena/0424/dreambooth_full_bjerge_f1 --seed 42 --val_data "bjerge" --model_selection "f1"


# filtered 3-7 fixed prompt
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_fixed_filtered --use_all_generated --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed_filtered
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_fixed_image_quality_filtered --use_all_generated --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed_image_quality_filtered
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_fixed_morph_fidelity_filtered --use_all_generated --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed_morph_fidelity_filtered
# python /data/vision/beery/scratch/serena/training_pipeline/split_creation.py --increment 1000 --out_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_fixed_filtered_aug --use_all_generated --aug --generated_base /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_fixed_filtered
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_fixed_filtered --out_dir /data/vision/beery/scratch/serena/0424/filtered_dreambooth_3_7_fixed_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_fixed_image_quality_filtered --out_dir /data/vision/beery/scratch/serena/0424/image_quality_dreambooth_3_7_fixed_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_fixed_morph_fidelity_filtered --out_dir /data/vision/beery/scratch/serena/0424/morph_fidelity_dreambooth_3_7_fixed_train --seed 42 --val_data "bjerge" --model_selection "f1"
# CUDA_VISIBLE_DEVICES=0 python /data/vision/beery/scratch/serena/training_pipeline/train_bjerge_val.py --data_dir /data/vision/beery/scratch/serena/dreambooth_new/dreambooth_cfg_3_7_GBIF_fixed_filtered_aug --out_dir /data/vision/beery/scratch/serena/0424/filtered_aug_flux_train --seed 42 --val_data "bjerge" --model_selection "f1"


## TODO:
# reran the filter pipeline for flux
# running filter pipeline for dreambooth cfg 3-7 