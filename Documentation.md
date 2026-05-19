# Relevant scripts
model_eval.ipynb: evaluation and visualizations of models
dreambooth.ipynb: visualizing different settings for dreambooth
generation.ipynb: generating flux and qwen full datasets
filter.ipynb: process LLM filtered data and visualize distributions
dreambooth.sh: submitting script to finetune dreambooth models
dreambooth_inference.py: generating dreambooth images with different settings
train_bjerge_val.py: training script
curriculum.py: curriculum learning code

# Flux and Qwen 
Datasets (original, downsampled, filtered, augmented): /data/vision/beery/scratch/serena/diffusion

# Dreambooth 
Datasets: /data/vision/beery/scratch/serena/dreambooth_new

# Trained models
Model folder (checkpoints + eval results): /data/vision/beery/scratch/serena/0424

# Scripts for training
/data/vision/beery/scratch/serena/training_pipeline

# Submitting training jobs
/data/vision/beery/scratch/serena/slurm_job/training.sh