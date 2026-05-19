# Diffusion Model Training & Evaluation Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![Status](https://img.shields.io/badge/status-research--project-lightgrey.svg)
![License](https://img.shields.io/badge/license-proprietary-red.svg)

This repository contains the full pipeline for dataset generation, filtering, training, and evaluation of diffusion-based models (Flux, Qwen, and DreamBooth) under multiple augmentation and curriculum learning strategies.

---

# 📁 Repository Structure

## 🔷 Flux and Qwen

Used for dataset generation and baseline diffusion experiments.

- **Datasets (original, downsampled, filtered, augmented):**  
  `/data/vision/beery/scratch/serena/diffusion`

- **Generation pipeline:**  
  `diffusion/generation.ipynb`

---

## 🔶 DreamBooth

Fine-tuning and evaluation for DreamBooth-based experiments.

- **Datasets:**  
  `/data/vision/beery/scratch/serena/dreambooth_new`

- **Diffusers training code:**  
  `/data/vision/beery/scratch/serena/diffusers/examples/dreambooth`

- **Visualization notebook:**  
  `/data/vision/beery/scratch/serena/insect_analysis/organized/dreambooth.ipynb`

- **Training submission script:**  
  `slurm_job/dreambooth.sh`

- **Inference (image generation):**  
  `slurm_job/dreambooth_inference.py`  
  Submit via: `slurm_job/dreambooth_inference.sh`

---

## 🧹 Filtering Pipeline

LLM-based dataset filtering and quality control.

- **Filtering logic:**  
  `slurm_job/llm_judge_old.py`

- **Job submission script:**  
  `slurm_job/llm_judge.sh`

---

## ⚙️ Training Pipeline

Core training infrastructure for all experiments.

**Location:**  
`training_pipeline/`

### Key Components

- `split_creation.py` — merges generated images with GBIF dataset
- `varying_ratio.py` — Siblings Augmentation implementation
- `train_bjerge_val.py` — main training script
- `curriculum.py` — curriculum learning implementation

- **Training submission script:**  
  `slurm_job/training.sh`

---

## 📊 Model Evaluation

- **Evaluation + visualization notebook:**  
  `insect_analysis/organized/model_eval.ipynb`

---

## 🧠 Trained Models

- **Checkpoints + evaluation results:**  
  `/data/vision/beery/scratch/serena/0424`

---

# 🚀 End-to-End Workflow

1. Generate datasets (Flux / Qwen)
2. Filter datasets using LLM judge
3. Construct training splits
4. Train models (standard / curriculum / augmentation)
5. Run inference
6. Evaluate results in notebooks
