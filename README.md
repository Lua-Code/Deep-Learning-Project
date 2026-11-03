# From Pixels to Language

**CSE 485 / CSE 488: Deep Learning, Final Project**
Ain Shams University, Faculty of Engineering
Instructors: Dr. Mahmoud Khalil, Eng. Seif Yasser

An end-to-end multimodal deep learning system on the CUB-200-2011 bird dataset. One shared CNN encoder powers three tasks:

1. **Classification:** identify the bird species (200 classes).
2. **Colorization:** turn a grayscale bird photo into a color image.
3. **Captioning:** generate a natural-language description of a bird photo.

| Milestone | Scope | Status |
|---|---|---|
| 1 (1 Nov) | EDA, classification, encoder training, tuning, Grad-CAM | Done |
| 2 (18 Dec) | Colorization and captioning | In progress |

---

## Table of contents

1. [Team](#1-team)
2. [Project overview](#2-project-overview)
3. [Dataset](#3-dataset)
4. [Repository structure](#4-repository-structure)
5. [Environment setup](#5-environment-setup)
6. [Data preparation](#6-data-preparation)
7. [Usage (CLI)](#7-usage-cli)
8. [Configuration files](#8-configuration-files)
9. [Experiment tracking](#9-experiment-tracking)
10. [Hyperparameter tuning](#10-hyperparameter-tuning)
11. [Explainability (Grad-CAM)](#11-explainability-grad-cam)
12. [Evaluation metrics](#12-evaluation-metrics)
13. [Results](#13-results)
14. [Trained models](#14-trained-models)
15. [Reproducibility](#15-reproducibility)



---

## 1. Team

**Team number:** 17

| Member | ID |
|---| ---|
| Kevin David | 23P0160
| Mahmoud Hassan | 23P0104
| Jana Tamer | 23P0173
| Ahmed Meligy | 23P0112
| Madonna Emad | 23P0325
| Jana Mohammed | 2301032


---

## 2. Project overview

### 2.1 Pipeline

```
                        ┌─────────────────────────┐
  CUB images ──► EDA ──►│  Shared CNN encoder     │
  (+ captions)          │  (scratch or pretrained)│
                        └───────┬───────┬─────────┘
                                │       │        │
                   Classifier   │   U-Net decoder│   Caption decoder
                   (200 species)│   (colorization)   (LSTM/Transformer)
```

### 2.2 Tasks

**Task 1: Classification and encoder training.**
- A ResNet-style CNN trained **from scratch** as the baseline.
- **Transfer learning** with pretrained backbones (<ResNet-50 / EfficientNet-B0 / ...>), frozen-head stage then full fine-tuning.
- Hyperparameter tuning with **grid search and random search**, compared.
- Grad-CAM explanations for correct and wrong predictions.

**Task 2: Colorization.**
- The trained encoder is the contracting path of a U-Net.
- Works in **LAB color space**: the model receives L (lightness) and predicts the a and b channels.
- Loss: <L1 / L1 + perceptual>. Metrics: PSNR, MAE, <SSIM>.

**Task 3: Captioning.**
- The encoder is **frozen**; its spatial feature maps feed an autoregressive decoder (<LSTM with attention / Transformer>).
- Decoding: greedy and beam search. Metrics: BLEU-1..4, ROUGE-L.

### 2.3 Key design decisions

- **One encoder interface.** Every encoder returns multi-scale feature maps, so classifier, colorizer and captioner accept any encoder. Swapping scratch for pretrained is a config change.
- **Lazy image loading.** Images are decoded on demand and augmented on the fly. Nothing is pre-resized or stored, so the aspect ratio is preserved and augmentation stays random.
- **Stratified validation split** from the official train set. The official test set is never used for tuning.
- **Shared trainer base class.** Mixed precision, logging, checkpointing and resume live in one place; each task only defines its step and validation logic.

---

## 3. Dataset

**Caltech-UCSD Birds-200-2011 (CUB-200-2011)** with captions (Reed et al., 2016).

| Property | Value |
|---|---|
| Species | 200 |
| Total images | 11,788 (about 1.1 GB) |
| Train / test Splot | 5,994 / 5,794 |
| Captions per image | 10 (human-written) |
| Source | Hugging Face: `Multimodal-Fatima/CUB_train`, `Multimodal-Fatima/CUB_test` |
| Homepage | https://www.vision.caltech.edu/datasets/cub_200_2011/ |

Fields: `image`, `label`, `description`, `file_name`, `id`.

**Splits used in this project**

| Split | Source | Use |
|---|---|---|
| Train | official train, stratified | training |
| Validation | official train, stratified (default 15%, seed 42) | tuning, early stopping, model selection |
| Test | official test | final reporting only |

**Colorization data.** The dataset has no gray/color pairs; they are built on the fly. The original image is the target and its lightness channel is the input.

**Captions.** The `description` string is split into ten captions and cleaned (<list your steps: lowercase, punctuation, typo handling, length cap>). See `src/data/caption_utils.py`.

The dataset downloads automatically on first use. It is **not** stored in this repository.

---

## 4. Repository structure

```
project/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── cls_scratch.yaml        # scratch CNN classification run
│   ├── cls_transfer.yaml       # pretrained classification run
│   ├── colorization.yaml
│   ├── caption.yaml
│   └── tuning/
│       └── cls_search.yaml     # grid / random search spaces
├── data/                       # gitignored
│   ├── raw/                    # optional Hugging Face cache
│   └── processed/              # cleaned captions, vocab, split indices, cached features
├── notebooks/                  # experiments only (not part of the final code)
├── src/
│   ├── cli.py                  # command line entry point
│   ├── config.py               # YAML loading
│   ├── data/
│   │   ├── cub_dataset.py      # loading, label mapping, stratified split, Dataset, DataLoader
│   │   ├── transforms.py       # train / eval transforms, denormalize
│   │   ├── caption_utils.py    # caption splitting, cleaning, vocab, tokenizer
│   │   ├── colorization_dataset.py
│   │   └── eda.py              # exploratory data analysis
│   ├── models/
│   │   ├── encoders/           # base.py, scratch_cnn.py, pretrained.py, registry.py
│   │   ├── classifier.py
│   │   ├── colorizer.py        # U-Net on the encoder
│   │   └── caption_decoder.py
│   ├── training/
│   │   ├── base_trainer.py
│   │   ├── cls_trainer.py
│   │   ├── color_trainer.py
│   │   ├── caption_trainer.py
│   │   └── tuning.py           # grid + random (+ Optuna)
│   ├── evaluation/
│   │   ├── cls_metrics.py
│   │   ├── color_metrics.py
│   │   ├── caption_metrics.py
│   │   └── benchmark.py
│   ├── visualization/
│   │   ├── plots.py
│   │   └── gradcam.py
│   └── utils/                  # seed.py, checkpoint.py, logging.py
├── outputs/                    # figures and logs (gitignored if large)
├── checkpoints/                # gitignored; download from the Drive link below
└── report/                     # LaTeX source and figures
```

Notebooks are used only for trying ideas. All final code is plain Python run through the CLI.

---

## 5. Environment setup

Requirements: Python <3.10+>, an NVIDIA GPU is recommended (a free Colab/Kaggle T4 is enough). About 3 GB free disk for the dataset cache.

### 5.1 Local machine

```bash
git clone https://github.com/Lua-Code/Deep-Learning-Project
cd project

# create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Windows CMD:         .venv\Scripts\activate.bat
# macOS / Linux:       source .venv/bin/activate

python -m pip install --upgrade pip
```

For an NVIDIA GPU, install the matching PyTorch build first from https://pytorch.org/get-started/locally/, then:

```bash
pip install -r requirements.txt
```

Without a GPU you can install everything from `requirements.txt` directly; training will be slow.

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 5.2 Google Colab

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone <repo-url>
%cd project
!pip install -r requirements.txt   # torch and torchvision are preinstalled
```

Point the dataset cache at Drive so it downloads only once (see section 6) and save checkpoints under your Drive folder, because the Colab disk is wiped when the session ends.

### 5.3 Kaggle

Enable **Internet** and select a **T4 GPU** in the notebook settings, then clone and install as above. Write outputs to `/kaggle/working`.

### 5.4 Main dependencies

| Purpose | Package |
|---|---|
| Deep learning | `torch`, `torchvision`, `timm` |
| Data | `datasets`, `pillow`, `numpy`, `pandas`, `scikit-learn` |
| Visualization | `matplotlib`, `seaborn` |
| Tracking | `mlflow` <or `wandb`> |
| Tuning | `scikit-learn`, `optuna` (optional) |
| Metrics | `scikit-image`, `nltk`/`sacrebleu`, `rouge-score` |
| Explainability | `grad-cam` (or the in-repo implementation) |
| Config and utilities | `pyyaml`, `tqdm` |

Full pinned versions are in `requirements.txt`.

---

## 6. Data preparation

The dataset is fetched from Hugging Face on first use. To control where it is cached:

```bash
# Linux / macOS
export HF_HOME=/path/to/cache
# Windows PowerShell
$env:HF_HOME = "D:\hf_cache"
```

Build the processed artifacts (cleaned captions, vocabulary, split indices):

```bash
python -m src.cli prepare-data
```

Quick data check (prints class, split and sample information):

```bash
python -m src.data.cub_dataset
```

Run the exploratory analysis (class distribution, image sizes, caption statistics, duplicate/corrupt checks, augmentation examples). Figures are written to `outputs/eda/`:

```bash
python -m src.data.eda
```

---

## 7. Usage (CLI)

All tasks are run through `src/cli.py`:

```bash
python -m src.cli <command> --task <task> [options]
```

| Command | Purpose |
|---|---|
| `prepare-data` | build cleaned captions, vocab and split indices |
| `train` | train a model for a task |
| `eval` | evaluate a checkpoint on validation or test |
| `predict` | run inference on a single image |
| `tune` | grid / random hyperparameter search |
| `gradcam` | generate Grad-CAM visualizations |

Tasks: `classification`, `colorization`, `caption`. Run `python -m src.cli --help` or `python -m src.cli train --help` for the full option list.

### 7.1 Classification

Train the scratch baseline:

```bash
python -m src.cli train --task classification --config configs/cls_scratch.yaml
```

Train with transfer learning:

```bash
python -m src.cli train --task classification --config configs/cls_transfer.yaml
```

Resume an interrupted run:

```bash
python -m src.cli train --task classification --config configs/cls_transfer.yaml --resume checkpoints/<run>/last.pt
```

Evaluate on the test set:

```bash
python -m src.cli eval --task classification --config configs/cls_transfer.yaml --checkpoint checkpoints/<run>/best.pt --split test
```

Predict a single image (prints the top-5 species with probabilities):

```bash
python -m src.cli predict --task classification --checkpoint checkpoints/<run>/best.pt --image path/to/bird.jpg
```

### 7.2 Hyperparameter tuning

```bash
python -m src.cli tune --task classification --method grid
python -m src.cli tune --task classification --method random
```

### 7.3 Grad-CAM

```bash
python -m src.cli gradcam --task classification --checkpoint checkpoints/<run>/best.pt --split test --num-images 16 --out outputs/gradcam
```

### 7.4 Colorization (Milestone 2)

```bash
python -m src.cli train --task colorization --config configs/colorization.yaml
python -m src.cli eval --task colorization --config configs/colorization.yaml --checkpoint checkpoints/<run>/best.pt --split test
python -m src.cli predict --task colorization --checkpoint checkpoints/<run>/best.pt --image path/to/bird.jpg --out outputs/colorized.png
```

### 7.5 Captioning (Milestone 2)

```bash
python -m src.cli train --task caption --config configs/caption.yaml
python -m src.cli eval --task caption --config configs/caption.yaml --checkpoint checkpoints/<run>/best.pt --split test
python -m src.cli predict --task caption --checkpoint checkpoints/<run>/best.pt --image path/to/bird.jpg --beam-size 3
```

### 7.6 Recommended order to reproduce everything

```bash
python -m src.cli prepare-data
python -m src.data.eda
python -m src.cli train --task classification --config configs/cls_scratch.yaml
python -m src.cli train --task classification --config configs/cls_transfer.yaml
python -m src.cli tune --task classification --method grid
python -m src.cli tune --task classification --method random
python -m src.cli gradcam --task classification --checkpoint checkpoints/cls_scratch/best.pt 
```

---

## 8. Configuration files

Every run is described by a YAML file in `configs/`. Command line options override values in the file.

Example (`configs/cls_transfer.yaml`):

```yaml
seed: 42
data:
  val_fraction: 0.15
  image_size: 224
  batch_size: 32
  num_workers: 2
  cache_dir: null
model:
  encoder: resnet50          # scratch_cnn | resnet50 | efficientnet_b0 | ...
  pretrained: true
  dropout: 0.3
train:
  epochs: 30
  lr: 0.001
  backbone_lr_factor: 0.1
  weight_decay: 0.0001
  label_smoothing: 0.1
  freeze_epochs: 3
  amp: true
  early_stopping_patience: 7
tracking:
  backend: mlflow
  experiment: classification
```

<Replace with your real keys.>

---

## 9. Experiment tracking

Tool: **<MLflow / Weights & Biases>**. Every run logs hyperparameters, train/validation loss and metric curves, and checkpoint references.

```bash
# MLflow
mlflow ui --port 5000
# then open http://localhost:5000
```

Runs are grouped by experiment name (`classification`, `tuning`, `colorization`, `caption`). Each run name encodes the encoder and key hyperparameters so any run can be compared against the others. Exports and screenshots are in `outputs/tracking/` and in the report.

---

## 10. Hyperparameter tuning

- **Tuned parameters:** <learning rate, weight decay, batch size, augmentation strength>.
- **Methods:** grid search and random search with matching budgets (same number of trials and epochs).
- **Comparison:** best validation score reached, number of trials, and total time for each method. Results are in `outputs/tuning/` and section 13.
- **Search spaces:** `configs/tuning/cls_search.yaml`.
- Optuna is optional and <used / not used>.

---

## 11. Explainability (Grad-CAM)

Grad-CAM highlights the image regions that most influenced a prediction. We generate heatmaps for both **correct** and **misclassified** test images and look for patterns (for example, the model attending to the background instead of the bird). Outputs are saved to `outputs/gradcam/`, and the analysis is in the report.

---

## 12. Evaluation metrics

| Task | Metric | What it measures |
|---|---|---|
| Classification | Accuracy (top-1, top-5) | fraction of images with the right species |
| | Precision / Recall / F1 (macro) | per-class quality averaged over species; useful because classes are small |
| | Confusion matrix | which species get confused with which |
| Colorization | PSNR | pixel-level fidelity in RGB (higher is better) |
| | MAE | mean absolute error in RGB |
| | SSIM <optional> | structural similarity |
| Captioning | BLEU-1..4 | n-gram overlap with the reference captions |
| | ROUGE-L | longest common subsequence overlap |

Colorization metrics are computed in RGB [0, 1] after converting the prediction back from LAB. Captioning is scored against all ten references per image.

---

## 13. Results


### 13.1 Classification (test set)

| Model | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|
| Scratch CNN | 0.54 | 0.77 | 0.53 | 
| <ResNet-50, transfer> | 0.77 | 0.94 | 0.77 | 

### 13.2 Tuning comparison (Scratch CNN)

| Method | Trials | Best val accuracy | Total time |
|---|---|---|---|
| Grid search | 12 | 0.694 | 28.5 |
| Random search | 12 | 0.685 | 28.5 |

### 13.3 Colorization (TBA)

| Backbone | PSNR | MAE | SSIM |
|---|---|---|---|
| <Scratch encoder> | TBA | TBA | TBA |
| <Pretrained encoder> | TBA | TBA | TBA |

### 13.4 Captioning (TBA)

| Decoder | BLEU-1 | BLEU-4 | ROUGE-L |
|---|---|---|---|
| <LSTM + attention> | TBA | TBA | TBA |
| <Transformer> | TBA | TBA | TBA |

### 13.5 Final selected models

| Task | Model | Why |
|---|---|---|
| Classification | Resnet50 Transfer | Resnet50 is way better as it is already trained on millions of images and its hyperparams are well tuned |
| Colorization | TBA | TBA |
| Captioning | TBA | TBA |

---

## 14. Trained models

All trained checkpoints: **https://drive.google.com/drive/folders/154goXbIDzcxHnKMLR9e9KCz1AepWgXgz?usp=sharing**

| File | Task | Description |
|---|---|---|
| `cls_scratch_best.pt` | classification | scratch CNN |
| `cls_transfer_best.pt` | classification | pretrained, fine-tuned |
| `encoder_backbone.pt` | shared | encoder weights only (no head) |
| `colorizer_best.pt` | colorization | U-Net |
| `caption_best.pt` | captioning | decoder (+ vocab) |

Download them into `checkpoints/` and use the paths in the commands above.

---

## 15. Reproducibility

- Seeds are fixed (default `42`) for Python, NumPy and PyTorch, and for the train/validation split and shuffling.
- Package versions are pinned in `requirements.txt`.
- Every run saves its resolved config next to its checkpoint.
- GPU nondeterminism (cuDNN, mixed precision) can cause small differences between runs.
- To reproduce a result, use the same config file and seed (section 7.6).

---
