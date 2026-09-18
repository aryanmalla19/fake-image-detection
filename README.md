# Deepfake Image Detection with a Hybrid CNN-Transformer

This repository implements a binary real/fake image classifier for the CIFAKE dataset using a hybrid CNN-Transformer architecture and a controlled CNN baseline.

The workflow retains configurations, split indices, environment details, checkpoints, metrics and plots for coursework evidence.

## Project Goal

Build and evaluate a deepfake image detector that combines:

- a CNN branch for local texture and artifact features;
- a transformer encoder for global relationships between image regions;
- a fusion classifier for final real/fake prediction.

## Expected Dataset

Use the CIFAKE dataset with this folder layout:

```text
data/cifake/
  train/
    REAL/
    FAKE/
  test/
    REAL/
    FAKE/
```

The proposal expects 100,000 training images and 20,000 test images, balanced evenly between `REAL` and `FAKE`.

Source: [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), dataset identifier `birdy654/cifake-real-and-ai-generated-synthetic-images`. `torchvision.datasets.ImageFolder` maps classes alphabetically, so `FAKE=0` and `REAL=1`; evaluation explicitly treats `FAKE` as the positive class.

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --config configs/cifake_hybrid.yaml
```

The controlled experiment configurations are:

- `configs/cifake_cnn.yaml`: CNN baseline.
- `configs/cifake_hybrid.yaml`: proposed hybrid model.
- `configs/cifake_hybrid_no_augmentation.yaml`: augmentation ablation.

Outputs are written to `runs/cifake_hybrid/` by default:

- `best_model.pt`
- `history.csv`
- training logs
- the exact configuration, split indices and environment metadata;
- training curves and run summary.

## Evaluate

```bash
python -m src.evaluate --config configs/cifake_hybrid.yaml --checkpoint runs/cifake_hybrid/best_model.pt
```

Evaluation produces:

- accuracy, precision, recall, F1, ROC-AUC;
- confusion matrix;
- ROC curve;
- precision-recall curve;
- `metrics.json`.

## Dataset Audit

Before training, validate counts, dimensions, colour modes, corrupt files and exact train/test duplicates:

```bash
python -m src.check_dataset --root data/cifake
```

## Compare Experiments

After training and evaluating all three models:

```bash
python -m src.compare runs/cifake_cnn runs/cifake_hybrid runs/cifake_hybrid_no_augmentation
```

## Tests

```bash
python -m unittest discover -s tests -v
```

For Google Colab, use `notebooks/cifake_hybrid_colab.ipynb`; it downloads the dataset, runs the audit and tests, executes the experiments, and preserves `runs/` in Google Drive.

## Implementation Plan

1. Prepare CIFAKE in the expected layout.
2. Run a dataset sanity check to confirm image counts and class balance.
3. Train the baseline hybrid CNN-Transformer model.
4. Evaluate on the held-out test set.
5. Replace proposal mock figures with real result plots.
6. Compare against the CNN-only baseline and no-augmentation ablation.

## Notes

CIFAKE images are natively 32x32. The default model keeps this resolution instead of upsampling to ImageNet size, which makes training cheaper and fits the proposal's local/global feature idea.

CIFAKE represents a particular synthetic-image generation setup. Strong in-dataset performance does not establish generalisation to newer generators, compressed social-media images or real-world deepfakes.
