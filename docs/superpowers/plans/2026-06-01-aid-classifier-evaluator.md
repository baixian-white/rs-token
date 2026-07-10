# AID Classifier Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build E16: a fixed AID image classifier evaluator trained on this project's AID split for external baseline task comparisons.

**Architecture:** Add one focused training script and one YAML config. The evaluator is an ImageNet-pretrained ResNet34 fine-tuned on `data/AID_splits/train.csv`, selected on val accuracy, and evaluated once on test.

**Tech Stack:** Python, PyTorch, torchvision, YAML, existing `models.datasets` AID loader.

---

### Task 1: Add E16 Config

**Files:**
- Create: `rstoken/configs/aid_classifier_resnet34.yaml`

- [ ] **Step 1: Create a config with explicit paths and training defaults**

Use ImageNet-pretrained ResNet34 by default, 30 classes, batch size 32, 30 epochs, AMP enabled, and output paths under `checkpoints/aid_classifier_resnet34` and `logs/aid_classifier_resnet34`.

### Task 2: Add Training Script

**Files:**
- Create: `rstoken/scripts/08_train_aid_classifier.py`

- [ ] **Step 1: Load config and set seeds**

Reuse YAML loading and seed style from `scripts/03_train_vqvae.py`.

- [ ] **Step 2: Build ImageNet-normalized AID dataloaders**

Use the same CSV split files, but classifier transforms should use ImageNet normalization rather than `[-1, 1]`.

- [ ] **Step 3: Build ResNet34 classifier**

Use `torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)` when configured. Replace the final FC layer with `num_classes=30`.

- [ ] **Step 4: Train, validate, save best**

Optimize cross-entropy. Save `best.pt`, `last.pt`, `metrics.csv`, and `test_metrics.json`.

- [ ] **Step 5: Support smoke mode**

`--smoke` runs two train batches and one val/test batch to verify the full pipeline.

### Task 3: Run Verification

**Files:**
- Modify: `rstoken/experiments/pre_experiments_log.md`

- [ ] **Step 1: Run smoke**

Command:

```powershell
$py = "C:\Users\Windows11\.conda\envs\rstoken\python.exe"
& $py -X utf8 scripts/08_train_aid_classifier.py --config configs/aid_classifier_resnet34.yaml --smoke
```

Expected: script prints train/val/test batch counts, runs without exceptions, and writes smoke artifacts.

- [ ] **Step 2: Run full training**

Use the same command without `--smoke`. If ImageNet weights are not cached and download fails under sandbox/network limits, rerun with approved network access.

- [ ] **Step 3: Update E16 log**

Record whether pretrained weights were used, clean val/test top-1, macro-F1, worst-class accuracy, and artifact paths.
