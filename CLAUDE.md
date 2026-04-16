# VeReMi Misbehavior Detection — ML Project

## Project Overview
**Title:** Misbehavior Detection in Vehicle-to-Everything (V2X) Networks Using the VeReMi Extension Dataset  
**Student:** Akid Abrar, Civil, Construction and Environmental Engineering, Graduate (Ph.D.)

## Goal
Supervised **20-class classification** of Basic Safety Messages (BSMs): one benign class and 19 distinct misbehavior/attack types (e.g., Constant Position, DoS, Sybil attacks).

## Dataset
- **File:** `mixalldata_clean.csv` (1.21 GB, ~3.19 million rows, 30 columns)
- **Target column:** `class` (int 0–19)
- **Class imbalance:** class 0 (Normal) has ~1.9M rows; attack classes ~42K–175K each

### Feature columns (after removing static/non-feature columns)
Keep: `posx, posy, posx_n, posy_n, spdx, spdy, spdx_n, spdy_n, aclx, acly, aclx_n, acly_n, hedx, hedy, hedx_n, hedy_n`  
Drop: `type` (constant=4), `posz/posz_n/spdz/spdz_n/aclz/aclz_n/hedz/hedz_n` (all zeros), `sendTime, sender, senderPseudo, messageID` (non-kinematic identifiers)

### Class Map
| ID | Name | Category |
|----|------|----------|
| 0 | Normal behavior | Normal |
| 1 | Constant position | Fault |
| 2 | Constant position offset | Fault |
| 3 | Random position | Fault |
| 4 | Random position offset | Fault |
| 5 | Constant speed | Fault |
| 6 | Constant speed offset | Fault |
| 7 | Random speed | Fault |
| 8 | Random speed offset | Fault |
| 9 | Disruptive | Attack |
| 10 | Data replay | Attack |
| 11 | DoS | Attack |
| 12 | DoS random | Attack |
| 13 | DoS disruptive | Attack |
| 14 | Data replay sybil | Attack |
| 15 | Traffic congestion sybil | Attack |
| 16 | DoS random sybil | Attack |
| 17 | DoS disruptive sybil | Attack |
| 18 | Extended Attack A | Attack |
| 19 | Extended Attack B | Attack |

## Required Models (from proposal)
1. **Random Forest** (primary) — class_weight='balanced', feature importance
2. **XGBoost** — gradient boosting with regularization
3. **k-Nearest Neighbors (k-NN)** — baseline, no distributional assumptions

## Evaluation Metrics
- **Primary:** Macro-averaged F1-score (equal weight across all 20 classes)
- Also: Overall accuracy, per-class precision/recall/F1, confusion matrix

## Graduate Requirement: Uncertainty Quantification
1. RF class-probability outputs (fraction of trees voting for each class)
2. 5-fold stratified cross-validation → mean ± std of macro-F1
3. XGBoost calibration curves (predicted probability vs. observed frequency)

## Reference Notebooks (do NOT modify)
- `veremi-simulated-vehicle-attack-detection.ipynb` — EDA + LSTM binary detection (Kaggle)
- `wn-fl-project.ipynb` — Async Federated Learning BiLSTM/GRU (Kaggle, teammate collab)

## Main Project Notebook
- `veremi_misbehavior_detection.ipynb` — **THE deliverable** implementing the proposal

## Implementation Notes
- Dataset is 1.21 GB; use stratified sampling (~300K rows) for tractable training on local CPU
- For k-NN, reduce sample further (~50K) to avoid excessive memory/time
- Use `class_weight='balanced'` or stratified sampling to handle imbalance
- Random seed: 42 throughout for reproducibility
