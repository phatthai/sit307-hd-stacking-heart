# Reproducing and Extending a Stacking Ensemble for Heart Disease Prediction

SIT307/SIT720 Machine Learning, Task 11.1HD (Machine Learning Mini Research)
Author: Tien Phat Thai (225213834), Deakin University

This repository reproduces and extends:

> M. Bhagat, A. Sharma and P. Agarwal, "An efficient stacking-based ensemble technique for early heart attack prediction," *Multimedia Tools and Applications*, vol. 84, pp. 36351-36375, 2025, doi: 10.1007/s11042-024-19293-7.

* **Part 1 (reproduction):** the six base classifiers (LR, DT, RF, XGBoost, Naive Bayes, KNN) and the 5-fold stacking ensemble, evaluated with the paper's protocol (80/20 record-level split, Accuracy, Precision, Recall, F1, AUC), a recovered train/test split, 100 random splits, an audit of the paper's reported metrics and a data leakage analysis.
* **Part 2 (proposed solution):** LA-DPS, Leakage-Aware Diversity-Pruned Stacking, evaluated with patient-level repeated stratified 5-fold cross-validation (10 repeats) on the original 1025-row file, with an ablation study and paired statistical tests.

## Repository structure

```
.
├── SIT307_11.1HD_Stacking_Reproduction_and_LA-DPS.ipynb   # all code, executed, outputs visible
├── data/heart.csv          # dataset used by the paper (1025 records)
├── results/                # CSV tables written by the notebook
├── figures/                # PNG figures written by the notebook
├── requirements.txt
└── README.md
```

## Dataset

`data/heart.csv` is the Kaggle "Heart Disease Dataset" (D. Lapp, https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), derived from the UCI Heart Disease database (Janosi, Steinbrunn, Pfisterer and Detrano, doi: 10.24432/C52P4X). It has 1025 records, 13 predictors and a binary `target` (1 = disease). The notebook checks the file with a content hash (`8726740167315441832`) so the exact file used in the report is confirmed.

Important property found in this study: the file contains only **302 unique records**; 723 records are exact duplicates (every unique record appears 3 to 8 times).

## How to run

### Option A: Google Colab (recommended)
1. Open https://colab.research.google.com, choose **File > Open notebook > GitHub**, and paste this repository's URL.
2. Select the notebook and choose **Runtime > Run all**.
3. The first cell pins `scikit-learn==1.8.0` and `xgboost==3.4.1`; the dataset is downloaded automatically from this repository.

### Option B: Local
```bash
git clone https://github.com/phatthai/sit307-hd-stacking-heart.git
cd sit307-hd-stacking-heart
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook SIT307_11.1HD_Stacking_Reproduction_and_LA-DPS.ipynb   # then Run All
```

Expected runtime: about 20 minutes on Google Colab (2 CPUs). The notebook writes every table to `results/` and every figure to `figures/`. Expensive loops are checkpointed in `results/cache/` (ignored by git), so a disconnected session resumes where it stopped; delete that folder to recompute everything from scratch.

## Reproducibility

* All randomness is seeded (model seed 42; split seeds and CV seeds are fixed in the notebook). Parallel execution (`N_JOBS`) does not change any result.
* Results in the report were produced on Google Colab with Python 3.13, scikit-learn 1.8.0, xgboost 3.4.1, numpy 2.1, pandas 2.2, scipy 1.16 and matplotlib 3.10; an independent run with Python 3.12, numpy 2.4, pandas 3.0 and scipy 1.17 produced identical outputs. Other versions of scikit-learn or xgboost can change the numbers slightly; the notebook prints the versions it ran with.
* Preprocessing is fitted inside each training fold in all leakage-free experiments.

## Mapping between the report and the outputs

| Report item | Output file |
| --- | --- |
| Data audit (duplicates, invalid codes) | `results/data_audit.csv`, `figures/fig_duplicates.png` |
| Recovered split | `results/split_recovery.csv` |
| Reproduced results vs paper | `results/reproduction_vs_paper.csv`, `figures/fig_repro_cm_roc.png` |
| Paper protocol over 100 splits | `results/paper_protocol_summary.csv` |
| Audit of the paper's Table 11 | `results/audit_table11.csv` |
| Capacity sensitivity | `results/sensitivity_capacity.csv`, `figures/fig_sensitivity.png` |
| Optimism gap and meta-learner weights | `results/optimism_gap.csv`, `figures/fig_optimism_and_weights.png` |
| Part 2 results and ablation | `results/part2_summary.csv`, `figures/fig_part2_boxplots.png` |
| Statistical tests | `results/part2_stat_tests.csv` |
| LA-DPS member selection | `results/part2_selection_frequency.csv`, `figures/fig_selection.png` |
| Final comparison | `results/summary_comparison.csv` |
