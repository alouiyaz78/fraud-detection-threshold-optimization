

# Credit Card Fraud Detection — Imbalanced ML Pipeline (Kaggle Dataset)

## Overview
This project focuses on **credit card fraud detection** using the popular Kaggle dataset `creditcard.csv`.  
Instead of immediately applying powerful black-box models (XGBoost, LightGBM), the main goal is to build a **robust, realistic, and business-oriented Machine Learning pipeline**, emphasizing:

- extreme class imbalance handling
- proper evaluation metrics (PR-AUC instead of Accuracy)
- strict anti-data leakage methodology
- decision threshold tuning aligned with business constraints

In fraud detection, achieving a high ROC-AUC is not enough: the model must generate an operationally realistic number of alerts.

---

## Dataset
**Source:** Kaggle — Credit Card Fraud Detection

- **Transactions:** 284,807  
- **Frauds:** 492 (0.17%)  
- **Features:** 31  
  - `Time`
  - `Amount`
  - `V1 ... V28` (PCA anonymized components)
  - `Class` (target)

---

## Problem Statement
With highly imbalanced datasets, **Accuracy becomes misleading**.  
A model predicting only "non-fraud" can achieve over 99% accuracy while detecting no fraud at all.

The goal is to:
- maximize fraud detection (**Recall**)
- while controlling false alerts (**Precision / False Positives**)

---

## Methodology

### 1. Train/Test Split (Anti-Leakage)
A strict anti-leakage protocol is applied:

- stratified **80/20 train-test split** before any transformation
- all preprocessing statistics (scaling, quantiles, transformations) computed only on the training set
- test set remains fully isolated to ensure realistic evaluation

**Key rule: SMOTE is never applied on the test set.**

---

### 2. Business-Driven Feature Engineering
Fraud often occurs in extreme behaviors.

Engineered features:
- `Hour` extracted from `Time`
- binary flags based on training quantiles:
  - `is_very_small_amount` (Q10)
  - `is_very_large_amount` (Q90)

These features capture:
- micro-transactions (card testing)
- extreme transactions (cash-out behavior)

---

### 3. Mathematical Transformations
To handle outliers and skewness:

- `log1p(Amount)` transformation
- `RobustScaler` (median & IQR based scaling)
- `PowerTransformer (Yeo-Johnson)` for PCA feature skewness (supports negative values)

---

### 4. Imbalance Strategies Tested
Several strategies were compared:

- Baseline Logistic Regression
- Class Weight Balanced
- Random Under-sampling
- SMOTE Over-sampling

---

## Model
The main model is a **Logistic Regression classifier**.

Why Logistic Regression?
- interpretable baseline
- fast and stable
- strong benchmark for fraud detection pipelines
- allows clear analysis of imbalance strategies and threshold tuning

---

## Evaluation Metrics
Metrics were chosen to reflect minority-class performance:

- **PR-AUC (Precision-Recall AUC)** (main metric)
- Precision / Recall
- F1-score
- Confusion Matrix
- threshold-based performance analysis

ROC-AUC is still reported, but PR-AUC is prioritized since it better reflects fraud detection performance.

---

## Threshold Tuning (Business Decision Layer)
A key part of this project is **decision threshold optimization**.

The default threshold (0.50) produces too many false positives, making the model unusable in real operations.  
The threshold was tuned to maximize the **F1-score** while reducing the alert volume.

**Final selected threshold: 0.95**

---

## Final Results (Threshold = 0.95)
- **Recall:** 0.89  
- **Precision:** 0.25  
- **F1-score:** 0.39  
- **False Positives:** reduced from 1567 → 262 (≈ 6× fewer alerts)

This configuration detects most fraud cases while keeping alert volume operationally realistic.

---


└── README.md
