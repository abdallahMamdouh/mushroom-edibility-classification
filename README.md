# Project 7: Mushroom Edibility Classification with Logistic Regression

## Overview & Problem Statement
Foraging for wild mushrooms presents grave risks: misidentifying a toxic species as edible can cause severe poisoning or death. While experienced mycologists examine intricate morphological attributes, automated algorithmic triage provides an objective safety filter.

The objective of this project is to build a binary classification model that accurately distinguishes poisonous mushrooms from edible specimens based on macroscopic anatomical traits (cap morphology, gill structure, stem dimensions, and season).

---

## Dataset Description
- **Source**: Secondary Mushroom Classification Dataset (`mushroom.csv`).
- **File**: `mushroom.csv`
- **Volume**: 54,035 observations, 9 features.
- **Target Variable**: `target` (originally `class`: Binary classification: `0` = Edible, `1` = Poisonous).
- **Predictor Attributes**:
  - `cap-diameter`: Diameter of the mushroom cap in centimeters.
  - `cap-shape`: Categorical shape code (bell, conical, convex, flat, etc.).
  - `gill-attachment`: Attachment structure (adnate, adnexed, decurrent, free).
  - `gill-color`: Color classification code of the hymenium.
  - `stem-height`: Total stem height in centimeters.
  - `stem-width`: Stem thickness in millimeters.
  - `stem-color`: Stem pigmentation category.
  - `season`: Season of observation (spring, summer, autumn, winter).

---

## Methodology & Machine Learning Workflow

### 1. Data Cleaning & Integrity Verification
- Verified missing value distributions.
- Checked for and purged duplicated records (`drop_duplicates`) to prevent artificial inflation of test metrics.
- Renamed target feature column to `target` for clean pipeline consumption.

### 2. Feature Splitting & Scaling
- Split the dataset into 80% training and 20% test partitions (`random_state=20, test_size=0.2`).
- Continuous anatomical metrics (`cap-diameter`, `stem-height`, `stem-width`) exhibit distinct unit ranges. Applied `StandardScaler` fitted strictly on train data to normalize dimensional magnitudes to zero-mean unit-variance.

### 3. Model Architecture & Evaluation
- Trained a baseline `LogisticRegression` classifier with L2 regularization.
- Generated predictions on the 10,747 test specimens and assessed performance via confusion matrix analysis, precision, recall, and F1 metrics.

---

## Performance & Evaluation Metrics

Evaluated on the independent test set ($N = 10,747$):

| Metric | Class 0 (Edible) | Class 1 (Poisonous) | Overall / Weighted Average |
| :--- | :---: | :---: | :---: |
| **Precision** | **0.54** | **0.71** | **0.64** |
| **Recall** | **0.61** | **0.65** | **0.64** |
| **F1-Score** | **0.58** | **0.68** | **0.64** |
| **Accuracy** | — | — | **63.6% (~0.64)** |

- **Macro Average F1**: **0.63**
- **Weighted Average F1**: **0.64**
- **Analysis**: Linear decision boundaries demonstrate moderate capability (64% accuracy) on this secondary dataset. The non-linear morphological relationships between cap, gill, and stem suggest that non-linear models (Decision Trees, Random Forests, XGBoost) would yield higher precision, which is recommended for safety-critical domain applications.

---

## Quickstart Guide

### 1. Environment Setup
Navigate to the project folder:
```bash
cd 07-mushroom-edibility-classification
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Notebook
Launch Jupyter:
```bash
jupyter notebook test.ipynb
```
Run all cells to execute data loading, duplicate pruning, standard scaling, logistic regression modeling, and performance reporting.
