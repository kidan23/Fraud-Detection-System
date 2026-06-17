# Fraud Detection System  for Adey Innovations

A machine learning system designed to identify fraudulent transactions in e-commerce platforms and banking credit card records.

---

## Project Overview

Adey Innovations needs a unified fraud detection system that handles:
- **E-commerce transactions** — with user, device, and behavioral data
- **Bank credit card transactions** — with anonymized PCA features

The project covers the full pipeline: data cleaning, feature
engineering, model training and comparison, and model
explainability using SHAP, with insights translated into
concrete business recommendations.

---


## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/              # Original datasets (git ignored)
│   └── processed/        # Cleaned datasets and charts (git ignored)
├── notebooks/
│   ├── eda-fraud-data.ipynb        # Task 1 - EDA & Preprocessing
│   ├── eda-creditcard.ipynb        # Task 1 - Credit Card EDA
│   ├── feature-engineering.ipynb   # Task 1 - Feature Engineering
│   ├── modeling.ipynb              # Task 2 - Model Training & Comparison
│   └── shap-explainability.ipynb   # Task 3 - SHAP Analysis & Recommendations
├── src/                  # Reusable source code
├── tests/                # Unit tests
├── models/               # Saved trained models (LR, RF, XGBoost, LightGBM)
├── scripts/              # Standalone scripts
├── requirements.txt      # Python dependencies
└── .github/workflows/    # CI/CD pipeline
```

---

## Datasets

| Dataset                  | Description                   | Rows    |
| ------------------------ | ----------------------------- | ------- |
| Fraud_Data.csv           | E-commerce transactions       | 151,112 |
| IpAddress_to_Country.csv | IP to country mapping         | 138,846 |
| creditcard.csv           | Bank credit card transactions | 284,807 |

---

## Task 1 — Data Analysis and Preprocessing

### What was done

- Loaded and explored all three datasets
- Removed 1,081 duplicate rows from credit card data
- Fixed datetime types for signup and purchase times
- Merged e-commerce data with IP geolocation table
- Engineered new fraud signal features
- Applied SMOTE to balance training data

### Key Findings

| Finding                | Value  |
| ---------------------- | ------ |
| E-commerce fraud rate  | 9.36%  |
| Credit card fraud rate | 0.17%  |
| New account fraud rate | 87.34% |
| Rows lost in IP merge  | 21,966 |

### Features Engineered

- `time_since_signup` — hours between signup and purchase
- `hour_of_day` — hour the transaction happened
- `day_of_week` — day the transaction happened
- `transaction_velocity` — purchases per hour per user
- `is_new_account` — account less than 24 hours old
- `country` — derived from IP address lookup

### Class Imbalance Handling

- Applied SMOTE on training data only
- Balanced fraud ratio from 9.36% → 50%
- Test data left untouched and real

---

## Task 2 — Model Building and Training

### What was done

- Split data using stratified train-test split (80/20)
- Applied SMOTE on training data only
- Trained Logistic Regression as an interpretable baseline
- Trained Random Forest, XGBoost, and LightGBM as ensemble models
- Ran 5-fold stratified cross validation on all 4 models
- Identified and corrected a data leakage issue in cross validation
- Compared all models on performance and interpretability
- Selected LightGBM as the final model

### Data Leakage — Found and Fixed

An initial cross validation run produced suspiciously high
scores (up to 0.9987 F1) because SMOTE was applied before
splitting into folds, allowing synthetic fraud examples to
leak between training and validation folds.

This was corrected using an `imblearn` Pipeline that applies
SMOTE only within the training portion of each fold, giving
honest, leakage-free results.

### Corrected 5-Fold Cross Validation Results

| Model               | Fraud Data F1 | Fraud Std | Credit Card F1 | Credit Std |
| ------------------- | ------------- | --------- | -------------- | ---------- |
| Logistic Regression | 0.6636        | 0.0086    | 0.0965         | 0.0089     |
| Random Forest       | 0.6954        | 0.0080    | 0.6857         | 0.0699     |
| XGBoost             | 0.6979        | 0.0063    | 0.7201         | 0.0594     |
| **LightGBM**        | **0.6975**    | 0.0070    | **0.7313**     | **0.0542** |

### Why LightGBM Was Selected

- Best F1 score (0.7313) on the harder, severely imbalanced credit card dataset
- Lowest standard deviation among ensemble models on credit card data
- Virtually tied with XGBoost on fraud data (0.6975 vs 0.6979)
- Equally interpretable — supports feature importance and SHAP
- Fast and memory efficient for real-time scoring

---

## Task 3 — Model Explainability with SHAP

### What was done

- Extracted built-in feature importance from LightGBM (top 10 features)
- Generated a global SHAP summary plot across all transactions
- Generated SHAP force plots for 3 individual cases:
  - **True Positive** — correctly identified fraud (99.75% probability)
  - **False Positive** — legitimate transaction wrongly flagged (53.83% probability)
  - **False Negative** — missed fraud, low value purchase from an old account (7.92% probability)
- Compared SHAP importance against built-in feature importance
- Identified the top 5 drivers of fraud predictions
- Documented surprising findings and translated insights into business recommendations

### Top 5 Drivers of Fraud (SHAP)

| Rank | Feature               | SHAP Importance |
| ---- | --------------------- | --------------- |
| 1    | time_since_signup     | 2.1230          |
| 2    | country_United_States | 0.5641          |
| 3    | source_SEO            | 0.2886          |
| 4    | browser_IE            | 0.2383          |
| 5    | country_China         | 0.2297          |

### Key Findings

- `time_since_signup` is by far the strongest fraud signal in both
  built-in importance and SHAP, confirming the Task 1 finding that
  new accounts (under 24 hours old) have an 87.34% fraud rate
- Country features (US, China) have a much larger directional
  impact via SHAP than their built-in importance suggests
- Browser and traffic source features (IE, Firefox, Safari, SEO, Direct)
  carry meaningful signal, possibly reflecting bot or scripted traffic

### Business Recommendations

1. **Mandatory verification for new accounts** — require additional
   identity checks for any purchase made within 24 hours of signup
2. **Geographic risk scoring** — add country-based risk as a secondary
   signal alongside the model's prediction
3. **Borderline probability review zone** — route transactions with
   prediction probability between 0.40–0.60 to a secondary check
   instead of an automatic block, and add extra checks for small
   purchases from older accounts
4. **Browser and channel monitoring** — flag risky combinations
   (e.g. new account + IE + SEO traffic) for closer review

---

## Final Model Summary

| Item                          | Value             |
| ----------------------------- | ----------------- |
| Final model                   | LightGBM          |
| Credit Card F1 (corrected CV) | 0.7313            |
| Fraud Data F1 (corrected CV)  | 0.6975            |
| Strongest fraud signal        | time_since_signup |
| New account fraud rate        | 87.34%            |

---


## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Furtunaa/Fraud-detection-bank-week-5-6.git
cd fraud-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open notebooks
jupyter lab
```

---

