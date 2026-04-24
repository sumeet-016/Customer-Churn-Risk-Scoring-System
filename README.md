# 📊 Customer Churn Risk Scoring System

> An end-to-end Machine Learning system that identifies high-risk telecom customers likely to churn, using a tuned Soft Voting Ensemble, domain-driven feature engineering, and a production-ready Streamlit application with a built-in Retention ROI Calculator.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-red)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.7-yellow)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Highlights](#project-highlights)
- [Key EDA Findings](#key-eda-findings)
- [Project Architecture](#project-architecture)
- [Feature Engineering](#feature-engineering)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Development](#model-development)
- [Why These 3 Models](#why-these-3-models)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Final Performance](#final-performance)
- [Business Impact & ROI](#business-impact--roi)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## Problem Statement

In the telecom industry, acquiring a new customer costs **5–7x more** than retaining an existing one. Yet most companies lack the data-driven infrastructure to proactively identify who is about to leave — and intervene before it's too late.

This project builds a **Customer Churn Risk Scoring System** that goes beyond binary Yes/No classification. It assigns each customer a **churn probability score**, enabling business teams to prioritize the highest-risk customers for targeted retention campaigns — before revenue is lost.

---

## Project Highlights

- Built a fully modular, production-grade ML pipeline with custom exception handling and logging
- Conducted deep EDA with statistical validation (T-Test, Chi-Square) to confirm and quantify churn drivers
- Benchmarked **7 classification algorithms** using 5-fold cross-validation before selecting the best ensemble
- Tuned top 3 models using `RandomizedSearchCV` and combined them into a **Soft Voting Ensemble**
- **Threshold tuned to 0.35** (vs default 0.5) — improved Recall from 50% to **70%**, catching significantly more actual churners
- Identified the **"Red Alert" segment** (Month-to-Month + Fiber Optic + Tenure ≤ 12 months) with a **70% churn rate** — 2.6x the baseline
- Implemented a custom `FeatureEngine` class (sklearn-compatible) with **zero data leakage** — all statistics learned exclusively from training data
- Deployed as an interactive Streamlit app with churn probability score, 4-tier risk classification, and built-in ROI calculator

---

## Key EDA Findings

### Primary Churn Drivers (Statistically Validated)

| Factor | Finding | Validation |
|---|---|---|
| **Tenure** | Churn heavily concentrated in first 12 months | T-Test p < 0.05 |
| **Contract Type** | Month-to-month = highest churn rate by far | Chi-Square p < 0.05 |
| **Internet Service** | Fiber optic users churn more than DSL | Strong association |
| **Payment Method** | Electronic check users have highest churn | Clear pattern |
| **Monthly Charges** | Churners pay significantly higher monthly charges | T-Test p < 0.05 |
| **Gender** | No significant difference across genders | Weak predictor — excluded |

### The "Red Alert" Segment

```
Month-to-Month Contract  +  Fiber Optic  +  Tenure ≤ 12 months

  → 70% Churn Rate   (vs 26.5% overall baseline)
  → 2.6x higher likelihood to churn than the average customer
  → Priority target for retention budget
```

### Revenue Impact

- ~**30% of Monthly Recurring Revenue** is at risk from churning customers
- **High-value customers** (top 25% by monthly charges) churn at **32%** — above average
- Most painful loss: high-value customers leaving in their first year
- `TotalCharges` and `tenure` are strongly correlated (0.83) — both retained as they carry distinct signals with and without the other

---

## Project Architecture

```
Raw CSV Data
      │
      ▼
Data Ingestion          — fixes TotalCharges dtype, encodes Churn (Yes→1 / No→0),
                          stratified train/test split, logs class balance
      │
      ▼
Feature Engineering     — learns stats from train only (no leakage),
                          creates ServiceCount, tenure_group, Is_MonthToMonth,
                          Is_FiberOptic, Charges_Ratio
      │
      ▼
Data Transformation     — encodes categoricals, scales numerics,
                          bundles full pipeline → saves as preprocessor.pkl
      │
      ▼
Model Training          — trains Soft Voting Ensemble with class balancing,
                          threshold tuned to 0.35, saves model.pkl
      │
      ▼
Streamlit App           — loads preprocessor.pkl + model.pkl,
                          shows probability score + risk tier + ROI calculator
```

The pipeline is fully modular — each component is independent, testable, and replaceable.

---

## Feature Engineering

A custom sklearn-compatible `FeatureEngine` transformer engineers domain-specific features. All statistics (medians, group means) are learned **only from training data** — preventing leakage to test or inference data.

| Feature | Logic | Business Meaning |
|---|---|---|
| `tenure_group` | Bucketed: 0-1yr, 1-2yrs, 2-4yrs, 4+yrs | Early-tenure customers are the highest-risk cohort |
| `ServiceCount` | Count of active optional services (`== 'Yes'`) | More services = higher switching cost = lower churn |
| `Is_MonthToMonth` | `Contract == 'Month-to-month'` → 1/0 | Single strongest churn predictor — no contract lock-in |
| `Is_FiberOptic` | `InternetService == 'Fiber optic'` → 1/0 | Fiber users churn disproportionately |
| `Charges_Ratio` | `MonthlyCharges / (TotalCharges + 1)` | High ratio = new customer paying high price = churn risk |

---

## Preprocessing Pipeline

A modular `ColumnTransformer` handles all encoding and imputation after feature engineering. The full pipeline (`FeatureEngine` + `ColumnTransformer`) is bundled into a single `preprocessor.pkl` for clean, order-safe inference.

| Feature Group | Columns | Transformer |
|---|---|---|
| Numerical | `tenure`, `MonthlyCharges`, `TotalCharges`, `ServiceCount`, `SeniorCitizen`, `Is_MonthToMonth`, `Is_FiberOptic`, `Charges_Ratio` | `SimpleImputer(median)` → `StandardScaler` |
| Categorical | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure_group` | `SimpleImputer` → `OneHotEncoder` |

---

## Model Development

### Step 1 — Baseline Benchmarking (5-Fold Cross-Validation)

All 7 models were evaluated using 5-fold CV on the training set before any tuning:

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 0.8068 |
| SVM | 0.8005 |
| **CatBoost** | **0.8009** |
| Gradient Boosting | 0.8027 |
| LightGBM | 0.8004 |
| Random Forest | 0.7927 |
| XGBoost | 0.7879 |

**Top 3 selected for tuning:** Logistic Regression, Gradient Boosting, CatBoost — best accuracy with diverse learning strategies.

---

## Hyperparameter Tuning

Top 3 models tuned using `RandomizedSearchCV` with 5-fold CV:

### Logistic Regression
```
Search Space : C ∈ [0.1, 1, 10, 100] | solver ∈ [liblinear, lbfgs]
Best Score   : 0.8073
Best Params  : C=10, solver='liblinear'
```

### Gradient Boosting
```
Search Space : n_estimators ∈ [100, 200] | learning_rate ∈ [0.01, 0.05, 0.1] | max_depth ∈ [3, 4, 5]
Best Score   : 0.8044
Best Params  : n_estimators=200, max_depth=5, learning_rate=0.01
```

### CatBoost
```
Search Space : iterations ∈ [100, 200, 500] | learning_rate ∈ [0.01, 0.05, 0.1] | depth ∈ [4, 6, 8]
Best Score   : 0.8094
Best Params  : iterations=500, depth=4, learning_rate=0.01
```

### Step 2 — Soft Voting Ensemble

Tuned models combined into a **Soft Voting Ensemble** — blends predicted probabilities from all 3 for a more stable, generalizable output:

```
LogisticRegression (C=10, solver=liblinear) ──┐
                                               ├──► Soft Vote ──► Churn Probability
GradientBoosting   (n=200, depth=5, lr=0.01) ──┤
                                               │
CatBoost           (iter=500, depth=4, lr=0.01)┘
```

### Step 3 — Threshold Tuning

Default threshold of 0.50 was evaluated against lower values using the Precision-Recall curve:

| Threshold | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|
| 0.50 (default) | ~0.67 | ~0.50 | ~0.57 |
| **0.35 (tuned)** | **0.55** | **0.70** | **0.62** |

Lowering to **0.35** improved Recall by **+20 percentage points** — catching significantly more actual churners at a controlled precision cost.

---

## Final Performance

**Ensemble Model at Threshold = 0.35 — Test Set Results:**

| | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Churn (0) | 0.88 | 0.80 | 0.83 | 1,033 |
| **Churn (1)** | **0.55** | **0.70** | **0.62** | 374 |
| **Accuracy** | | | **0.77** | 1,407 |
| Macro Avg | 0.72 | 0.75 | 0.73 | 1,407 |
| Weighted Avg | 0.79 | 0.77 | 0.78 | 1,407 |

### Reading The Metrics

- **Recall 0.70** — Of every 10 actual churners, the model catches 7 before they leave. Up from 5 at the default threshold
- **Precision 0.55** — Of every 10 customers flagged as churners, ~6 actually churn. A deliberate trade-off to maximize coverage
- **No Churn Precision 0.88** — The model still correctly identifies the vast majority of stable customers, avoiding wasted retention spend
- **Accuracy 0.77** — Meaningful here because class balance is moderate (73/27), unlike severely imbalanced datasets where accuracy is misleading

> **Why Recall over Precision?** Missing a churner (False Negative) costs the business a lost customer and their full lifetime value. Falsely flagging a loyal customer (False Positive) costs only a small retention incentive. The asymmetry justifies optimizing for Recall.

---

## Business Impact & ROI

### Built-In ROI Calculator (Live in the App)

The Streamlit app allows business teams to enter their own incentive cost and average revenue to compute live ROI estimates.

```
Scenario: Targeted retention campaign on the Red Alert Segment

  Incentive Cost per Customer  : $60
  Avg Monthly Revenue          : $70
  Target Customers             : 1,000 high-risk customers
  Goal                         : Retain 10% (100 customers)

  Revenue Saved : 100 × $70 × 12 months = $84,000
  Campaign Cost : 1,000 × $60           = $60,000
  ─────────────────────────────────────────────────
  Net Annual Gain              :          $24,000 ✅
```

### Risk Tier Classification (App Output)

| Churn Probability | Risk Tier | Recommended Action |
|---|---|---|
| ≥ 70% | 🔴 Critical | Immediate outreach + high-value incentive |
| 50–69% | 🟠 High | Proactive contact + moderate incentive |
| 35–49% | 🟡 Medium | Monitor + soft engagement |
| < 35% | 🟢 Low | No action needed |

---

## Why These 3 Models

The ensemble was designed for both **performance and business trust** — each model plays a distinct and complementary role:

| Model | Role | Why Chosen |
|---|---|---|
| **Logistic Regression** | Baseline / Interpreter | Provides a transparent linear baseline. Coefficients directly explain how factors like Month-to-Month contract impact churn odds — easy to communicate to non-technical stakeholders |
| **Gradient Boosting** | Pattern Finder | Captures complex non-linear interactions (e.g., Fiber Optic behaviour for new vs. long-tenure customers). Feature importance directly validates the Red Alert segmentation logic from EDA |
| **CatBoost** | Categorical Specialist | Handles high-cardinality categorical features natively (e.g., `PaymentMethod`, `InternetService`), reducing encoding noise and overfitting on specific category combinations |

This combination balances **interpretability, predictive power, and business trust** — no single black-box model.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML Framework | Scikit-Learn 1.5.2 |
| Boosting Model | CatBoost 1.2.7 |
| Statistical Testing | SciPy |
| Data Processing | Pandas, NumPy |
| Serialization | Dill |
| Web App | Streamlit 1.37.1 |
| Logging | Python `logging` module |
| Version Control | Git & GitHub |

---

## Project Structure

```
Customer Churn Risk Scoring System/
│
├── artifacts/                          # Generated at runtime
│   ├── preprocessor.pkl                # FeatureEngine + ColumnTransformer bundled
│   ├── model.pkl                       # Trained VotingClassifier
│   ├── train.csv
│   └── test.csv
│
├── NoteBook/
│   ├── Data/
│   │   └── Telco-Customer-Churn.csv    # Raw dataset
│   ├── EDA.ipynb                       # Full EDA with statistical validation & business insights
│   └── ModelTraining.ipynb             # Model benchmarking, tuning, threshold selection
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py           # Reads, cleans, stratified splits data
│   │   ├── feature_engineering.py      # Custom sklearn FeatureEngine (no leakage)
│   │   ├── data_transformation.py      # Encoding, imputing, saves preprocessor.pkl
│   │   └── model_trainer.py            # Trains VotingClassifier, saves model.pkl
│   ├── pipeline/
│   │   └── predict_pipeline.py         # PredictPipeline + CustomData for inference
│   ├── exception.py                    # Custom exception with file + line tracking
│   ├── logger.py                       # Timestamped logging
│   └── utils.py                        # save_object / load_object using dill
│
├── app.py                              # Streamlit web application
├── main.py                             # Orchestrates full training pipeline
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/sumeet-016/customer-churn-risk-system.git
cd customer-churn-risk-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Pipeline
```bash
python main.py
```
Generates `artifacts/preprocessor.pkl` and `artifacts/model.pkl`.

### 4. Launch the App
```bash
streamlit run app.py
```

> **Note:** Always run from the project root using Anaconda Prompt for environment consistency.

---

## Future Enhancements

- SHAP explainability — show which features drove each individual customer's risk score
- Batch prediction — CSV upload for scoring entire customer lists at once
- Segment-level dashboards — churn rate breakdowns by contract type, tenure band, and payment method
- SMOTE oversampling — handle class imbalance at the data level during training
- Cloud deployment — AWS / GCP / Hugging Face Spaces
- Model monitoring and data drift detection with automated retraining triggers

---

## Author

**Sumeet Kumar Pal**
Aspiring Data Scientist | Machine Learning Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-sumeet--016-black?logo=github)](https://github.com/sumeet-016)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-palsumeet-blue?logo=linkedin)](https://www.linkedin.com/in/palsumeet)
