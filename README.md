
# 📊 Customer Churn Risk Scoring System

## 🚀 Project Overview

This project is an end-to-end Machine Learning solution designed to predict customer churn for a telecommunications provider.  
By moving beyond simple binary classification, this system implements a **Risk Scoring approach** using a **Soft Voting Ensemble** to identify high-risk customers before they leave.

The project has transitioned from exploratory notebooks to a **Modular Production Architecture**, ensuring scalability, maintainability, and easy deployment.

This repository demonstrates a full ML lifecycle — from EDA and model experimentation to production-ready pipelines and business-driven decision logic.

---

## 🔍 Key Insights from EDA

- **Baseline Churn:** Approximately **26.5%**
- **Red Alert Segment:** Month-to-Month + Fiber Optic + Tenure < 12 months → ~70% churn
- **Payment Behavior:** Electronic Check users churn more
- **High-Value Leakage:** 32% churn among high-value users
- **Data Health:** Fixed missing values in `TotalCharges`

---

## 🛠️ Model Development & Performance

Algorithms tested:
- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

### 🏆 Winning Ensemble
- Logistic Regression
- Gradient Boosting Classifier
- CatBoost

Threshold optimized to **0.35** for higher recall.

- Precision: ~0.62  
- Recall: ~0.69  

---

## 🧠 Why These 3 Models? (The Explainability Factor)

We chose a **Soft Voting Ensemble** of Logistic Regression, Gradient Boosting, and CatBoost to ensure the model is not a black box, but a transparent business decision tool.

| Model | Role in Project | Why This Model? |
|--------|------------------|-----------------|
| **Logistic Regression** | Baseline / Interpreter | Provides a transparent linear baseline. Coefficients allow direct explanation of how individual factors (e.g., Month-to-Month contract) impact churn odds. |
| **Gradient Boosting (GBT)** | Pattern Finder | Captures complex non-linear feature interactions (e.g., Fiber Optic behavior for new vs. long-tenure customers). Feature importance supports the Red Alert segmentation logic. |
| **CatBoost** | Categorical Specialist | Handles high-cardinality categorical features natively (e.g., Gender, InternetService), reducing encoding noise and minimizing overfitting on category combinations. |

This combination balances **interpretability, performance, and business trust**.

---

## 🏗️ Modular Architecture

src/components – Ingestion, Transformation, Training  
src/pipeline – Training & Prediction Pipelines  
artifacts – Saved model & preprocessor  
setup.py – Local package installation  

---

## 💰 Business ROI Example

Target: 1,000 high-risk customers  
Incentive: $60  

Gross Savings = 100 × $70 × 12 = $84,000  
Cost = 1,000 × $60 = $60,000  

✅ Net Profit = **$24,000 annually**

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## ▶️ Run Training

```bash
python -m src.pipeline.train_pipeline
```

---

## 🔮 Future Work

- Streamlit dashboard
- Batch prediction for CSV upload
- Model monitoring & drift detection

---