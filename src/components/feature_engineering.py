import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_total_charges_ = None

    def fit(self, X, y=None):
        try:
            df = X.copy()

            # ✅ Learn median TotalCharges from train only
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                self.median_total_charges_ = df['TotalCharges'].median()

            logging.info("FeatureEngine fit complete")
            return self

        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X, y=None):
        try:
            logging.info("Feature Engineering started")
            df = X.copy()

            # ─── Fix TotalCharges ──────────────────────────────────────────
            # ✅ Use train median — not current data's median
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'] = df['TotalCharges'].fillna(
                    self.median_total_charges_ if self.median_total_charges_ else 0
                )

            # ─── Tenure Bucketing ──────────────────────────────────────────
            if 'tenure' in df.columns:
                df['tenure_group'] = pd.cut(
                    df['tenure'],
                    bins=[0, 12, 24, 48, 72],
                    labels=['0-1 yr', '1-2 yrs', '2-4 yrs', '4+ yrs'],
                    include_lowest=True
                ).astype(str)
                logging.info("Created tenure_group")

            # ─── Service Count ─────────────────────────────────────────────
            service_cols = [
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            if all(col in df.columns for col in service_cols):
                df['ServiceCount'] = df[service_cols].apply(
                    lambda x: (x == 'Yes').sum(), axis=1
                )
                logging.info("Created ServiceCount")

            # ─── High Risk Binary Flags ────────────────────────────────────
            # ✅ These ARE now listed in data_transformation numerical_columns
            if 'Contract' in df.columns:
                df['Is_MonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
                logging.info("Created Is_MonthToMonth")

            if 'InternetService' in df.columns:
                df['Is_FiberOptic'] = (df['InternetService'] == 'Fiber optic').astype(int)
                logging.info("Created Is_FiberOptic")

            # ─── Charges Ratio ─────────────────────────────────────────────
            # ✅ New — ratio of monthly to total charges signals early churn risk
            if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
                df['Charges_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
                logging.info("Created Charges_Ratio")

            logging.info(f"Feature Engineering completed — shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)