import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            logging.info("Feature Engineering started")
            df = X.copy()

            # TotalCharges conversion
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'] = df['TotalCharges'].fillna(0)

            # Tenure bucketing
            def tenure_bucket(t):
                if t <= 12: return '0-1 yr'
                elif t <= 24: return '1-2 yrs'
                elif t <= 48: return '2-4 yrs'
                else: return '> 4 yrs'

            if 'tenure' in df.columns:
                df['tenure_group'] = df['tenure'].apply(tenure_bucket)
            
            # Service Counting
            service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            if all(col in df.columns for col in service_cols):
                df['ServiceCount'] = df[service_cols].apply(lambda x: x == 'Yes').sum(axis=1)

            # Binary Flags for High Risk
            if 'Contract' in df.columns:
                df['Is_MonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
            
            if 'InternetService' in df.columns:
                df['Is_FiberOptic'] = (df['InternetService'] == 'Fiber optic').astype(int)

            # Dropping Noisy Columns
            cols_to_drop = ['customerID']
            for col in cols_to_drop:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                
            return df
        except Exception as e:
            raise CustomException(e, sys)