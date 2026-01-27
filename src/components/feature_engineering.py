import sys
import pandas as pd
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
            logging.info("Started Feature Engineering transformation")
            df = X.copy()

            #Convert TotalCharges to numeric, handling errors

            if df['TotalCharge'].dtype == 'object':
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'].fillna(0, inplace=True)


            def tenure_bucket(t):
                if t <=12: return '0-1 yr'
                elif t <= 24: return '1-2 yrs'
                elif t <= 48: return '2-4 yrs'
                else: return '> 4 yrs'

            if 'tenure' in df.columns:
                df['tenure_group'] = df['tenure'].apply(tenure_bucket)
            
            # Drop customerID as it's not a feature
            if 'customerID' in df.columns:
                df.drop(columns=['customerID'], inplace=True)
                
            logging.info("Feature Engineering transformations completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)