import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.components.feature_engineering import FeatureEngine
import dill

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTranformationConfig()

    def get_data_tranformation_object(self):
        try:
            numerical_columns = ['tenure', 'MonthlyCharges', 'TotalChanges']

            # Note: customerID is dropped in FE. 'tenure_group' is added in FE (categorical).
            
            categorical_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                'PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaperlessBilling', 'PaymentMethod',
                'tenure_group' # Added from FE
            ]

            
        except:
            pass