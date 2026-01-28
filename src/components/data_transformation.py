import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.components.feature_engineering import FeatureEngine
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ServiceCount']
            categorical_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ], remainder='drop') 

            return Pipeline(steps=[
                ('feature_engine', FeatureEngine()),
                ('preprocessing', preprocessor)
            ])

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()
            target_col = "Churn"
            
            X_train = train_df.drop(columns=[target_col], axis=1)
            y_train = train_df[target_col].map({'Yes': 1, 'No': 0}).astype(int)

            X_test = test_df.drop(columns=[target_col], axis=1)
            y_test = test_df[target_col].map({'Yes': 1, 'No': 0}).astype(int)

            train_arr_processed = preprocessing_obj.fit_transform(X_train)
            test_arr_processed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[train_arr_processed, np.array(y_train)]
            test_arr = np.c_[test_arr_processed, np.array(y_test)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)