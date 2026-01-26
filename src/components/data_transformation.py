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

            numerical_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler())
            ])

            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipe', numerical_pipe, numerical_columns),
                    ('cat_pipe', categorical_pipe, categorical_columns)
                ]
            )
            
            logging.info("Preprocessor completed")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Initialize Feature Engineering
            fe = FeatureEngine()
            
            logging.info("Applying Feature Engineering")
            train_df = fe.transform(train_df)
            test_df = fe.transform(test_df)

            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Churn"
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name].apply(lambda x: 1 if x == 'Yes' else 0)

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name].apply(lambda x: 1 if x == 'Yes' else 0)

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)