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
            # ✅ SeniorCitizen moved here — already 0/1 integer
            # ✅ Add the 3 new engineered features
            numerical_columns = [
                'tenure', 'MonthlyCharges', 'TotalCharges',
                'ServiceCount', 'SeniorCitizen',
                'Is_MonthToMonth',   # ✅ added
                'Is_FiberOptic',     # ✅ added
                'Charges_Ratio'      # ✅ added
            ]

            # ✅ SeniorCitizen removed from here
            categorical_columns = [
                'gender', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod', 'tenure_group'
            ]

            logging.info('Building preprocessing pipeline')

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # ✅ Removed StandardScaler after OneHotEncoder
                # OHE produces binary 0/1 — scaling distorts meaning
                ('one_hot_encoder', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ], remainder='drop')

            logging.info('Preprocessing pipeline built successfully')

            # ✅ No FeatureEngine here — handled separately in initiate_data_transformation
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            target_col = 'Churn'

            # ─── Step 1: Feature Engineering on full df ────────────────────
            # Applied BEFORE X/y split — FeatureEngine may drop/add rows
            feature_engine = FeatureEngine()
            train_df = feature_engine.fit_transform(train_df)
            test_df  = feature_engine.transform(test_df)

            logging.info(f"After FeatureEngine — Train: {train_df.shape} | Test: {test_df.shape}")

            # ─── Step 2: Split X and y ─────────────────────────────────────
            X_train = train_df.drop(columns=[target_col])
            X_test  = test_df.drop(columns=[target_col])

            # ✅ Churn is already 0/1 from data_ingestion — no .map() needed
            y_train = train_df[target_col].astype(int)
            y_test  = test_df[target_col].astype(int)

            logging.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
            logging.info(f"X_test : {X_test.shape}  | y_test : {y_test.shape}")
            logging.info(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

            # ─── Step 3: Fit preprocessor ──────────────────────────────────
            preprocessing_obj = self.get_data_transformer_object()
            train_arr_processed = preprocessing_obj.fit_transform(X_train)
            test_arr_processed  = preprocessing_obj.transform(X_test)

            # ─── Step 4: Combine features + target ────────────────────────
            train_arr = np.c_[train_arr_processed, np.array(y_train)]
            test_arr  = np.c_[test_arr_processed,  np.array(y_test)]

            logging.info(f"Final train_arr: {train_arr.shape} | test_arr: {test_arr.shape}")

            # ─── Step 5: Bundle FeatureEngine + preprocessor → single pkl ─
            full_pipeline = Pipeline(steps=[
                ('feature_engine', feature_engine),
                ('preprocessing',  preprocessing_obj)
            ])

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=full_pipeline
            )

            logging.info("Saved full pipeline to preprocessor.pkl")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)