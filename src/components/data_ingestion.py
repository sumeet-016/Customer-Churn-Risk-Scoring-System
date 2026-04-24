import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path:  str = os.path.join('artifacts', 'test.csv')
    raw_data_path:   str = os.path.join('artifacts', 'data.csv')
    dataset_path:    str = os.path.join('NoteBook', 'Data', 'Telco-Customer-Churn.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv(self.ingestion_config.dataset_path)
            logging.info(f"Raw dataset loaded — shape: {df.shape}")

            # ── Fix 1: Drop customerID — unique identifier, no predictive value ──
            df = df.drop(columns=['customerID'])
            logging.info("Dropped customerID column")

            # ── Fix 2: TotalCharges has 11 hidden empty strings ─────────────────
            # tenure=0 customers have ' ' instead of 0.0 in TotalCharges
            df['TotalCharges'] = df['TotalCharges'].str.strip()
            df['TotalCharges'] = df['TotalCharges'].replace('', '0.0')
            df['TotalCharges'] = df['TotalCharges'].astype(float)
            logging.info("Fixed TotalCharges — converted to float, replaced empty strings with 0.0")

            # ── Fix 3: Encode target — Yes=1, No=0 ─────────────────────────────
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            logging.info("Encoded Churn — Yes:1, No:0")

            # ── Log class balance ───────────────────────────────────────────────
            churn_counts = df['Churn'].value_counts()
            churn_pct    = df['Churn'].value_counts(normalize=True) * 100
            logging.info(f"Class distribution — No Churn: {churn_counts[0]} ({churn_pct[0]:.1f}%) | Churn: {churn_counts[1]} ({churn_pct[1]:.1f}%)")
            logging.info("⚠️ Dataset is imbalanced — consider SMOTE or class_weight during model training")

            logging.info(f"Final dataset shape: {df.shape}")

            # ── Save raw + split ────────────────────────────────────────────────
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # ✅ Stratify on Churn to preserve class balance in both splits
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df['Churn']   # ✅ ensures 73/27 split in both train and test
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,   index=False, header=True)

            logging.info(f"Train size: {len(train_set):,} | Test size: {len(test_set):,}")
            logging.info("Data Ingestion complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)