import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_pipeline():
    try:
        logging.info("Pipeline Started")

        # Step 1 — Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Complete")

        # Step 2 — Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info(f"Data Transformation Complete | Preprocessor: {preprocessor_path}")

        # Step 3 — Training
        trainer = ModelTrainer()
        model_path = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training Complete | Model: {model_path}")

        logging.info("Pipeline Finished Successfully")

    except Exception as e:
        logging.error("Pipeline Failed")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_pipeline()