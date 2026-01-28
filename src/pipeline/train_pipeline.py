import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_path, test_path)
            model_path = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            print(f"Training Pipeline Complete. Model saved at: {model_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()