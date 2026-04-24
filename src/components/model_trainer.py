import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # ✅ Cast both to int
            y_train = y_train.astype(int)
            y_test  = y_test.astype(int)

            lr = LogisticRegression(
                solver='liblinear',
                C=10,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )

            gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )

            cb = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                auto_class_weights='Balanced',
                verbose=0,
                random_state=42
            )

            ensemble = VotingClassifier(
                estimators=[('lr', lr), ('gb', gb), ('cb', cb)],
                voting='soft',
                weights=[1, 2, 2]
            )

            logging.info("Training ensemble model...")
            ensemble.fit(X_train, y_train)
            logging.info("Training complete")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=ensemble
            )

            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)