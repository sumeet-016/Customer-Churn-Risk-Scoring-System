import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # ✅ Load once at startup — not on every prediction call
        model_path        = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        logging.info("Loading model and preprocessor")
        self.model        = load_object(file_path=model_path)
        self.preprocessor = load_object(file_path=preprocessor_path)
        logging.info("Model and preprocessor loaded successfully")

    def predict(self, features: pd.DataFrame):
        try:
            # ─── Transform ────────────────────────────────────────────────
            data_transformed = self.preprocessor.transform(features)

            # ─── Predict Probability ──────────────────────────────────────
            churn_probability = self.model.predict_proba(data_transformed)[:, 1]

            # ─── Apply Tuned Threshold ────────────────────────────────────
            # 0.35 threshold improves Recall — catches more actual churners
            churn_prediction = (churn_probability >= 0.35).astype(int)

            return churn_prediction[0], round(float(churn_probability[0]) * 100, 2)
            # returns: (0 or 1,  probability as percentage e.g. 74.23)

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender          : str,
        SeniorCitizen   : int,
        Partner         : str,
        Dependents      : str,
        tenure          : int,
        PhoneService    : str,
        MultipleLines   : str,
        InternetService : str,
        OnlineSecurity  : str,
        OnlineBackup    : str,
        DeviceProtection: str,
        TechSupport     : str,
        StreamingTV     : str,
        StreamingMovies : str,
        Contract        : str,
        PaperlessBilling: str,
        PaymentMethod   : str,
        MonthlyCharges  : float,
        TotalCharges    : float,
    ):
        self.gender           = gender
        self.SeniorCitizen    = SeniorCitizen
        self.Partner          = Partner
        self.Dependents       = Dependents
        self.tenure           = tenure
        self.PhoneService     = PhoneService
        self.MultipleLines    = MultipleLines
        self.InternetService  = InternetService
        self.OnlineSecurity   = OnlineSecurity
        self.OnlineBackup     = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport      = TechSupport
        self.StreamingTV      = StreamingTV
        self.StreamingMovies  = StreamingMovies
        self.Contract         = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod    = PaymentMethod
        self.MonthlyCharges   = MonthlyCharges
        self.TotalCharges     = TotalCharges

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame({
                "gender"          : [self.gender],
                "SeniorCitizen"   : [int(self.SeniorCitizen)],
                "Partner"         : [self.Partner],
                "Dependents"      : [self.Dependents],
                "tenure"          : [int(self.tenure)],
                "PhoneService"    : [self.PhoneService],
                "MultipleLines"   : [self.MultipleLines],
                "InternetService" : [self.InternetService],
                "OnlineSecurity"  : [self.OnlineSecurity],
                "OnlineBackup"    : [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport"     : [self.TechSupport],
                "StreamingTV"     : [self.StreamingTV],
                "StreamingMovies" : [self.StreamingMovies],
                "Contract"        : [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod"   : [self.PaymentMethod],
                "MonthlyCharges"  : [float(self.MonthlyCharges)],
                "TotalCharges"    : [float(self.TotalCharges)],
            })

        except Exception as e:
            raise CustomException(e, sys)