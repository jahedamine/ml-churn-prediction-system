import joblib
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger()


class ChurnService:

    def __init__(self):

        logger.info("Loading churn model...")

        self.model = joblib.load("models/artifacts/model.pkl")
        self.scaler = joblib.load("models/artifacts/scaler.pkl")
        self.feature_order = joblib.load("models/artifacts/feature_config.pkl")

    def predict(self, data):

        try:
            features = []

            for f in self.feature_order:
                features.append(data.get(f, 0))

            # 🔥 FIX IMPORTANT: keep feature names
            X = pd.DataFrame([features], columns=self.feature_order)

            scaled = self.scaler.transform(X)

            prob = self.model.predict_proba(scaled)[0][1]

            return {
                "churn_probability": float(prob),
                "risk": "HIGH" if prob > 0.5 else "LOW"
            }

        except Exception as e:
            return {"error": str(e)}