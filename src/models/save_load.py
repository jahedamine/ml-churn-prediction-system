import joblib
import os


def save_model(model, scaler, feature_config):

    os.makedirs("models/artifacts", exist_ok=True)

    joblib.dump(model, "models/artifacts/model.pkl")
    joblib.dump(scaler, "models/artifacts/scaler.pkl")
    joblib.dump(feature_config, "models/artifacts/feature_config.pkl")