import joblib
import pandas as pd
from src.evaluation.metrics import evaluate


def run_evaluation():

    print("📊 Loading dataset...")

    df = pd.read_csv("data/raw/customer_churn_dataset-testing-master.csv")

    df.columns = df.columns.str.lower().str.strip()
    df.columns = df.columns.str.replace(" ", "_")

    if "churn" not in df.columns and "churned" in df.columns:
        df["churn"] = df["churned"]

    print("⚙️ Loading model...")

    model = joblib.load("models/artifacts/model.pkl")
    scaler = joblib.load("models/artifacts/scaler.pkl")
    feature_config = joblib.load("models/artifacts/feature_config.pkl")

    X = df[feature_config]
    y = df["churn"]

    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)

    results = evaluate(y, y_pred)

    print("\n📊 FINAL EVALUATION REPORT")
    print("---------------------------")
    print(results)


if __name__ == "__main__":
    run_evaluation()