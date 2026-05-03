from src.data.load import load_data
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.save_load import save_model
from src.models.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger()


def run_pipeline():

    logger.info("Starting churn pipeline...")

    df = load_data("data/raw/customer_churn_dataset-testing-master.csv")

    df = build_features(df)

    # 🔥 auto-detect target column
    if "churn" not in df.columns:
        if "churned" in df.columns:
            df["churn"] = df["churned"]
        elif "target" in df.columns:
            df["churn"] = df["target"]

    model, scaler, X_test, y_test, feature_config = train_model(df)

    evaluate_model(model, X_test, y_test)

    save_model(model, scaler, feature_config)

    logger.info("Pipeline completed ✔")


if __name__ == "__main__":
    run_pipeline()