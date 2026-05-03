from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def train_model(df):

    if "churn" not in df.columns:
        raise ValueError("Missing target column 'churn'")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    # 🔥 FIX 1: encode categorical variables
    X = pd.get_dummies(X)

    # 🔥 align train/test safety
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test, X.columns.tolist()