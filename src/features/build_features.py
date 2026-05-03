def build_features(df):

    required_cols = [
        "last_interaction",
        "num_sessions",
        "purchase_frequency",
        "support_tickets",
        "tenure_months"
    ]

    # 🔥 safety check
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    df["recency_score"] = 1 / (df["last_interaction"] + 1)

    df["engagement_score"] = df["num_sessions"] * df["purchase_frequency"]

    df["support_interactions"] = df["support_tickets"] / (df["tenure_months"] + 1)

    return df