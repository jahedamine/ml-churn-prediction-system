import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    # 🔥 Normalize column names (VERY IMPORTANT)
    df.columns = df.columns.str.lower().str.strip()
    df.columns = df.columns.str.replace(" ", "_")

    return df