from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"

def load_csv(name: str) -> pd.DataFrame:
    path = DATA_PROC / name
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)

def load_adult_income_dataset(split:str = "train"):

    if split not in ["train", "val", "test"]:
        raise ValueError("split must be one of 'train', 'val', or 'test'")

    df = load_csv(f"adult_{split}.csv")
    y = df["income_binary"]
    A = df["sex"]
    X = df.drop(columns=["income_binary", "income"])

    return X, y, A, df

def load_german_credit_dataset(split:str = "train"):

    if split not in ["train", "val", "test"]:
        raise ValueError("split must be one of 'train', 'val', or 'test'")

    df= load_csv(f"german_{split}.csv")
    y = df["credit_risk_binary"]
    A = df["personal_status_sex"]
    X = df.drop(columns=["credit_risk_binary", "credit_risk", "target_raw"])
   
    return X, y, A, df


