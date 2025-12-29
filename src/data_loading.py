from pathlib import Path
import pandas as pd

# Resolving project root based on this file
# <project_root>/src/data_loading.py -> parents[1] == <project_root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROC = PROJECT_ROOT / "data" / "processed"

def load_csv(name: str) -> pd.DataFrame:
    """
    This function loads a processed CSV file from data/processed

    Args:
        name: Filename of the processed CSV

    Returns:
        a DataFrame containing the CSV content

    Raises:
        FileNotFoundError: If the requested file does not exist
    """
    path = DATA_PROC / name
    # fails if the file is missing
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)

def load_adult_income_dataset(split:str = "train"):
    """
    This function loads the Adult Income dataset split and returns (X, y, A, df)

    X: Features (all columns except label columns and protected attribute)
    y: Binary target label (income_binary)
    A: Sensitive/protected attribute used for fairness evaluation (sex)
    df: Full DataFrame for the split 
    Args:
        split: One of "train", "val", "test"

    Returns:
        X (DataFrame), y (Series), A (Series), df (DataFrame)

    Raises:
        ValueError: If split is not one of the supported values
    """
    # Validating the split
    if split not in ["train", "val", "test"]:
        raise ValueError("split must be one of 'train', 'val', or 'test'")
    
    # Loading the processed split
    df = load_csv(f"adult_{split}.csv")
    # Target variable used for model training/evaluation
    y = df["income_binary"]
    # Sensitive/protected attribute used for fairness metrics
    A = df["sex"]
    # Feature matrix: dropping labels (binary + original text label and sex) so only predictors remain
    X = df.drop(columns=["income_binary", "income", "sex"])

    return X, y, A, df

def load_german_credit_dataset(split:str = "train"):
    """
    This function loads the German Credit dataset split and return (X, y, A, df)

    X: Features (all columns except label columns and protected attribute)
    y: Binary target label (credit_risk_binary)
    A: Sensitive/protected attribute used for fairness evaluation (personal_status_sex)
    df: Full DataFrame for the split 

    Args:
        split: One of "train", "val", "test"

    Returns:
        X (DataFrame), y (Series), A (Series), df (DataFrame)

    Raises:
        ValueError: If split is not one of the supported values
    """
    # Validating the split
    if split not in ["train", "val", "test"]:
        raise ValueError("split must be one of 'train', 'val', or 'test'")
    # Loading the processed split
    df= load_csv(f"german_{split}.csv")
    # Target variable used for model training/evaluation
    y = df["credit_risk_binary"]
    # Sensitive/protected attribute used for fairness metrics (derived from personal_status_sex)
    A = df["sex"]
    # Feature matrix: dropping labels (binary + original text label and sex) so only predictors remain
    X = df.drop(columns=["credit_risk_binary", "credit_risk", "target_raw", "personal_status_sex", "sex"])
   
    return X, y, A, df


