from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Resolving the project root folder based on this file location.
# Example: <project_root>/src/preprocessing_data.py -> parents[1] == <project_root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Defining input/output directories for raw and processed data
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"

# Ensuring the processed data directory exists
DATA_PROC.mkdir(parents=True, exist_ok=True)


def prepare_adult_income_data_symbolic():

    """
    Loading the UCI Adult Income dataset files (adult.data + adult.test),
    cleaning them, creating a binary target column, and writing train/val/test CSVs.
    """

    # Simple console output to make script runs easier to follow
    print("=======================")
    print("Preparing Adult dataset")
    print("=======================")

    # Column names as defined in adult.names / UCI documentation
    cols_adult = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
    

    # Loading "adult.data" (train+val pool)
    # - header=None: file has no header row
    # - names=cols_adult: assigning human-readable column names
    # - na_values="?": treat '?' entries as missing values
    # - skipinitialspace=True: trim leading spaces after delimiters
    adult_trainval = pd.read_csv(DATA_RAW / "adult.data", header=None, names=cols_adult, na_values="?", skipinitialspace=True)

     # Removing extra whitespace that can appear in the label column
    adult_trainval["income"] = adult_trainval["income"].str.strip()

    # Dropping rows with any missing values to keep preprocessing simple and consistent
    adult_trainval = adult_trainval.dropna().reset_index(drop=True)

    # Converting the string target into a binary target for classification
    adult_trainval["income_binary"] = adult_trainval["income"].map({"<=50K": 0, ">50K": 1})
   
    # Loading "adult.test" (official test set) 
    # adult.test contains comment lines starting with '|' that should be ignored
    adult_test = pd.read_csv(DATA_RAW / "adult.test", header=None, names=cols_adult, na_values="?", skipinitialspace=True, comment="|")

    # In adult.test, labels often end with a period (e.g., "<=50K.")
    # Removing '.' and strip whitespace to match the mapping keys
    adult_test["income"] = (adult_test["income"].astype(str).str.replace(".", "", regex=False).str.strip())

    # Dropping missing values and rebuild a clean index
    adult_test = adult_test.dropna().reset_index(drop=True)

    # Creating the binary target for the test set
    adult_test["income_binary"] = adult_test["income"].map({"<=50K": 0, ">50K": 1})
   
    # Spliting train vs validation from the original training pool
    # stratify ensures class balance between train and val (important for fair comparison)
    train_df, val_df = train_test_split(adult_trainval, test_size=0.2, random_state=42, stratify=adult_trainval["income_binary"])
    
    # Print dataset sizes to verify splits quickly
    print(f"Adult train shape: {train_df.shape}")
    print(f"Adult test shape: {adult_test.shape}")
    print(f"Adult val shape: {val_df.shape}")
    
    # Saving splits as CSV so downstream steps always load the same data consistently
    train_df.to_csv(DATA_PROC / "adult_train.csv", index=False)
    val_df.to_csv(DATA_PROC / "adult_val.csv", index=False)
    adult_test.to_csv(DATA_PROC / "adult_test.csv", index=False)


def prepare_german_credit_data_symbolic():

    """
    Loading the German Credit dataset (symbolic/categorical version),
    create readable + binary targets, and write train/val/test CSVs.
    """

    print("==========================================")
    print("Preparing German Credit (symbolic) dataset")
    print("==========================================")

    # Column names based on the UCI German Credit documentation 
    cols_credit = ["status_existing_checking","duration_months","credit_history",
                   "purpose","credit_amount","savings","employment_since","installment_rate",
                   "personal_status_sex","other_debtors","residence_since","property","age",
                   "other_installment_plans","housing","existing_credits","job","people_liable",
                   "telephone","foreign_worker","target_raw"]
    
    # Loading the dataset (space separated, no header row)
    german = pd.read_csv(DATA_RAW / "german.data", sep=" ", header=None, names=cols_credit)

    german["personal_status_sex"] = german["personal_status_sex"].astype(str).str.strip()
    # Deriving a clean binary sex attribute from the combined personal_status_sex codes
    PERSONAL_STATUS_SEX_TO_SEX = {
    "A91": "male",   # male: divorced/separated
    "A92": "female", # female: divorced/separated/married
    "A93": "male",   # male: single
    "A94": "male",   # male: married/widowed
    "A95": "female", # female: single
    }

    german["sex"] = german["personal_status_sex"].map(PERSONAL_STATUS_SEX_TO_SEX)

    # Safety check: ensure all codes were mapped successfully
    if german["sex"].isna().any():
        unknown = sorted(german.loc[german["sex"].isna(), "personal_status_sex"].unique().tolist())
        raise ValueError(f"Unknown personal_status_sex codes: {unknown}")

    german["sex"] = german["sex"].astype("category")
    
    # Creating a human-readable target label
    german["credit_risk"] = german["target_raw"].map({1: "good", 2: "bad"})

    # Creating a binary target for ML:
    german["credit_risk_binary"] = german["target_raw"].map({1: 1, 2: 0})

    # Splitting off the test set (80/20), stratified by the binary target
    train_full, test_df = train_test_split(german, test_size=0.2, random_state=42, stratify=german["credit_risk_binary"])

    # Splitting the remaining 80% into train/val:
    # test_size=0.25 of 80% => 20% overall validation set
    train_df, val_df = train_test_split(train_full, test_size=0.25, random_state=42, stratify=train_full["credit_risk_binary"])

    
    # Printing dataset sizes to verify splits quickly
    print(f"German train shape: {train_df.shape}")
    print(f"German test shape: {test_df.shape}")
    print(f"German val shape: {val_df.shape}")
    
    # Saving splits as CSV
    train_df.to_csv(DATA_PROC / "german_train.csv", index=False)
    val_df.to_csv(DATA_PROC / "german_val.csv", index=False)
    test_df.to_csv(DATA_PROC / "german_test.csv", index=False)



def main():
    prepare_adult_income_data_symbolic()
    prepare_german_credit_data_symbolic()
    print("All done.")


if __name__ == "__main__":
    main()

