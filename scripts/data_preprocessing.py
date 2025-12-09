from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)


def prepare_adult_income_data_symbolic():

    print("=======================")
    print("Preparing Adult dataset")
    print("=======================")

    cols_adult = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
    
    adult_trainval = pd.read_csv(DATA_RAW / "adult.data", header=None, names=cols_adult, na_values="?", skipinitialspace=True)
    adult_trainval["income"] = adult_trainval["income"].str.strip()
    adult_trainval = adult_trainval.dropna().reset_index(drop=True)
    adult_trainval["income_binary"] = adult_trainval["income"].map({"<=50K": 0, ">50K": 1})
   
    adult_test = pd.read_csv(DATA_RAW / "adult.test", header=None, names=cols_adult, na_values="?", skipinitialspace=True, comment="|")# ignore comment lines starting with '|'
    adult_test["income"] = (adult_test["income"].astype(str).str.replace(".", "", regex=False).str.strip())
    adult_test = adult_test.dropna().reset_index(drop=True)
    adult_test["income_binary"] = adult_test["income"].map({"<=50K": 0, ">50K": 1})
   
    train_df, val_df = train_test_split(adult_trainval, test_size=0.2, random_state=42, stratify=adult_trainval["income_binary"])
    
    print(f"Adult train shape: {train_df.shape}")
    print(f"Adult test shape: {adult_test.shape}")
    print(f"Adult val shape: {val_df.shape}")
    

    train_df.to_csv(DATA_PROC / "adult_train.csv", index=False)
    val_df.to_csv(DATA_PROC / "adult_val.csv", index=False)
    adult_test.to_csv(DATA_PROC / "adult_test.csv", index=False)


def prepare_german_credit_data_symbolic():

    print("==========================================")
    print("Preparing German Credit (symbolic) dataset")
    print("==========================================")

    cols_credit = ["status_existing_checking","duration_months","credit_history","purpose","credit_amount","savings","employment_since","installment_rate","personal_status_sex","other_debtors","residence_since","property","age","other_installment_plans","housing","existing_credits","job","people_liable","telephone","foreign_worker","target_raw"]
   
    german = pd.read_csv(DATA_RAW / "german.data", sep=" ", header=None, names=cols_credit)
    german["credit_risk"] = german["target_raw"].map({1: "good", 2: "bad"})
    german["credit_risk_binary"] = german["target_raw"].map({1: 1, 2: 0})
   
    train_full, test_df = train_test_split(german, test_size=0.2, random_state=42, stratify=german["credit_risk_binary"])
    train_df, val_df = train_test_split(train_full, test_size=0.25, random_state=42, stratify=train_full["credit_risk_binary"])
    
    print(f"German train shape: {train_df.shape}")
    print(f"German test shape: {test_df.shape}")
    print(f"German val shape: {val_df.shape}")
    
    train_df.to_csv(DATA_PROC / "german_train.csv", index=False)
    val_df.to_csv(DATA_PROC / "german_val.csv", index=False)
    test_df.to_csv(DATA_PROC / "german_test.csv", index=False)



def main():
    prepare_adult_income_data_symbolic()
    prepare_german_credit_data_symbolic()
    print("All done.")


if __name__ == "__main__":
    main()

