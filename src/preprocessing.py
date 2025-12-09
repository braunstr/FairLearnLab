from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADULT_NUMERIC = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
ADULT_CATEGORICAL = ["workclass","education","marital_status","occupation","relationship","race","sex","native_country"]

GERMAN_NUMERIC = ["duration_months","credit_amount","installment_rate","residence_since","age","existing_credits","people_liable"]
GERMAN_CATEGORICAL = ["status_existing_checking","credit_history","purpose","savings","employment_since","personal_status_sex","other_debtors","other_installment_plans","housing","property","job","telephone","foreign_worker"]

def adult_preprocessor():
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", StandardScaler(), ADULT_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"),ADULT_CATEGORICAL)
        ]
    )
    return preprocessor

def german_preprocessor():
    preprocessor= ColumnTransformer(
        transformers = [
            ("num",StandardScaler(),GERMAN_NUMERIC),
            ("cat",OneHotEncoder(handle_unknown="ignore"),GERMAN_CATEGORICAL)
        ]
    )
    return preprocessor

