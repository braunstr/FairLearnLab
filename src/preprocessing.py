from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature definitions (Adult Income)
# Numeric columns: scaled to zero-mean/unit-variance for models like Logistic Regression
ADULT_NUMERIC = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
# Categorical columns: one-hot encoded to convert categories into model-friendly numeric features
ADULT_CATEGORICAL = ["workclass","education","marital_status","occupation","relationship","race","native_country"]


# Feature definitions (German Credit)
# Numeric columns: scaled for consistent magnitude across features
GERMAN_NUMERIC = ["duration_months","credit_amount","installment_rate","residence_since","age","existing_credits","people_liable"]

# Categorical columns: one-hot encoded
GERMAN_CATEGORICAL = ["status_existing_checking","credit_history","purpose","savings","employment_since","other_debtors","other_installment_plans","housing","property","job","telephone","foreign_worker"]

def adult_income_preprocessor():
    """
    This function builds the preprocessing pipeline for the Adult Income dataset

    Uses a ColumnTransformer to apply:
      - StandardScaler on numeric features
      - OneHotEncoder on categorical features

    Returns:
        a ColumnTransformer that can be used inside an sklearn Pipeline
    """
    preprocessor = ColumnTransformer(
        transformers = [
            # Scaling numeric features in order to models don't get biased by raw magnitudes
            ("num", StandardScaler(), ADULT_NUMERIC),
            # One-hot encode categorical features
            # handle_unknown="ignore" prevents errors if unseen categories appear in val/test
            ("cat", OneHotEncoder(handle_unknown="ignore"),ADULT_CATEGORICAL)
        ]
    )
    return preprocessor

def german_credit_preprocessor():
    """
    This function builds the preprocessing pipeline for the German Credit dataset

    Uses a ColumnTransformer to apply:
      - StandardScaler on numeric features
      - OneHotEncoder on categorical features

    Returns:
        a ColumnTransformer that can be used inside an sklearn Pipeline
    """
    preprocessor= ColumnTransformer(
        transformers = [
            ("num",StandardScaler(),GERMAN_NUMERIC),
            ("cat",OneHotEncoder(handle_unknown="ignore"),GERMAN_CATEGORICAL)
        ]
    )
    return preprocessor

