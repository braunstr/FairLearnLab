
from typing import Dict

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from .data_loading import load_adult_income_dataset, load_german_credit_dataset
from .preprocessing import adult_income_preprocessor, german_credit_preprocessor


def get_baseline_classifiers() -> Dict[str, object]:
    """
    This method defines a small set of baseline classifiers for comparison

    Returns:
        Dict mapping a short model name -> an unfitted sklearn estimator
        These are used across both datasets for consistent benchmarking
    """
    return {
        # Simple baseline: always predicts the most frequent class (majority label)
        "dummy": DummyClassifier(strategy="most_frequent"),
        # Linear baseline: strong and interpretable for tabular data
        "logreg": LogisticRegression(max_iter=1000),
        # Non-linear baseline: decision tree (prone to overfitting, but good reference)
        "tree": DecisionTreeClassifier(random_state=42),
        # Ensemble baseline: random forest (robust non-linear model)
        # n_estimators fixed for stable results across runs
        "rf": RandomForestClassifier(n_estimators=200,random_state=42),
        # Boosting baseline: gradient boosting (often strong on tabular data)
        "gb": GradientBoostingClassifier(random_state=42)
    }


def train_adult_income_baselines():

    """
    Training baseline models 

    Workflow:
        - load train + validation split
        - build a preprocessing + classifier pipeline for each estimator
        - fit on train split
        - evaluate on validation split (accuracy only)
        - returns dict of fitted pipelines

    Returns:
        Dict mapping model name -> fitted sklearn Pipeline
    """
    # Loading data splits 
    X_train, y_train, A_train, df_train = load_adult_income_dataset("train")
    X_val, y_val, A_val, df_val = load_adult_income_dataset("val")
    # Get dataset-specific preprocessing (scaling numeric + one-hot encoding categoricals)
    preprocessor = adult_income_preprocessor()
    # Baseline estimators to train
    estimators = get_baseline_classifiers()
    # Collecting fitted pipelines so they can be evaluated later (fairness metrics, test set, etc.)
    models: Dict[str, Pipeline] = {}

    for name, clf in estimators.items():
        # Standard sklearn pattern: preprocessing + model in one pipeline
        pipe = Pipeline(steps=[("preprocess", preprocessor),("clf", clf)])
        # Fit model on training data
        pipe.fit(X_train, y_train)
        # Storing fitted model for downstream evaluation (fairness metrics, mitigations, etc.)
        models[name] = pipe
        # Quick validation accuracy (sanity check / baseline performance reference)
        y_val_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"[Adult income] {name}: val accuracy = {acc:.3f}")
        
    return models


def train_german_credit_baselines():
    """
    Training baseline models

    Same workflow as Adult:
        - load train + validation split
        - pipeline(preprocess -> classifier)
        - fit on train, evaluate on validation
        - return fitted pipelines

    Returns:
        Dict mapping model name -> fitted sklearn Pipeline
    """
    X_train, y_train, A_train, df_train = load_german_credit_dataset("train")
    X_val, y_val, A_val, df_val = load_german_credit_dataset("val")
    preprocessor = german_credit_preprocessor()
    estimators = get_baseline_classifiers()
    models: Dict[str, Pipeline] = {}

    for name, clf in estimators.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor),("clf", clf)])
        pipe.fit(X_train, y_train)
        models[name] = pipe
        y_val_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"[German credit] {name}: val accuracy = {acc:.3f}")

    return models
