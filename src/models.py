
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

    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000),
        "tree": DecisionTreeClassifier(random_state=42),
        "rf": RandomForestClassifier(n_estimators=200,random_state=42),
        "gb": GradientBoostingClassifier(random_state=42)
    }


def train_adult_income_baselines():
    
    X_train, y_train, A_train, df_train = load_adult_income_dataset("train")
    X_val, y_val, A_val, df_val = load_adult_income_dataset("val")
    preprocessor = adult_income_preprocessor()
    estimators = get_baseline_classifiers()
    models: Dict[str, Pipeline] = {}

    for name, clf in estimators.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor),("clf", clf)])
        pipe.fit(X_train, y_train)
        models[name] = pipe
        y_val_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"[Adult income] {name}: val accuracy = {acc:.3f}")
    return models


def train_german_credit_baselines():
    
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
