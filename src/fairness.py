# src/fairness.py


from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from .data_loading import load_adult_income_dataset, load_german_credit_dataset


def metric_frame(y_true, y_pred, A) -> MetricFrame:
  
    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate
    }

    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=A)
    return mf


def evaluate_adult_income_fairness(models: Dict[str, Pipeline], protected_attr: str = "sex") -> Dict[str, MetricFrame]:
    
    X_test, y_test, A_test, df_test = load_adult_income_dataset("test")

    if protected_attr != "sex":
        A_test = df_test[protected_attr]

    results: Dict[str, MetricFrame] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        mf = metric_frame(y_test, y_pred, A_test)
        results[name] = mf

        print(f"\nAdult income model: {name} (protected: {protected_attr})")
        print("Overall metrics:")
        print(mf.overall)
        print("\nBy group:")
        print(mf.by_group)

    return results


def evaluate_german_credit_fairness(models: Dict[str, Pipeline], protected_attr: str = "personal_status_sex") -> Dict[str, MetricFrame]:
   
    X_test, y_test, A_test, df_test = load_german_credit_dataset("test")

    if protected_attr != "personal_status_sex":
        A_test = df_test[protected_attr]

    results: Dict[str, MetricFrame] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        mf = metric_frame(y_test, y_pred, A_test)
        results[name] = mf

        print(f"\nGerman credit model: {name} (protected: {protected_attr})")
        print("Overall metrics:")
        print(mf.overall)
        print("\nBy group:")
        print(mf.by_group)

    return results



def fairness_summary_from_metric_frame(mf: MetricFrame, model_name: str, dataset_name: str) -> dict:
    
    overall = mf.overall    
    diff = mf.difference()   
    ratio = mf.ratio()       

    return {
        "dataset": dataset_name,
        "model": model_name,
        "accuracy": overall["accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "statistical_parity_diff": diff["selection_rate"],
        "disparate_impact_ratio": ratio["selection_rate"],
        "equal_opportunity_diff": diff["tpr"],
        "predictive_parity_diff": diff["precision"]
}


def summarize_fairness_results(results: Dict[str, MetricFrame], dataset_name: str) -> pd.DataFrame:

    rows = []
    for model_name, mf in results.items():

        rows.append(fairness_summary_from_metric_frame(mf, model_name, dataset_name))
        
    return pd.DataFrame(rows)



def calibration_within_groups(models: Dict[str, Pipeline], X_test, y_test, A_test, dataset_name: str) -> pd.DataFrame:

    rows = []
    for model_name, model in models.items():
        
        if not hasattr(model, "predict_proba"):
            continue

        y_prob = model.predict_proba(X_test)[:, 1]
        mf_cal = MetricFrame(metrics={"brier": brier_score_loss}, y_true=y_test, y_pred=y_prob, sensitive_features=A_test)

        overall = mf_cal.overall["brier"]
        by_group = mf_cal.by_group["brier"]

        rows.append({"dataset": dataset_name, "model": model_name, "brier_overall": overall, "brier_min_group": by_group.min(), "brier_max_group": by_group.max(), "brier_diff": by_group.max() - by_group.min()})

    return pd.DataFrame(rows)
