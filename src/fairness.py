from typing import Dict
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from .data_loading import load_adult_income_dataset, load_german_credit_dataset


def metric_frame(y_true, y_pred, A) -> MetricFrame:
    """
    Thisfunction builds a Fairlearn MetricFrame for a set of predictions and a sensitive feature

    MetricFrame computes:
      - overall metrics (aggregated over all samples)
      - per-group metrics (computed separately for each sensitive group)

    Args:
        y_true: Ground-truth labels
        y_pred: Model predictions (binary class labels)
        A: protected attribute values aligned with y_true/y_pred (sex, personal_status_sex)

    Returns:
        a Fairlearn MetricFrame object with overall and per-group metrics
    """
    #Mix of standard ML metrics and fairness-relevant rates
    #precision/recall/f1 use zero_division=0 to avoid exceptions when a group has no positive predictions
    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate
    }

    # MetricFrame will compute each metric overall + broken down by sensitive groups in A
    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=A)
    return mf


def evaluate_adult_income_fairness(models: Dict[str, Pipeline], protected_attr: str = "sex") -> Dict[str, MetricFrame]:
    """
    Evaluating a set of trained models on the Adult Income test split and print metrics overall + by group

    Args:
        models: Dict mapping model name -> fitted sklearn Pipeline
        protected_attr: Column in the test dataframe to use as sensitive attribute

    Returns:
        Dict mapping model name -> MetricFrame (overall + by_group metrics)
    """
    # Loading test split, df_test is returned to allow selecting alternative protected attributes
    X_test, y_test, A_test, df_test = load_adult_income_dataset("test")

    # If the protected attribute is different than sex, it will be overwritten
    if protected_attr != "sex":
        A_test = df_test[protected_attr]

    results: Dict[str, MetricFrame] = {}

    # Evaluating each model on the same test set
    for name, model in models.items():
        # Predicting class labels for fairness/accuracy metrics
        y_pred = model.predict(X_test)
        # Building MetricFrame for overall + per-group evaluation
        mf = metric_frame(y_test, y_pred, A_test)
        results[name] = mf

        print(f"\nAdult income model: {name} (protected: {protected_attr})")
        print("Overall metrics:")
        print(mf.overall)
        print("\nBy group:")
        print(mf.by_group)

    return results


def evaluate_german_credit_fairness(models: Dict[str, Pipeline], protected_attr: str = "personal_status_sex") -> Dict[str, MetricFrame]:
    """
    Evaluating a set of trained models on the German Credit test split and print metrics overall + by group

    Args:
        models: Dict mapping model name -> fitted sklearn Pipeline
        protected_attr: Column in the test dataframe to use as sensitive attribute

    Returns:
        Dict mapping model name -> MetricFrame (overall + by_group metrics)
    """
   
    X_test, y_test, A_test, df_test = load_german_credit_dataset("test")

    # If the protected attribute is different than personal_status_sex, it will be overwritten
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
    """
    This method converts a MetricFrame into a flat dict row suitable for building result tables

    Uses:
      - mf.overall for performance metrics
      - mf.difference() for absolute group gaps (max - min across groups)
      - mf.ratio() for ratio-based disparities (min / max across groups)

    Notes:
      - Statistical parity difference is derived from selection_rate difference
      - Disparate impact ratio is derived from selection_rate ratio
      - Equal opportunity difference is derived from TPR difference
      - Predictive parity difference is proxied via precision difference

    Args:
        mf: MetricFrame from metric_frame(...)
        model_name: Human-readable model identifier
        dataset_name: Dataset identifier ("adult_income", "german_credit")

    Returns:
        a dict representing one row in a results table
    """

    # Overall metrics
    overall = mf.overall 

    # Fairness gaps computed across groups:
    # - difference(): absolute gap (max - min) per metric
    # - ratio(): min/max per metric (useful for Disparate Impact)   
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

    """
    This method builds a summary table (DataFrame) from multiple MetricFrames

    Args:
        results: Dict mapping model name -> MetricFrame (from evaluate_* functions)
        dataset_name: Dataset label written into each row

    Returns:
        a DataFrame with one row per model containing performance + key fairness metrics
    """
    rows = []
    for model_name, mf in results.items():

        # Flattening MetricFrame into one row of summary metrics
        rows.append(fairness_summary_from_metric_frame(mf, model_name, dataset_name))
        
    return pd.DataFrame(rows)



def calibration_within_groups(models: Dict[str, Pipeline], X_test, y_test, A_test, dataset_name: str) -> pd.DataFrame:

    """
    Computing a simple calibration-within-groups diagnostic using the Brier score

    Brier score measures probabilistic calibration quality:
      - lower is better
      - computed overall + per sensitive group using MetricFrame

    This function reports:
      - overall Brier score
      - min/max Brier score across groups
      - difference between worst and best group (calibration disparity)

    Args:
        models: Dict of trained models (some may not support predict_proba)
        X_test: Test features
        y_test: Test labels
        A_test: Sensitive attribute values aligned with X_test/y_test
        dataset_name: Dataset identifier

    Returns:
        a DataFrame with one row per model that supports predict_proba
    """
    rows = []
    for model_name, model in models.items():
        # Only models with predict_proba can provide probabilities for calibration metrics
        if not hasattr(model, "predict_proba"):
            continue

        # Using predicted probability of the positive class
        y_prob = model.predict_proba(X_test)[:, 1]

        # MetricFrame here uses y_pred as probabilities (allowed because brier_score_loss expects probabilities)
        mf_cal = MetricFrame(metrics={"brier": brier_score_loss}, y_true=y_test, y_pred=y_prob, sensitive_features=A_test)

        overall = mf_cal.overall["brier"]
        by_group = mf_cal.by_group["brier"]

        # Collect summary statistics
        rows.append({"dataset": dataset_name, "model": model_name, "brier_overall": overall, "brier_min_group": by_group.min(), "brier_max_group": by_group.max(), "brier_diff": by_group.max() - by_group.min()})

    return pd.DataFrame(rows)