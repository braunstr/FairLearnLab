from __future__ import annotations

from typing import Literal, Optional

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

from .data_loading import load_adult_income_dataset, load_german_credit_dataset
from .preprocessing import adult_income_preprocessor, german_credit_preprocessor


# Allowed model keys 
ModelKey = Literal["logreg", "tree", "rf", "gb"]
DatasetKey = Literal["adult_income", "german_credit"]


class SampleWeightPipeline(Pipeline):
    """
    Custom sklearn Pipeline which forwards sample_weight to the classifier step

    sklearn.Pipeline.fit does not accept sample_weight directly unless passed to a specific
    step via stepname__sample_weight

    Fairlearn reductions (ExponentiatedGradient) may call estimator.fit(..., sample_weight=...)
    This wrapper intercepts sample_weight and forwards it to the "clf" step
    """
    def fit(self, X, y=None, sample_weight=None, **fit_params):
        # If Fairlearn provides sample weights, forward them to the classifier step
        if sample_weight is not None:
            fit_params["clf__sample_weight"] = sample_weight

        # Delegating to the original Pipeline.fit with correctly formatted fit_params
        return super().fit(X, y, **fit_params)


def _make_classifier(model: ModelKey) -> BaseEstimator:
    """Return an unfitted sklearn classifier for the given model key"""
    if model == "logreg":
        return LogisticRegression(max_iter=1000)
    if model == "tree":
        return DecisionTreeClassifier(random_state=42)
    if model == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if model == "gb":
        return GradientBoostingClassifier(random_state=42)
    raise ValueError(f"Unsupported model '{model}'. Use one of: logreg, tree, rf, gb")


def _make_pipeline(dataset: DatasetKey, model: ModelKey) -> Pipeline:
    """
    Build dataset-specific preprocessing + classifier pipeline
    Returns SampleWeightPipeline so ExponentiatedGradient can pass sample_weight
    """
    if dataset == "adult_income":
        pre = adult_income_preprocessor()
    elif dataset == "german_credit":
        pre = german_credit_preprocessor()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    clf = _make_classifier(model)
    return SampleWeightPipeline(steps=[("preprocess", pre), ("clf", clf)])


def _load(dataset: DatasetKey, split: str):
    if dataset == "adult_income":
        return load_adult_income_dataset(split)
    elif dataset == "german_credit":
        return load_german_credit_dataset(split)
    raise ValueError(f"Unknown dataset '{dataset}'")



# Generic mitigation APIs (work for logreg/tree/rf/gb)

def train_fair_dp(dataset: DatasetKey, model: ModelKey = "logreg", eps: float = 0.01, protected_attr: str = "sex") -> ExponentiatedGradient:
    """
    Train ExponentiatedGradient + DemographicParity for any supported model

    Args:
        dataset: "adult_income" or "german_credit"
        model: "logreg" | "tree" | "rf" | "gb"
        eps: Allowed constraint violation (smaller eps => stricter fairness)
        protected_attr: sensitive attribute column in df ("sex")
    """
    X_train, y_train, _, df_train = _load(dataset, "train")

    if protected_attr not in df_train.columns:
        raise KeyError(f"protected_attr='{protected_attr}' not found in df_train columns.")
    A_train = df_train[protected_attr]

    base_pipeline = _make_pipeline(dataset, model)
    constraint = DemographicParity()

    mitigator = ExponentiatedGradient(estimator=base_pipeline, constraints=constraint, eps=eps)
    mitigator.fit(X_train, y_train, sensitive_features=A_train)
    return mitigator


def train_threshold_optimizer(dataset: DatasetKey, model: ModelKey = "logreg", constraint: str = "equalized_odds", protected_attr: str = "sex", predict_method: Optional[str] = None) -> ThresholdOptimizer:
    """
    Train ThresholdOptimizer (post-processing) for any supported model

    Workflow:
      1) Fit base pipeline on TRAIN
      2) Learn group thresholds on VAL to satisfy the constraint

    Args:
        dataset: "adult_income" or "german_credit"
        model: "logreg" | "tree" | "rf" | "gb"
        constraint: "equalized_odds" or "demographic_parity"
        protected_attr: sensitive attribute column in df ("sex")
        predict_method: usually "predict_proba"; if None, auto-select
    """
    # Train base model on train split
    X_train, y_train, _, _df_train = _load(dataset, "train")
    base_pipeline = _make_pipeline(dataset, model)
    base_pipeline.fit(X_train, y_train)

    # Learn thresholds on validation split
    X_val, y_val, _, df_val = _load(dataset, "val")

    if protected_attr not in df_val.columns:
        raise KeyError(
            f"protected_attr='{protected_attr}' not found in df_val columns."
        )
    A_val = df_val[protected_attr]

    # Auto-pick predict_method if not provided
    if predict_method is None:
        if hasattr(base_pipeline, "predict_proba"):
            predict_method = "predict_proba"
        elif hasattr(base_pipeline, "decision_function"):
            predict_method = "decision_function"
        else:
            predict_method = "predict"

    thresh_opt = ThresholdOptimizer(estimator=base_pipeline, constraints=constraint, predict_method=predict_method)
    thresh_opt.fit(X_val, y_val, sensitive_features=A_val)
    return thresh_opt


# ============================================================
# Backwards-compatible functions, so notebooks wont break
# ============================================================

def adult_income_logreg_pipeline() -> Pipeline:
    
    return _make_pipeline(dataset="adult_income", model="logreg")


def german_credit_logreg_pipeline() -> Pipeline:
    return _make_pipeline(dataset="german_credit", model="logreg")


def train_adult_income_logreg_fair_dp(eps: float = 0.01) -> ExponentiatedGradient:
    return train_fair_dp(dataset="adult_income", model="logreg", eps=eps, protected_attr="sex")


def train_german_credit_logreg_fair_dp(eps: float = 0.01) -> ExponentiatedGradient:
    return train_fair_dp(dataset="german_credit", model="logreg", eps=eps, protected_attr="sex")


def train_adult_income_logreg_threshold(constraint: str = "equalized_odds") -> ThresholdOptimizer:
    return train_threshold_optimizer(dataset="adult_income", model="logreg", constraint=constraint,protected_attr="sex",predict_method="predict_proba")


def train_german_credit_logreg_threshold(constraint: str = "equalized_odds") -> ThresholdOptimizer:
    return train_threshold_optimizer(dataset="german_credit", model="logreg", constraint=constraint, protected_attr="sex", predict_method="predict_proba")