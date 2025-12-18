from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from .data_loading import load_adult_income_dataset, load_german_credit_dataset
from .preprocessing import adult_income_preprocessor, german_credit_preprocessor


class SampleWeightPipeline(Pipeline):
    """
    Custom sklearn Pipeline which forwards sample_weight to the classifier step
    sklearn.Pipeline.fit does not accept sample_weight directly unless passed to a specific step via stepname__sample_weight
    
    This wrapper intercepts sample_weight and forwards it to the "clf" step
    """
    def fit(self, X, y=None, sample_weight=None, **fit_params):

        # if Fairlearn provides sample weights, forward them to the classifier step
        if sample_weight is not None:
            fit_params["clf__sample_weight"] = sample_weight
        
        # Delegating to the original Pipeline.fit with correctly formatted fit_params
        return super().fit(X, y, **fit_params)


def adult_income_logreg_pipeline() -> Pipeline:
    """
    Building the baseline pipeline for Adult Income:
    - preprocess: column transformer (scaling + one-hot)
    - clf: logistic regression classifier

    Returns:
        a SampleWeightPipeline so Fairlearn reductions can pass sample weights
    """
    return SampleWeightPipeline(steps=[("preprocess", adult_income_preprocessor()),("clf", LogisticRegression(max_iter=1000))])


def german_credit_logreg_pipeline() -> Pipeline:
    """
    Building the baseline pipeline for German Credit:
    - preprocess: column transformer (scaling + one-hot)
    - clf: logistic regression classifier

    Returns:
        a SampleWeightPipeline so Fairlearn reductions can pass sample weights
    """
    return SampleWeightPipeline(steps=[("preprocess", german_credit_preprocessor()),("clf", LogisticRegression(max_iter=1000))])


def train_adult_income_logreg_fair_dp(eps: float = 0.01) -> ExponentiatedGradient:
    """
    Training a fairness-mitigated model using Fairlearn's ExponentiatedGradient reduction + DemographicParity

    Demographic Parity aims to equalize the selection_rate across sensitive groups

    Args:
        eps: Allowed constraint violation (smaller eps => stricter fairness, often more accuracy trade-off)

    Returns:
        A fitted ExponentiatedGradient object (acts like an estimator with predict())
    """
    # Loading training split
    X_train, y_train, A_train, df_train = load_adult_income_dataset("train")
    # Explicitly choosing the protected attribute from the raw dataframe
    A_train = df_train["sex"]
    # Base estimator pipeline (preprocess + logistic regression)
    base_pipeline = adult_income_logreg_pipeline()
    # Fairness constraint (Demographic Parity)
    constraint = DemographicParity()
    # ExponentiatedGradient will train a mixture of models that satisfies the fairness constraint (approximately)
    mitigator = ExponentiatedGradient(estimator=base_pipeline,constraints=constraint,eps=eps)
    # Fit the mitigator, passing the sensitive attribute explicitly
    mitigator.fit(X_train, y_train, sensitive_features=A_train)

    return mitigator


def train_german_credit_logreg_fair_dp(eps: float = 0.01) -> ExponentiatedGradient:
    """
    Training a fairness-mitigated model using ExponentiatedGradient + DemographicParity

    Args:
        eps: Allowed constraint violation (smaller => stricter fairness)

    Returns:
        a fitted ExponentiatedGradient object
    """
    X_train, y_train, A_train, df_train = load_german_credit_dataset("train")
    # Protected attribute for German Credit
    A_train = df_train["personal_status_sex"]
    base_pipeline = german_credit_logreg_pipeline()
    constraint = DemographicParity()
    mitigator = ExponentiatedGradient(estimator=base_pipeline,constraints=constraint,eps=eps)
    mitigator.fit(X_train, y_train, sensitive_features=A_train)

    return mitigator

def train_adult_income_logreg_threshold(constraint: str = "equalized_odds") -> ThresholdOptimizer:

    """
    Training a post-processing mitigator (ThresholdOptimizer)

    Approach:
    1) Train a standard (unconstrained) classifier on the training split
    2) Learn group-specific thresholds on the validation split such that a fairness constraintis satisfied (e.g., equalized_odds or demographic_parity)

    Args:
        constraint: ThresholdOptimizer constraint, commonly "equalized_odds" or "demographic_parity"

    Returns:
        a fitted ThresholdOptimizer model
    """
    # Fit a base model on the training set   
    X_train, y_train, A_train, df_train = load_adult_income_dataset("train")
    base_pipeline = adult_income_logreg_pipeline()
    base_pipeline.fit(X_train, y_train)
    # validation set to learn thresholds per sensitive group
    X_val, y_val, A_val, df_val = load_adult_income_dataset("val")
    A_val = df_val["sex"]
    # ThresholdOptimizer requires probabilities to adjust decision thresholds
    thresh_opt = ThresholdOptimizer(estimator=base_pipeline,constraints=constraint,predict_method="predict_proba")
    # Fit thresholds using the validation set
    thresh_opt.fit(X_val, y_val, sensitive_features=A_val)

    return thresh_opt


def train_german_credit_logreg_threshold(constraint: str = "equalized_odds") -> ThresholdOptimizer:
    """
    Training a post-processing mitigator (ThresholdOptimizer)

    Same logic as Adult:
    - Train base model on train split
    - Learn group-specific thresholds on validation split

    Args:
        constraint: Fairness constraint to enforce in post-processing

    Returns:
        a fitted ThresholdOptimizer model
    """
    X_train, y_train, A_train, df_train = load_german_credit_dataset("train")
    base_pipeline = german_credit_logreg_pipeline()
    base_pipeline.fit(X_train, y_train)
    X_val, y_val, A_val, df_val = load_german_credit_dataset("val")
    A_val = df_val["personal_status_sex"]
    thresh_opt = ThresholdOptimizer(estimator=base_pipeline,constraints=constraint,predict_method="predict_proba")
    thresh_opt.fit(X_val, y_val, sensitive_features=A_val)

    return thresh_opt