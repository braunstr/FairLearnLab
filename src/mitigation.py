from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from .data_loading import load_adult_income_dataset, load_german_credit_dataset
from .preprocessing import adult_income_preprocessor, german_credit_preprocessor


class SampleWeightPipeline(Pipeline):
    
    def fit(self, X, y=None, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params["clf__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


def adult_income_logreg_pipeline() -> Pipeline:
    return SampleWeightPipeline(steps=[("preprocess", adult_income_preprocessor()),("clf", LogisticRegression(max_iter=1000))])


def german_credit_logreg_pipeline() -> Pipeline:
    return SampleWeightPipeline(steps=[("preprocess", german_credit_preprocessor()),("clf", LogisticRegression(max_iter=1000))])


def train_adult_income_logreg_fair_dp(eps: float = 0.01) -> ExponentiatedGradient:
    
    X_train, y_train, A_train, df_train = load_adult_income_dataset("train")
    A_train = df_train["sex"]
    base_pipeline = adult_income_logreg_pipeline()
    constraint = DemographicParity()
    mitigator = ExponentiatedGradient(estimator=base_pipeline,constraints=constraint,eps=eps)
    mitigator.fit(X_train, y_train, sensitive_features=A_train)

    return mitigator


def train_german_credit_logreg_fair_dp(eps: float = 0.01) -> ExponentiatedGradient:
    
    X_train, y_train, A_train, df_train = load_german_credit_dataset("train")
    A_train = df_train["personal_status_sex"]
    base_pipeline = german_credit_logreg_pipeline()
    constraint = DemographicParity()
    mitigator = ExponentiatedGradient(estimator=base_pipeline,constraints=constraint,eps=eps)
    mitigator.fit(X_train, y_train, sensitive_features=A_train)

    return mitigator

def train_adult_income_logreg_threshold(constraint: str = "equalized_odds") -> ThresholdOptimizer:
    
   
    X_train, y_train, A_train, df_train = load_adult_income_dataset("train")
    base_pipeline = adult_income_logreg_pipeline()
    base_pipeline.fit(X_train, y_train)
    X_val, y_val, A_val, df_val = load_adult_income_dataset("val")
    A_val = df_val["sex"]
    thresh_opt = ThresholdOptimizer(estimator=base_pipeline,constraints=constraint,predict_method="predict_proba")
    thresh_opt.fit(X_val, y_val, sensitive_features=A_val)

    return thresh_opt


def train_german_credit_logreg_threshold(constraint: str = "equalized_odds") -> ThresholdOptimizer:
    
    X_train, y_train, A_train, df_train = load_german_credit_dataset("train")
    base_pipeline = german_credit_logreg_pipeline()
    base_pipeline.fit(X_train, y_train)
    X_val, y_val, A_val, df_val = load_german_credit_dataset("val")
    A_val = df_val["personal_status_sex"]
    thresh_opt = ThresholdOptimizer(estimator=base_pipeline,constraints=constraint,predict_method="predict_proba")
    thresh_opt.fit(X_val, y_val, sensitive_features=A_val)

    return thresh_opt