# FairLearnLab — Developing a Framework for Measuring and Optimizing Algorithmic Fairness in Machine Learning Models

This project contains the code and experiments for my **bachelor thesis** on **measuring and improving algorithmic fairness** in supervised **tabular classification** tasks.  
Using the **UCI Adult Census Income** dataset and the **Statlog (German Credit Data)** dataset, the project trains baseline models, evaluates performance and fairness metrics across protected groups, and applies mitigation techniques using **Fairlearn** (e.g., Demographic Parity constraints and threshold optimization).

The outputs of this project are reproducible **CSV tables** and **PNG plots** (generated through notebooks and saved under `results/`), which are used directly for the thesis evaluation and discussion.


## Scope / What’s included

This project contains a full, reproducible experimentation pipeline for **fairness evaluation and mitigation** on two tabular classification datasets.

### Datasets

- **UCI Adult Census Income**
  - Target: `income_binary` (<=50K vs >50K)
  - Protected attribute evaluated: `sex`

- **Statlog (German Credit Data)**
  - Target: `credit_risk_binary` (good vs bad)
  - Protected attribute evaluated: `sex` *(derived from the original `personal_status_sex` attribute)*

> Note: In the German Credit dataset, the original attribute `personal_status_sex` encodes personal status together with sex.  
> For a consistent **sex-focused** fairness evaluation across datasets, this project derives a clean binary `sex` column during preprocessing and uses it as the protected attribute.


## What the code does

### Data preprocessing & splits
- Converts the original dataset files into processed CSV files.
- Creates fixed **train / validation / test** splits.
- Stores processed data under: `data/processed/`.
- For German Credit: derives `sex` from `personal_status_sex` and stores the derived column in the processed splits.


### Baseline model training
Trains multiple baseline classifiers with a shared preprocessing pipeline:
- DummyClassifier
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Baseline performance & fairness results are saved to `results/`.

### Fairness evaluation
Computes *performance metrics* and *fairness metrics* using `fairlearn.metrics.MetricFrame`, including:
- Accuracy, Precision, Recall, F1 
- Statistical Parity Difference (via selection rate gap)
- Disparate Impact Ratio (via selection rate ratio)
- Equal Opportunity Difference (TPR gap)
- Average Odds Difference (TPR/FPR gap summary)
- Predictive Parity Difference (precision gap)

**Calibration within groups** is measured using the **Brier score** (**baseline models only**):
- overall Brier score
- group Brier gap (max group − min group)

### Mitigation techniques (Fairlearn)

#### In-processing: ExponentiatedGradient + DemographicParity
- Applied to all baseline models that produce meaningful outputs:
  - `logreg`, `tree`, `rf`, `gb` (dummy excluded)
- Uses an `eps` sweep to visualize the fairness/performance trade-off
- Runs with a **light/heavy** split to manage runtime:
  - light: `logreg`, `tree`
  - heavy: `rf`, `gb`

#### Post-processing: ThresholdOptimizer
- Applied to: `logreg`, `tree`, `rf`, `gb` (dummy excluded)
- Evaluated under two constraints:
  - `equalized_odds`
  - `demographic_parity`
- Requires access to the protected attribute at prediction time via `sensitive_features`


## Outputs (used for the thesis)

- Result tables are saved as **CSV** in `results/`
  - baseline fairness summaries
  - mitigation results (all models)
  - legacy logreg-only exports (for backwards compatibility)
  - calibration tables (baseline models)
  - final summary tables (e.g., Pareto/frontier outputs)

- Figures are saved as **PNG** in `results/plots/`
  - data diagnostics
  - baseline comparisons
  - trade-off plots (accuracy vs fairness metrics)
  - mitigation comparisons and summary plots

## Project structure

FairLearnLab/
    data/                       # Included for reproducibility
        processed/              # Generated CSV splits used to train models
            adult_test.csv
            adult_train.csv
            adult_val.csv
            german_test.csv
            german_train.csv
            german_val.csv
        raw/                    # Original downloaded files (not modified)
            adult.data
            adult.names
            adult.test
            german.data

    notebooks/                  # Jupyter notebooks (experiments + plots + exports)
        01_environment_and_data_sanity.ipynb
        02_baselines_train_eval.ipynb
        03_calibration_within_groups.ipynb
        04_mitigation_exponentiated_gradient_dp.ipynb
        05_mitigation_threshold_optimizer.ipynb
        06_final_results_and_plots.ipynb

    results/
        ...                     # All CSV files generated during experiments
        plots/
            ...                 # All plots generated during experiments


    scripts/                    # Data conversion + fixed train/val/test splits
        data_preprocessing.py

    src/                        # Reusable Python code (importable modules)
        __init__.py
        data_loading.py
        fairness.py
        mitigation.py
        models.py
        preprocessing.py

    README.md
    requirements.txt            # Environment dependencies
    .gitignore


### Notes
- `src/` contains reusable framework code (FairLearnLab)
- `notebooks/` orchestrate experiments and export:
  - CSV result tables → `results/`
  - PNG figures → `results/plots/`

## Install and run FairLearnLab

**Tested with Python 3.12.**

### Create & activate a virtual environment

**Windows (PowerShell)**
- `py -3.12 -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`

**macOS / Linux**
- `python3.12 -m venv .venv`
- `source .venv/bin/activate`

### Install dependencies
- `pip install -r requirements.txt`

### Run the notebooks
Open and run the notebooks in `notebooks/` to reproduce the CSV results and plots (saved under `results/`).

### Using the source code modules
Core functionality is implemented in `src/` and can be imported in notebooks/scripts, e.g.:
- `from src.data_loading import load_adult_income_dataset`
- `from src.models import train_adult_income_baselines`
- `from src.mitigation import train_fair_dp, train_threshold_optimizer`

### Reproducing results (where outputs are saved)
When notebooks are executed, outputs are saved automatically to:
- **Result tables (CSV):** `results/`
- **Plots (PNG):** `results/plots/`