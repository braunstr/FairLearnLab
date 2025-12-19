#  FairLearnLab — Developing a Framework for Measuring and Optimizing Algorithmic Fairness in Machine Learning Model

This project contains the code and experiments for my ***bachelor thesis** on **measuring and improving algorithmic fairness** in supervised **tabular classification** tasks.  
Using the **UCI Adult Census Income** dataset and the **Statlog (German Credit Data)** dataset, the project trains baseline models, evaluates performance and fairness metrics across protected groups, and applies mitigation techniques using **Fairlearn** (e.g., Demographic Parity constraints and threshold optimization).

The outputs of this project are reproducible **CSV tables** and **PNG plots** (, reproducible through notebooks, saved under `results/`), which are used directly for the thesis evaluation and discussion.

## Scope / What’s included

This project contains a full, reproducible experimentation pipeline for **fairness evaluation and mitigation** on two tabular classification datasets:

### Datasets
- **UCI Adult Census Income**
  - Target: `income_binary` (<=50K vs >50K)
  - Protected attributes evaluated: `sex` (main), optionally `race`
- **Statlog (German Credit Data)**
  - Target: `credit_risk_binary` (good vs bad)
  - Protected attribute evaluated: `personal_status_sex`

### What the code does
- **Data preprocessing & splits**
  - Converts the original UCI `.data` / `.test` files into processed CSV files.
  - Creates fixed **train / validation / test** splits.
  - Stores processed data under: `data/processed/`.

- **Baseline model training**
  - Trains multiple baseline classifiers with a shared preprocessing pipeline:
    - DummyClassifier 
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
  - Saves baseline performance & fairness result tables to `results/`.

- **Fairness evaluation**
  - Computes *performance metrics* and *fairness metrics* using `fairlearn.metrics.MetricFrame`:
    - Accuracy, Precision, Recall, F1
    - Statistical Parity Difference (via selection rate gap)
    - Disparate Impact Ratio (via selection rate ratio)
    - Equal Opportunity Difference (TPR gap)
    - Predictive Parity Difference (precision gap)
    - Calibration within groups (Brier score, overall + group gap)

- **Mitigation techniques (Fairlearn)**
  - **In-processing:** `ExponentiatedGradient + DemographicParity` (Logistic Regression base estimator)
    - Includes an `eps` sweep to visualize the fairness/performance trade-off
  - **Post-processing:** `ThresholdOptimizer`
    - Supports `equalized_odds` and `demographic_parity`

### Outputs (used for the thesis)
- Result tables are saved as **CSV** in `results/`
- Figures are saved as **PNG** in `results/plots`
- Notebooks generate the final plots and summary tables used in the thesis discussion.


## Project structure
FairLearnLab/

    data/ - included for reproducability 
        processed/ - Generated CSV splits used to train models
            adult_test.csv
            adult_train.csv
            adult_val.csv
            german_test.csv
            german_train.csv
            german_val.csv
        raw/ - Original downloaded UCI files (not modified)
            adult.data
            adult.names
            adult.test
            german.data

    notebooks/ - Jupyter notebooks (experiments + plots + exports)
        01_environment_and_data_sanity.ipynb
        02_baselines_train_eval.ipynb
        03_calibration_within_groups.ipynb
        04_mitigation_exponentiated_gradient_dp.ipynb
        05_mitigation_threshold_optimizer.ipynb
        06_final_results_and_plots.ipynb

    results/
        ... all csv files that were generated during the experiments
        plots/
            ... all plots that were generated during the experiments


    scripts/ - used to convert data files to .csv files and split the data in train/test/validation sets
        data_preprocessing.py

    src/ - Reusable Python code (importable modules)
        __init__.py
        data_loading.py
        fairness.py
        mitigation.py
        models.py
        preprocessing.py

    README.md
    requirements.txt - environment dependencies (pip freeze)
    .gitignore

### Notes
- `src/` contains **clean, reusable code** (FairlearnLab framework)
- `notebooks/` contains **all experiments** and produces:
  - CSV result tables -> `results/`
  - PNG plots -> `results/plots/`
- `data/raw/` stores the **original dataset files**
- `data/processed/` stores the **final splits used in training/evaluation**
- `results/` stores the **csv files and plots generated during the experiments**

## Install and run FairlearnLab

IMPORTANT: In order to run the code in this project, python version 3.12 is required, since Fairlearn is not supported by any higher python versions yet

### Create & activate a virtual environment

**Windows (PowerShell)**

py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1

**macOS/ Linux**

python3.12 -m venv .venv

source .venv/bin/activate

### Install dependencies
pip install -r requirements.txt


### Run the notebooks
Open and run the notebooks in notebooks/ to verify that the project is working. (results csvs and pots will be saved to results/)

### Using the source code modules
Core functionality is implemented in src/ and can be imported in notebooks/scripts, e.g.:

from src.data_loading import load_adult_income_dataset
from src.models import train_adult_income_baselines
from src.fairness import evaluate_adult_income_fairness
from src.mitigation import train_adult_income_logreg_fair_dp


### Reproducing results (where outputs are saved)

When you run the notebooks, outputs are saved automatically:

- **Result tables (CSV):** `results/`
  - e.g. baseline fairness summaries, mitigation sweeps, threshold comparisons, calibration tables, Pareto/frontier tables

- **Plots (PNG):** `results/plots/`
  - e.g. general data distribution, accuracy vs fairness trade-offs, bar charts for fairness metrics, calibration disparity plots, Pareto visualizations

If you want to start clean, delete the `results/` folder contents and re-run the notebooks from top to bottom.