# Machine Learning in Computational Biology course project

# Cell-Type Transcriptomic Age Clocks in the *Drosophila* Brain

## Overview

This project builds a transparent, automated machine learning pipeline that predicts the chronological age of *Drosophila melanogaster* brain cells from single-cell RNA-seq data (Davie et al. 2018, GEO [GSE107451](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107451)).

Using **repeated nested cross-validation**, **mRMR feature selection**, and **XGBoost**, we predict age across six functional brain cell families (Kenyon Cells, Optic Lobe neurons, Monoaminergic neurons, Glia, Peptidergic neurons, Clock neurons), then use **SHAP** and **Jaccard similarity** to compare which genes drive aging predictions in each family, and test how well aging signatures generalize across sexes.


## Using `nested_cv.py`

This is the core of the pipeline: a repeated **nested cross-validation** (rnCV) regressor with built-in feature selection and Optuna hyperparameter tuning, designed to avoid data leakage / optimism bias when working with high-dimensional, low-sample-size transcriptomic data.

### 1. The rnCV architecture

- **Outer loop** (`n_outer`-fold, repeated `n_rounds` times via `RepeatedStratifiedKFold`): estimates true generalization performance on completely held-out folds.
- **Inner loop** (`n_inner`-fold, only used when `tune=True`): an Optuna study tunes each model's hyperparameters (and optionally the mRMR `k`) using only inner-training data. The winning configuration is then refit on the full outer-training fold and scored on the untouched outer-test fold.

### 2. Two ways to run it

**a) `NestedCVRegressor`** - run the CV on a single `(X, y)` matrix (i.e. for one cell family).

**b) `run_per_family(...)`** - convenience wrapper that loops `NestedCVRegressor` over several cell families, builds `X, y` from AnnData sources per family, concatenates results, and can checkpoint everything to disk. This is the entry point you'll normally use.

> **Note on `run_per_family(...)`:** if `dev="tain"` it trains only on training dataset else it trains on the whole dataset (including the held-out test set).

### 3. Key parameters (`NestedCVRegressor.__init__`)

| Parameter | Meaning |
|---|---|
| `estimators` | `dict[str, sklearn-like estimator]`, e.g. from `default_models()` |
| `param_spaces` | `dict[str, Callable(trial) -> params]`, e.g. from `default_param_spaces()`. Only used when `tune=True`. |
| `feature_selector` | an mRMR selector instance (`MRMRSelector` from `feature_selection.py`), or `None` to skip feature selection |
| `n_rounds`, `n_outer`, `n_inner` | repeats × outer folds × inner folds |
| `n_trials` | number of Optuna trials per inner tuning call |
| `inner_scoring` | sklearn scorer name Optuna optimizes for, e.g. `"neg_root_mean_squared_error"` |
| `fs_mode` | `"outer"` - select features once per outer fold (fast, slight leakage risk from inner tuning reusing the same features) vs. `"inner"` - reselect features inside every inner fold (slow, leakage-safe). |
| `tune_k` | if `True` **and** `fs_mode="inner"`, the number of selected features `k` is itself tuned by Optuna. Very expensive - **not used in this assignment**. |
| `k_grid` | explicit list of `k` values to try for `MRMRSelector`. Takes priority over `k_step`. |
| `k_step` | if `k_grid` is `None`, a size-aware grid of `k` values is auto-generated using this step. |
| `k_safety_cap` | if `True`, caps mRMR `k` at a 5:1 training-samples-to-features ratio to avoid overfitting on small families. |
| `keep_models` | if `True`, every fitted fold pipeline is stored in `CVResult.models` - disk/memory heavy, debugging only. |

> **Note on `k_grid` / `active_k_grid`:** if you pass a fixed `k_grid`, make sure `tune_k=False` (the default) unless you explicitly want Optuna to search over it - a previously-fixed bug allowed `active_k_grid` auto-generation to silently override a user-supplied fixed `k`. This is now opt-in only.

### 4. Basic usage - single family, baseline (no tuning)

```python
reg = NestedCVRegressor(
    estimators=default_models(), # Dummy, LinearRegression, ElasticNet, SVR, RandomForest(s), XGBoost
    n_rounds=10,
    n_outer=5,
)
result = reg.run(X, y, tune=False, verbose=1)

print(result.summary) # median + 95% CI per model/metric
print(NestedCVRegressor.format_table(result.summary)) # compact display table
```

### 5. Full nested CV with mRMR + Optuna tuning - single family

```python
reg = NestedCVRegressor(
    estimators=default_models(),
    param_spaces=default_param_spaces(),
    feature_selector=MRMRSelector(k=100, relevance="f"),
    fs_mode="inner",
    inner_scoring="neg_root_mean_squared_error",
    n_rounds=10, n_outer=5, n_inner=3, n_trials=30,
    tune_k=False,
)
result = reg.run(X, y, tune=True, keep_models=False, verbose=1, feature_names=gene_names)

result.summary # per-model performance table
result.best_params
result.feature_stability
```

### 6. Running all six cell families at once

```python
families = {
    "Kenyon_Cells":  (train_source, test_source),
    "Optic_Lobe":    (train_source, test_source),
    "Monoaminergic": (train_source, test_source),
    "Glia":          (train_source, test_source),
    "Peptidergic":   (train_source, test_source),
    "Clock":         (train_source, test_source),
}

pfr = run_per_family(
    families,
    models=default_models(),
    param_spaces=alt_param_spaces(),
    feature_selector=MRMRSelector(k=100, relevance="f"),
    tune=True,
    dev="train",
    n_rounds=5, n_outer=5, n_inner=3,
    n_trials=20,
    inner_scoring="neg_root_mean_squared_error",
    keep_models=False,
    save_dir="/results",
    save_name="CV",
    verbose=1,
)

pfr.summary # concatenated per-family summary table
pfr.long # concatenated per-family long-format results (per model/round/fold)
```