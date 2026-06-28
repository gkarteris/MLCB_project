# Helper functions for the age-regression problem

from __future__ import annotations

import numpy as np
from scipy.stats import bootstrap

RAND_SEED = 42



# Optuna search spaces
# Used only when the class runs with tune=True

def elasticnet_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "max_iter": 10000,
        "random_state": RAND_SEED,
    }


def svr_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 5.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": "rbf",
    }


def rf_reg_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
        "random_state": RAND_SEED,
        "n_jobs": -1,
    }


def xgb_reg_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": RAND_SEED,
        "n_jobs": -1,
    }


# Bootstrap percentile CI for the median of a metric distribution
def median_ci(values, confidence=0.95, n_resamples=9_999, seed=RAND_SEED):
    values = np.asarray(getattr(values, "values", values), dtype=float)
    if np.allclose(values, values[0]):
        return float(values[0]), float(values[0])
    res = bootstrap(
        (values,),
        statistic=np.median,
        n_resamples=n_resamples,
        confidence_level=confidence,
        random_state=seed,
        method="percentile",
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)
