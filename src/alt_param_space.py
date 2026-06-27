# Parameter spaces for the age-regression problem

from __future__ import annotations

from typing import Callable

from .functions import RAND_SEED


def elasticnet_space_alt(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 1.0),
        "max_iter": 10000,
        "random_state": RAND_SEED,
    }


def svr_space_alt(trial):
    return {
        "C": trial.suggest_float("C", 0.1, 500.0, log=True),
        "epsilon": trial.suggest_categorical("epsilon", [0.01, 0.1, 0.5, 1.0]),
        "kernel": "rbf",
    }


def rf_reg_space_alt(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
        "random_state": RAND_SEED,
        "n_jobs": -1,
    }


def xgb_reg_space_alt(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RAND_SEED,
        "n_jobs": -1,
    }


def alt_param_spaces() -> dict[str, Callable]:
    spaces = {
        "ElasticNet": elasticnet_space_alt,
        "SVR": svr_space_alt,
        "RandomForest": rf_reg_space_alt,
    }
    try:
        import xgboost
        spaces["XGBoost"] = xgb_reg_space_alt
    except ImportError:
        pass
    return spaces
