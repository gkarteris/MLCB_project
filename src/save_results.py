# Save and reload CV results and fitted models

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone



# Environment capture (for reproducibility alongside the results)
def _versions() -> dict:
    import scipy
    import sklearn

    v = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for opt in ("xgboost", "optuna"):
        try:
            v[opt] = __import__(opt).__version__
        except Exception:
            pass
    return v


# Results (all metrics in CSV + JSON)
def save_result(result, outdir, name: str = "cv") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result.long.to_csv(outdir / f"{name}_long.csv", index=False)
    result.summary.to_csv(outdir / f"{name}_summary.csv", index=False)
    if getattr(result, "feature_stability", None) is not None \
            and not result.feature_stability.empty:
        result.feature_stability.to_csv(outdir / f"{name}_feature_stability.csv",
                                        index=False)

    # best_params is keyed by the tuple (model, round, fold); JSON has no tuple
    # keys, so flatten to a list of records (round/fold cast to plain int).
    records = [
        {"model": k[0], "round": int(k[1]), "fold": int(k[2]), "params": v}
        for k, v in result.best_params.items()
    ]
    (outdir / f"{name}_best_params.json").write_text(
        json.dumps(records, indent=2, default=str)
    )
    (outdir / f"{name}_meta.json").write_text(
        json.dumps(result.meta, indent=2, default=str)
    )
    (outdir / f"{name}_versions.json").write_text(json.dumps(_versions(), indent=2))
    return outdir


# Reload a CVResult saved by save_result function
def load_result(outdir, name: str = "cv", with_models: bool = False):

    from .nested_cv import CVResult

    outdir = Path(outdir)
    long = pd.read_csv(outdir / f"{name}_long.csv")
    summary = pd.read_csv(outdir / f"{name}_summary.csv")

    records = json.loads((outdir / f"{name}_best_params.json").read_text())
    best_params = {
        (r["model"], int(r["round"]), int(r["fold"])): r["params"] for r in records
    }
    meta = json.loads((outdir / f"{name}_meta.json").read_text())

    fs_path = outdir / f"{name}_feature_stability.csv"
    feature_stability = pd.read_csv(fs_path) if fs_path.exists() else pd.DataFrame()

    models = {}
    models_path = outdir / f"{name}_models.joblib"
    if with_models and models_path.exists():
        models = joblib.load(models_path)
    return CVResult(long=long, summary=summary, best_params=best_params,
                    meta=meta, models=models, feature_stability=feature_stability)


# Fitted models (joblib)
def save_models(result_or_models, path) -> Path:
    from .nested_cv import CVResult

    models = result_or_models.models if isinstance(result_or_models, CVResult) \
        else result_or_models
    if not models:
        raise ValueError("No fitted models to save.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)
    return path


# Reload a dict of fitted models saved by save_models
def load_models(path) -> dict:
    return joblib.load(path)


# joblib-dump a single fitted pipeline
def save_model(pipeline, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


# Reload a single fitted pipeline saved by save_model
def load_model(path):
    return joblib.load(path)


# Final model (depricated, use save_models/load_models instead)
#   reg -> configured NestedCVRegressor
#   params -> overrides the model's hyperparameters
def fit_final_model(reg, X, y, model_name: str, params: dict | None = None):
    params = dict(params) if params else {}
    k = params.pop("mrmr_k", None)
    est = clone(reg.estimators[model_name])
    if params:
        est.set_params(**params)
    selector = None
    if getattr(reg, "feature_selector", None) is not None:
        selector = clone(reg.feature_selector)
        cap = max(1, int(len(np.asarray(y)) // 5)) # 5:1 on all dev cells
        eff_k = min(getattr(selector, "k", 300) or 300, cap)
        selector.set_params(k=(k if k is not None else eff_k))
    pipe = reg._build_pipeline(est, selector=selector)
    pipe.fit(np.asarray(X, dtype=float), np.asarray(y))
    return pipe


# Hyperparameters of the best-scoring fold for model_name
def best_params_from_result(result, model_name: str, metric: str = "RMSE",
                            lower_is_better: bool = True) -> dict:
    sub = result.long[result.long["model"] == model_name]
    if sub.empty or not result.best_params:
        return {}
    idx = sub[metric].idxmin() if lower_is_better else sub[metric].idxmax()
    row = sub.loc[idx]
    key = (model_name, int(row["round"]), int(row["fold"]))
    return result.best_params.get(key, {})


# Save a whole run
def save_run(result, outdir, name: str = "cv", with_models: bool = False) -> Path:
    outdir = save_result(result, outdir, name=name)
    if with_models and getattr(result, "models", None):
        save_models(result, Path(outdir) / f"{name}_models.joblib")
    return Path(outdir)


# Per-family save
def save_per_family(per_family, outdir, name: str = "cv",
                    with_models: bool = False) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fam, res in per_family.results.items():
        save_run(res, outdir / fam, name=name, with_models=with_models)
    per_family.long.to_csv(outdir / "all_families_long.csv", index=False)
    per_family.summary.to_csv(outdir / "all_families_summary.csv", index=False)
    (outdir / "families.json").write_text(json.dumps(list(per_family.results), indent=2))
    return outdir


def load_per_family(outdir, name: str = "cv", with_models: bool = False):
    from .nested_cv import PerFamilyResult

    outdir = Path(outdir)
    families = json.loads((outdir / "families.json").read_text())
    results = {
        fam: load_result(outdir / fam, name=name, with_models=with_models)
        for fam in families
    }
    long = pd.read_csv(outdir / "all_families_long.csv")
    summary = pd.read_csv(outdir / "all_families_summary.csv")
    return PerFamilyResult(results=results, long=long, summary=summary)
