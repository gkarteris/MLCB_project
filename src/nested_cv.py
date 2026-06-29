# Repeated nested cross-validation for the age-regression problem
#   tune=False -> baseline (outer CV only, library-default hyperparameters)
#   tune=True -> nested CV (an Optuna inner loop tunes each model)
#   fs_mode="outer" vs "inner" -> feature selection is performed once every outer fold vs once per inner fold (very expensive computationally)
#   if k_grid is supplied the MRMRSelector use this for feature selection
#   if k_grid is None and k_step is supplied a size-aware K grid is generated for each family
#   if both k_grid and k_step are None, the MRMRSelector uses its default K value
#   keep_models=True retains each fitted fold pipeline in the result's .models dict

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import optuna
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    get_scorer,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .functions import (
    RAND_SEED,
    elasticnet_space,
    median_ci,
    rf_reg_space,
    svr_space,
    xgb_reg_space,
)

from .feature_selection import make_k_grid

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

optuna.logging.set_verbosity(optuna.logging.WARNING)

# estimator types that require feature scaling (linear models + SVR)
_SCALE_TYPES = (ElasticNet, Lasso, Ridge, SVR)

# used in older sklearn versions that don't have the RMSE metric
try:
    from sklearn.metrics import root_mean_squared_error as _rmse
except ImportError:
    from sklearn.metrics import mean_squared_error

    def _rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Default models and tuning spaces
def default_models(random_state: int = RAND_SEED) -> dict[str, Any]:
    models = {
        "Dummy": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(max_iter=10000, random_state=random_state),
        "SVR": SVR(), # rbf default
        # baseline RF 100 trees
        "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=random_state),
        # authors' fixed regressor
        "RandomForest_authors": RandomForestRegressor(n_estimators=8, max_depth=3, random_state=random_state),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(random_state=random_state, n_jobs=-1)
    return models


# Optuna spaces for the model tuning
def default_param_spaces() -> dict[str, Callable]:
    spaces = {
        "ElasticNet": elasticnet_space,
        "SVR": svr_space,
        "RandomForest": rf_reg_space,
    }
    if _HAS_XGB:
        spaces["XGBoost"] = xgb_reg_space
    return spaces


@dataclass
class CVResult:
    long: pd.DataFrame # one row per model, round and fold
    summary: pd.DataFrame # median + 95% CI per model and metric
    best_params: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict) # populated only if run(keep_models=True) with keys (model, round, fold)
    feature_stability: pd.DataFrame = field(default_factory=pd.DataFrame) # gene -> fraction of outer folds it was selected in (only populated if feature selection is used)


# Repeated (nested) CV age regressor
class NestedCVRegressor:
    def __init__(
        self,
        estimators: dict[str, Any],
        param_spaces: dict[str, Callable] | None = None, # only used when tune=True
        feature_selector: Any | None = None,
        n_rounds: int = 10,
        n_outer: int = 5,
        n_inner: int = 3,
        n_trials: int = 30,
        inner_scoring: str = "neg_root_mean_squared_error",
        tune_k: bool = False,
        k_grid: list[int] | None = None,
        k_step: int = 50,
        fs_mode: str = "outer", # "outer" or "inner"
        k_safety_cap: bool = True,
        random_state: int = RAND_SEED,
        verbose: int = 1,
    ):
        self.estimators = estimators
        self.param_spaces = param_spaces or {}
        self.feature_selector = feature_selector
        self.n_rounds = n_rounds
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.n_trials = n_trials
        self.inner_scoring = inner_scoring
        self.tune_k = tune_k
        self.k_grid = k_grid
        self.k_step = k_step
        self.fs_mode = fs_mode
        self.k_safety_cap = k_safety_cap
        self.random_state = random_state
        self.verbose = verbose


    # Pipelinebuilding and scaling
    @staticmethod
    def _needs_scaling(estimator) -> bool:
        return isinstance(estimator, _SCALE_TYPES)

    def _build_pipeline(self, estimator, selector=None) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        steps.append(("scaler", StandardScaler() if self._needs_scaling(estimator)
                      else "passthrough"))
        sel = selector if selector is not None else (
            clone(self.feature_selector)
            if (self.feature_selector is not None and self.fs_mode == "inner")
            else None
        )
        if sel is not None: # FS phase
            steps.append(("selector", sel))
        steps.append(("model", estimator))
        return Pipeline(steps)


    def _safe_k_cap(self, n_dev: int) -> int:
        inner_train = n_dev * (self.n_outer - 1) / self.n_outer \
                            * (self.n_inner - 1) / self.n_inner
        return max(1, int(inner_train // 5))


    # Metrics
    @staticmethod
    def _evaluate(y_true, y_pred) -> dict[str, float]:
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        ok = np.std(y_pred) > 0
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(_rmse(y_true, y_pred)),
            "MedAE": float(median_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "Pearson": float(pearsonr(y_true, y_pred)[0]) if ok else np.nan,
            "Spearman": float(spearmanr(y_true, y_pred)[0]) if ok else np.nan,
        }
    

    # Inner loop (Optuna) - only when tuning
    def _inner_optimize(self, X_tr, y_tr, name, estimator, inner_seed, k_grid=None) -> dict:
        space = self.param_spaces[name]
        inner_cv = StratifiedKFold(n_splits=self.n_inner, shuffle=True,
                                   random_state=inner_seed)
        
        fs = self.feature_selector if self.fs_mode == "inner" else None
        tune_k = self.tune_k and (fs is not None) and bool(k_grid)
        k_max = max(k_grid) if tune_k else None
        scorer = get_scorer(self.inner_scoring)

        folds = []
        for it, iv in inner_cv.split(X_tr, y_tr):
            Xit, Xiv, yit, yiv = X_tr[it], X_tr[iv], y_tr[it], y_tr[iv]
            if fs is None:
                folds.append((Xit, Xiv, yit, yiv, None))
            elif tune_k:
                sel = clone(fs).set_params(k=k_max).fit(Xit, yit)
                folds.append((Xit, Xiv, yit, yiv, sel.ranking_)) # slice later
            else:
                sel = clone(fs).fit(Xit, yit)
                cols = sel.get_support()
                folds.append((Xit[:, cols], Xiv[:, cols], yit, yiv, None))

        def _pipe(est):
            steps = [("scaler", StandardScaler() if self._needs_scaling(est)
                      else "passthrough"), ("model", est)]
            return Pipeline(steps)

        def objective(trial):
            params = space(trial)
            cand = clone(estimator).set_params(**params) # native params
            k = trial.suggest_categorical("mrmr_k", list(k_grid)) if tune_k else None
            scores = []
            for Xit, Xiv, yit, yiv, ranking in folds:
                if ranking is not None: # tuned K: slice the ranking
                    cols = ranking[:k]
                    Xi, Xv = Xit[:, cols], Xiv[:, cols]
                else: # fixed K (pre-reduced) or no FS
                    Xi, Xv = Xit, Xiv
                pipe = _pipe(clone(cand))
                pipe.fit(Xi, yit)
                scores.append(scorer(pipe, Xv, yiv))
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=inner_seed),
        )
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params


    # CV entry point (runs the whole repeated CV, nested if tune=True)
    #   keep_models=True retains every fitted fold pipeline in the result's .models dict
    def run(self, X, y, tune: bool = False, keep_models: bool = False,
            label: str | None = None, verbose: int = 1,
            feature_names: list | None = None) -> CVResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        outer = RepeatedStratifiedKFold(
            n_splits=self.n_outer, n_repeats=self.n_rounds,
            random_state=self.random_state,
        )

        rows: list[dict] = []
        best_params: dict = {}
        fold_models: dict = {}
        feat_masks: list = [] # per-fold selected-gene masks (inner mode)
        fold_cols: dict = {} # gene mask computed once per outer fold and reused across models

        outer_fs = self.feature_selector is not None and self.fs_mode == "outer"
        capped_k = None
        if outer_fs:
            base_k = getattr(self.feature_selector, "k", None) or 100
            # choose wether to apply a size-aware cap on the mRMR K value
            capped_k = min(base_k, X.shape[1])
            if self.k_safety_cap:
                capped_k = min(capped_k, self._safe_k_cap(len(y)))
            if verbose and capped_k != base_k:
                print(f"[{label or 'run'}] mRMR K: {base_k} -> {capped_k} (size-aware cap)")

        # create k-tuning grid or use the user-supplied one (only when selecting features AND tuning)
        active_k_grid = None
        if tune and self.feature_selector is not None and self.fs_mode == "inner" and self.tune_k:
            active_k_grid = self.k_grid or make_k_grid(
                len(y), self.n_outer, self.n_inner, self.k_step,
                n_features=X.shape[1],
            )


        for name, estimator in self.estimators.items():
            do_tune = tune and (name in self.param_spaces)
            if verbose:
                print(f"[{label or 'run'}] {name}: "
                      f"{'tuning' if do_tune else 'baseline'} ...", flush=True)

            for i, (tr, te) in enumerate(outer.split(X, y)):
                rnd, fold = divmod(i, self.n_outer)
                X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

                # select genes once per outer fold, reuse across models
                if outer_fs:
                    if i not in fold_cols:
                        s = clone(self.feature_selector).set_params(k=capped_k)
                        s.fit(X_tr, y_tr)
                        fold_cols[i] = s.get_support()
                    cols = fold_cols[i]
                    X_tr, X_te = X_tr[:, cols], X_te[:, cols]

                if verbose >= 2:
                    print(f"    {name} round {rnd} fold {fold}", flush=True)

                est = clone(estimator)
                sel_override = None
                if do_tune:
                    inner_seed = self.random_state + i
                    bp = self._inner_optimize(X_tr, y_tr, name, est, inner_seed,
                                              k_grid=active_k_grid)
                    k = bp.pop("mrmr_k", None) # present only in inner mode
                    est.set_params(**bp)
                    best_params[(name, rnd, fold)] = dict(
                        bp, **({"mrmr_k": k} if k is not None else {}))
                    if k is not None and self.feature_selector is not None:
                        sel_override = clone(self.feature_selector).set_params(k=k)

                pipe = self._build_pipeline(est, selector=sel_override)
                pipe.fit(X_tr, y_tr)
                metrics = self._evaluate(y_te, pipe.predict(X_te))
                rows.append({"model": name, "round": rnd, "fold": fold, **metrics})
                if keep_models:
                    fold_models[(name, rnd, fold)] = pipe
                if "selector" in pipe.named_steps:
                    feat_masks.append(pipe.named_steps["selector"].get_support())

        long = pd.DataFrame(rows)
        summary = self.summarize(long)

        masks = list(fold_cols.values()) if outer_fs else feat_masks
        if masks:
            freq = np.vstack(masks).mean(axis=0)
            names = feature_names if (feature_names is not None
                                      and len(feature_names) == len(freq)) else list(range(len(freq)))
            feature_stability = (pd.DataFrame({"gene": names, "frequency": freq})
                                 .sort_values("frequency", ascending=False)
                                 .reset_index(drop=True))
        else:
            feature_stability = pd.DataFrame()

        meta = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_rounds": self.n_rounds,
            "n_outer": self.n_outer,
            "n_inner": self.n_inner,
            "tuned": bool(tune),
            "feature_selection": self.feature_selector is not None,
            "inner_scoring": self.inner_scoring,
            "k_grid": active_k_grid,
            "fs_mode": self.fs_mode if self.feature_selector is not None else None,
            "mrmr_k": capped_k,
        }
        return CVResult(long=long, summary=summary, best_params=best_params,
                        meta=meta, models=fold_models, feature_stability=feature_stability)

    
    # Bootstrap CI of the median over the fold distribution
    @staticmethod
    def summarize(long: pd.DataFrame) -> pd.DataFrame:
        metric_cols = [c for c in long.columns if c not in ("model", "round", "fold")]
        out = []
        for name, g in long.groupby("model", sort=False):
            row = {"model": name}
            for m in metric_cols:
                vals = g[m].dropna().values
                med = float(np.median(vals)) if len(vals) else np.nan
                lo, hi = median_ci(vals) if len(vals) > 1 else (med, med)
                row[f"{m}_median"] = med
                row[f"{m}_lo"] = lo
                row[f"{m}_hi"] = hi
            out.append(row)
        return pd.DataFrame(out)


    # Compact median (lo-hi) table for the report
    @staticmethod
    def format_table(summary: pd.DataFrame,
                     metrics=("MAE", "RMSE", "R2", "Spearman")) -> pd.DataFrame:
        disp = {"model": summary["model"]}
        for m in metrics:
            disp[m] = [
                f"{r[f'{m}_median']:.3f} ({r[f'{m}_lo']:.3f}-{r[f'{m}_hi']:.3f})"
                for _, r in summary.iterrows()
            ]
        return pd.DataFrame(disp)


@dataclass
class PerFamilyResult:
    results: dict
    long: pd.DataFrame
    summary: pd.DataFrame

    def save(self, outdir, name: str = "cv", with_models: bool = False):
        from .save_results import save_per_family
        return save_per_family(self, outdir, name=name, with_models=with_models)


# Choose the trainig data (only used inside run_per_family)
#   dev="train" -> only the training set
#   dev="train_test" -> both training and test sets
def _dev_sources(value, dev: str):
    items = value if isinstance(value, (list, tuple)) else (value,)
    if dev == "train":
        return items[:1]
    if dev in ("train_test", "all"):
        return items
    raise ValueError("dev must be 'train' or 'train_test'")


# Run the per-family CV
#   keep_models=True retains each family's fitted fold pipelines
#   save_dir - checkpoints the whole sweep (one subfolder of CSV/JSON per family)
def run_per_family(
    families: dict[str, "tuple"],
    target: str = "Age",
    add_numi: bool = True,
    layer: str | None = None,
    tune: bool = False,
    dev: str = "train_test",
    keep_models: bool = False,
    save_dir=None,
    save_name: str = "cv",
    models: dict | None = None,
    param_spaces: dict | None = None,
    feature_selector=None,
    **cv_kwargs,
) -> PerFamilyResult:

    from .data import build_xy, load_anndata

    verbose = cv_kwargs.pop("verbose", 1)
    results: dict = {}
    long_parts, summary_parts = [], []
    for fam, src in families.items():
        sources = _dev_sources(src, dev)
        Xs, ys, feat = [], [], None
        for s in sources:
            X, y, feat = build_xy(load_anndata(s), target=target, layer=layer,
                                  extra_obs=("nUMI",) if add_numi else ())
            Xs.append(X); ys.append(y)
        X = np.vstack(Xs); y = np.concatenate(ys)

        if verbose:
            print(f"\n=== Family: {fam}  ({len(y)} cells, {X.shape[1]} features) ===",
                  flush=True)
        reg = NestedCVRegressor(
            models if models is not None else default_models(),
            param_spaces if param_spaces is not None else default_param_spaces(),
            feature_selector=feature_selector,
            **cv_kwargs,
        )
        res = reg.run(X, y, tune=tune, keep_models=keep_models, label=fam,
                      verbose=verbose, feature_names=feat)
        results[fam] = res

        lt = res.long.copy(); lt.insert(0, "family", fam); long_parts.append(lt)
        st = res.summary.copy(); st.insert(0, "family", fam); summary_parts.append(st)


    pfr = PerFamilyResult(
        results=results,
        long=pd.concat(long_parts, ignore_index=True),
        summary=pd.concat(summary_parts, ignore_index=True),
    )
    if save_dir is not None:
        pfr.save(save_dir, name=save_name, with_models=keep_models)
    return pfr
