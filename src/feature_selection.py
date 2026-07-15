# mRMR feature selection for regression

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin, f_regression, mutual_info_regression

# mRMR selector class
#   k = number of features to keep
#   relevance = "f" (F-statistic) or "mi" (mutual information)
class MRMRSelector(BaseEstimator, SelectorMixin):
    def __init__(self, k: int = 100, relevance: str = "f", random_state: int = 42):
        self.k = k
        self.relevance = relevance
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        k = max(1, min(int(self.k), p))

        # relevance of each gene to age
        if self.relevance == "mi":
            rel = mutual_info_regression(X, y, random_state=self.random_state)
        else:
            F, _ = f_regression(X, y)
            rel = np.nan_to_num(F, nan=0.0)

        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

        # relevance / mean |corr| to selected
        first = int(np.argmax(rel))
        selected = [first]
        remaining = np.ones(p, dtype=bool)
        remaining[first] = False
        red_sum = np.abs(Xs.T @ Xs[:, first]) / n # running |corr| to picked set

        while len(selected) < k and remaining.any():
            idx = np.where(remaining)[0]
            mean_red = red_sum[idx] / len(selected)
            score = rel[idx] / (mean_red + 1e-12)
            best = idx[int(np.argmax(score))]
            selected.append(best)
            remaining[best] = False
            red_sum += np.abs(Xs.T @ Xs[:, best]) / n

        self.ranking_ = np.asarray(selected) # selection order
        self.n_features_in_ = p
        mask = np.zeros(p, dtype=bool)
        mask[self.ranking_] = True
        self.support_ = mask
        return self

    def _get_support_mask(self):
        return self.support_


# Coarse grid of K values, size-aware and capped at a default 5:1 ratio in the inner training fold
def make_k_grid(
    n_dev: int,
    n_outer: int = 5,
    n_inner: int = 3,
    step: int = 50,
    ratio: int = 5,
    n_features: int = 2000,
) -> list[int]:
    inner_train = n_dev * (n_outer - 1) / n_outer * (n_inner - 1) / n_inner
    cap = int(min(inner_train // ratio, n_features))
    grid = list(range(step, cap + 1, step))
    if cap >= step and (not grid or grid[-1] != cap):
        grid.append(cap) # the safe ceiling
    if not grid: # for very small datasets, ensure at least one value
        grid = [max(1, cap)]
    return sorted(set(grid))
