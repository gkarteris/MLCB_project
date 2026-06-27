# Data loading and feature-matrix construction for the age-regression problem

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Union

import anndata as ad
import numpy as np
import scipy.sparse as sp


AnnDataOrPath = Union[ad.AnnData, str, Path]


# loads an AnnData object or a path to a .h5ad file
def load_anndata(obj: AnnDataOrPath) -> ad.AnnData:
    if isinstance(obj, ad.AnnData):
        return obj
    return ad.read_h5ad(str(obj))


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


# Build a (X, y, feature_names) from an AnnData
# layer -> axpression matrix to be used as features.
#   None = .X (normalised + log1p)
#   .layers["raw_counts"] = raw UMI counts
#   .layers["normalised"] = normalised expression
# extra_obs -> numeric .obs columns to append as additional features
#   (e.g. .obs["nGene"], default is nUMI = total library size)
def build_xy(
    adata: AnnDataOrPath,
    target: str = "Age",
    layer: str | None = None,
    extra_obs: Sequence[str] = ("nUMI",),
) -> tuple[np.ndarray, np.ndarray, list[str]]:

    adata = load_anndata(adata)

    if target not in adata.obs:
        raise KeyError(f"target column {target!r} not found in .obs")

    source = adata.X if layer is None else adata.layers[layer]
    X = _dense(source).astype(np.float64)
    feature_names = list(adata.var_names)

    extra_cols: list[np.ndarray] = []
    for col in extra_obs:
        if col not in adata.obs:
            raise KeyError(f"extra_obs column {col!r} not found in .obs")
        extra_cols.append(adata.obs[col].to_numpy(dtype=np.float64).reshape(-1, 1))
        feature_names.append(col)

    if extra_cols:
        X = np.hstack([X] + extra_cols)

    y = adata.obs[target].to_numpy(dtype=np.float64)
    return X, y, feature_names

