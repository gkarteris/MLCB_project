# Evaluation functions
#

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import build_xy, load_anndata
from .nested_cv import NestedCVRegressor
from .save_results import fit_final_model


# Evaluation results
@dataclass
class TestEvaluation:
    metrics: dict # MAE / RMSE / MedAE / R2 / Pearson / Spearman
    y_true: np.ndarray
    y_pred: np.ndarray
    model: object # the pipeline fit on the training side
    meta: dict = field(default_factory=dict)


