import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.sparse import issparse


#Converts the sparse matrix to a dense numpy array - X (features) and y (Age)
def build_features(adata_subset, add_numi=True):
   
    X = adata_subset.X.toarray() if issparse(adata_subset.X) else np.array(adata_subset.X)

    if add_numi:
        numi = adata_subset.obs["nUMI"].values.reshape(-1, 1)
        X = np.hstack([X, numi])   # nUMI as the last column

    y = adata_subset.obs["Age"].values.astype(float)
    return X, y
