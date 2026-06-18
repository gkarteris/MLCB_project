import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.sparse import issparse


#Remove numerically-labeled clusters
def is_named(label):
    return not re.fullmatch(r'\d+', str(label).strip())

def passes_filter(row):
    #timepoints where the cell type is present (>0)
    present = row[row > 0]
    #fail if any present bin has only 1 cell
    return (present >= 2).all()

