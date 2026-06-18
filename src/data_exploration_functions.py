import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split


#Function for stratified split
def stratified_split(obs, test_size, random_state):
    #Each family group will be splited seperately
    #Stratification split based on the age imbalance and the initial cell type 
    #so that every age bin and every initial cellular type will be proportionally represented in both train and test sets
    
    strat = obs["age_map"].astype(str) + "__" + obs["annotation"].astype(str)
    
    train_idx, test_idx = train_test_split(obs.index, test_size=test_size, stratify=strat, random_state=random_state)

    return train_idx, test_idx
