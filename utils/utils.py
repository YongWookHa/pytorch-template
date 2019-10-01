import numpy as np
import pandas as pd
from datetime import datetime

def fetch_cosine_values(seq_len, frequency=0.01, noise=0.1):
    np.random.seed(101)
    x = np.arange(0.0, seq_len, 1.0)
    return np.cos(2 * np.pi * frequency * x) + np.random.uniform(low=noise, high=noise, size=seq_len)

def format_dataset(values, temporal_features):
    feat_splits = [values[i:i+temporal_features] for i in range(len(values) - temporal_features)]
    feats = np.array(feat_splits)
    labels = np.array(values[temporal_features:])
    return feats, labels

def matrix_to_array(m):
    return np.asarray(m).reshape(-1)
