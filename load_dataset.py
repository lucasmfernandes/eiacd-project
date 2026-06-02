"""
Module: Load California Housing Dataset
Importable functions for loading the dataset.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_california_housing() -> tuple[pd.DataFrame, dict]:
    """
    Load the California Housing dataset.
    
    Returns:
        Tuple of (DataFrame with features and target, dataset info dict)
    """
    california = fetch_california_housing(as_frame=True)
    df = california.frame.copy()
    df = df.rename(columns={'MedHouseVal': 'target'})
    
    info = {
        'feature_names': list(california.feature_names),
        'description': california.DESCR,
        'n_samples': len(df),
        'n_features': len(california.feature_names)
    }
    
    return df, info
