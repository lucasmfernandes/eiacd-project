"""
Module: Prediction Pipeline
Importable functions for data preparation, model training, and predictions.

Tools:
- Tool 1: make_predictions() - predictions for table data
- Tool 3: process_text_input() - predictions from plain text
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def prepare_data(
    df: pd.DataFrame,
    target_column: str = 'target',
    is_training: bool = True
) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    """
    Prepare data for the regression model.
    
    Returns:
        Tuple of (feature matrix, target values or None, feature column names)
    """
    df = df.copy()
    
    if target_column in df.columns:
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column] if is_training else None
    else:
        feature_columns = df.columns.tolist()
        X = df[feature_columns]
        y = None
    
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    
    return X, y, feature_columns


def create_pipeline() -> Pipeline:
    """Create regression pipeline with StandardScaler and GradientBoosting."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10
        ))
    ])


def train_model(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train the regression model.
    
    Returns:
        Tuple of (trained pipeline, X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_train, X_test, y_train, y_test


def make_predictions(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Tool 1: Make predictions for new data stored in a table.
    """
    return pipeline.predict(X)


def process_text_input(text: str, feature_names: list[str]) -> pd.DataFrame:
    """
    Tool 3: Process new data described in plain text.
    
    Example input: "MedInc=5.5, HouseAge=30, AveRooms=6, ..."
    """
    defaults = {
        'MedInc': 3.87, 'HouseAge': 29.0, 'AveRooms': 5.43, 'AveBedrms': 1.10,
        'Population': 1425.0, 'AveOccup': 3.07, 'Latitude': 35.63, 'Longitude': -119.57
    }
    
    parsed = defaults.copy()
    text_clean = text.replace('\n', ',').replace(';', ',')
    
    for part in text_clean.split(','):
        part = part.strip()
        for sep in ('=', ':'):
            if sep in part:
                key, _, val = part.partition(sep)
                key = key.strip()
                try:
                    if key in parsed:
                        parsed[key] = float(val.strip())
                except ValueError:
                    pass
                break
    
    return pd.DataFrame([parsed])[feature_names]
