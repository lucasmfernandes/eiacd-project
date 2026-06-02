"""
Module: Model Confidence and Evaluation
Importable functions for model evaluation metrics.

Tool 2: get_model_confidence_summary() - confidence of predictions
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = ""
) -> dict[str, float | int | str]:
    """Calculate comprehensive regression metrics."""
    residuals = np.asarray(y_true) - y_pred
    
    return {
        'dataset': dataset_name,
        'n_samples': len(y_true),
        'r2_score': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'max_error': float(np.max(np.abs(residuals))),
    }


def calculate_prediction_intervals(
    pipeline: Pipeline,
    X: pd.DataFrame,
    confidence_level: float = 0.95
) -> dict[str, np.ndarray | float]:
    """
    Estimate prediction intervals using staged predictions.
    
    Returns:
        Dict with predictions, lower_bound, upper_bound, uncertainty
    """
    regressor = pipeline.named_steps['regressor']
    scaler = pipeline.named_steps['scaler']
    X_scaled = scaler.transform(X)
    
    staged_preds = list(regressor.staged_predict(X_scaled))
    n_stages = min(20, len(staged_preds))
    recent_staged_preds = np.array(staged_preds[-n_stages:])
    
    predictions = regressor.predict(X_scaled)
    uncertainty = np.sqrt(np.var(recent_staged_preds, axis=0))
    
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    return {
        'predictions': predictions,
        'lower_bound': predictions - z_score * uncertainty,
        'upper_bound': predictions + z_score * uncertainty,
        'uncertainty': uncertainty,
        'confidence_level': confidence_level
    }


def get_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importance from the model."""
    importance = pipeline.named_steps['regressor'].feature_importances_
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'importance_pct': importance * 100
    }).sort_values('importance', ascending=False)


def get_model_confidence_summary(
    train_metrics: dict,
    test_metrics: dict,
    intervals_test: dict
) -> dict[str, float | str]:
    """
    Tool 2: Generate comprehensive confidence summary for the model.
    
    Returns:
        Dict with overall_confidence_score, confidence_level, and detailed metrics
    """
    r2_component = max(0, test_metrics['r2_score']) * 40
    mape_component = max(0, (100 - test_metrics['mape'])) * 0.3
    
    generalization_ratio = test_metrics['r2_score'] / max(train_metrics['r2_score'], 0.01)
    generalization_component = min(generalization_ratio, 1.0) * 30
    
    confidence_score = r2_component + mape_component + generalization_component
    
    if confidence_score >= 70:
        confidence_level = 'High'
    elif confidence_score >= 50:
        confidence_level = 'Medium'
    else:
        confidence_level = 'Low'
    
    avg_interval_width = float(np.mean(
        intervals_test['upper_bound'] - intervals_test['lower_bound']
    ))
    
    return {
        'overall_confidence_score': round(confidence_score, 2),
        'confidence_level': confidence_level,
        'r2_score_test': round(test_metrics['r2_score'], 4),
        'r2_score_train': round(train_metrics['r2_score'], 4),
        'generalization_ratio': round(generalization_ratio, 4),
        'mape_test': round(test_metrics['mape'], 2),
        'rmse_test': round(test_metrics['rmse'], 4),
        'mae_test': round(test_metrics['mae'], 4),
        'avg_prediction_interval_width': round(avg_interval_width, 4),
    }
