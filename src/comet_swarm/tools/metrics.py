"""Competition-specific evaluation metrics."""

import numpy as np
import polars as pl
from scipy import stats

from comet_swarm.integrations.opik import traced


@traced(name="pearson_correlation")
def pearson_correlation(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
) -> float:
    """
    Calculate Pearson correlation coefficient.
    Used by Wundernn Predictorium.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Pearson correlation coefficient
    """
    if isinstance(y_true, pl.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pl.Series):
        y_pred = y_pred.to_numpy()
    
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return 0.0
    
    corr, _ = stats.pearsonr(y_true, y_pred)
    return float(corr)


@traced(name="weighted_pearson")
def weighted_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Calculate weighted Pearson correlation for multi-target predictions.
    Wundernn uses average of t0 and t1 correlations.
    
    Args:
        y_true: Shape (n_samples, n_targets)
        y_pred: Shape (n_samples, n_targets)
        weights: Optional weights per target (default: equal)
    
    Returns:
        Weighted average correlation
    """
    n_targets = y_true.shape[1] if y_true.ndim > 1 else 1
    
    if n_targets == 1:
        return pearson_correlation(y_true.flatten(), y_pred.flatten())
    
    if weights is None:
        weights = np.ones(n_targets) / n_targets
    
    correlations = []
    for i in range(n_targets):
        corr = pearson_correlation(y_true[:, i], y_pred[:, i])
        correlations.append(corr)
    
    return float(np.average(correlations, weights=weights))


@traced(name="rmsle")
def rmsle(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error.
    Used by Solafune Construction Cost.
    
    Args:
        y_true: Ground truth values (must be >= 0)
        y_pred: Predicted values (must be >= 0)
    
    Returns:
        RMSLE score (lower is better)
    """
    if isinstance(y_true, pl.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pl.Series):
        y_pred = y_pred.to_numpy()
    
    # Ensure non-negative
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)
    
    # Handle NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return 0.0
    
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return float(np.sqrt(np.mean(log_diff ** 2)))


@traced(name="mean_r2")
def mean_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: int | None = None,
) -> float:
    """
    Calculate mean R² (coefficient of determination).
    Used by Kaggle Urban Flood.
    
    Args:
        y_true: Ground truth, shape (n_samples,) or (n_samples, n_features)
        y_pred: Predictions, same shape as y_true
        axis: Axis to compute R² along (for multi-dimensional)
    
    Returns:
        Mean R² score
    """
    if y_true.ndim == 1:
        # Simple case
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return float(1 - ss_res / ss_tot)
    
    # Multi-dimensional case (e.g., multiple nodes in flood prediction)
    r2_scores = []
    
    if axis is None:
        # Compute R² for each column and average
        for i in range(y_true.shape[1]):
            col_true = y_true[:, i]
            col_pred = y_pred[:, i]
            
            # Skip if constant
            if np.std(col_true) == 0:
                continue
            
            ss_res = np.sum((col_true - col_pred) ** 2)
            ss_tot = np.sum((col_true - np.mean(col_true)) ** 2)
            
            if ss_tot > 0:
                r2_scores.append(1 - ss_res / ss_tot)
    
    return float(np.mean(r2_scores)) if r2_scores else 0.0


@traced(name="rmse")
def rmse(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        RMSE score (lower is better)
    """
    if isinstance(y_true, pl.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pl.Series):
        y_pred = y_pred.to_numpy()
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return 0.0
    
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@traced(name="mae")
def mae(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        MAE score (lower is better)
    """
    if isinstance(y_true, pl.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pl.Series):
        y_pred = y_pred.to_numpy()
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return 0.0
    
    return float(np.mean(np.abs(y_true - y_pred)))


# Metric registry for platform-specific selection
METRIC_REGISTRY = {
    "pearson": pearson_correlation,
    "weighted_pearson": weighted_pearson,
    "rmsle": rmsle,
    "r2": mean_r2,
    "rmse": rmse,
    "mae": mae,
}


def get_metric(name: str):
    """Get metric function by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]
