"""Comet-Swarm tools package."""

from comet_swarm.tools.eda import (
    correlation_analysis,
    distribution_analysis,
    profile_dataset,
)
from comet_swarm.tools.features import (
    datetime_features,
    frequency_encode,
    lag_features,
    rolling_stats,
    target_encode,
)
from comet_swarm.tools.metrics import (
    mean_r2,
    pearson_correlation,
    rmsle,
)
from comet_swarm.tools.training import (
    cross_validate,
    train_catboost,
    train_lightgbm,
    train_xgboost,
)

__all__ = [
    # EDA
    "profile_dataset",
    "correlation_analysis",
    "distribution_analysis",
    # Features
    "target_encode",
    "frequency_encode",
    "lag_features",
    "rolling_stats",
    "datetime_features",
    # Metrics
    "pearson_correlation",
    "rmsle",
    "mean_r2",
    # Training
    "train_lightgbm",
    "train_xgboost",
    "train_catboost",
    "cross_validate",
]
