"""Model training tools with Comet ML integration."""

from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from comet_swarm.integrations.comet_ml import get_tracker
from comet_swarm.integrations.opik import traced


@traced(name="train_lightgbm")
def train_lightgbm(
    X_train: np.ndarray | pl.DataFrame,
    y_train: np.ndarray | pl.Series,
    X_val: np.ndarray | pl.DataFrame | None = None,
    y_val: np.ndarray | pl.Series | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    log_to_comet: bool = True,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Train LightGBM model with optional early stopping.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        params: LightGBM parameters
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Early stopping patience
        log_to_comet: Whether to log to Comet ML
        feature_names: Optional feature names for importance
    
    Returns:
        Dictionary with model, metrics, and feature importance
    """
    import lightgbm as lgb
    
    # Convert Polars to numpy if needed
    if isinstance(X_train, pl.DataFrame):
        feature_names = feature_names or X_train.columns
        X_train = X_train.to_numpy()
    if isinstance(y_train, pl.Series):
        y_train = y_train.to_numpy()
    if isinstance(X_val, pl.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_val, pl.Series):
        y_val = y_val.to_numpy()
    
    # Default parameters
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    
    if params:
        default_params.update(params)
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_data]
    valid_names = ["train"]
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append("valid")
    
    # Training callbacks for history
    evals_result: dict[str, dict[str, list[float]]] = {}
    
    # Train
    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )
    
    # Feature importance
    importance_dict = {}
    if feature_names:
        importance = model.feature_importance(importance_type="gain")
        importance_dict = dict(zip(feature_names, importance.tolist()))
    
    # Prepare result
    result = {
        "model": model,
        "model_type": "lightgbm",
        "best_iteration": model.best_iteration,
        "training_history": evals_result,
        "feature_importance": importance_dict,
        "params": default_params,
    }
    
    # Extract final scores
    if "valid" in evals_result:
        metric_name = list(evals_result["valid"].keys())[0]
        result["val_score"] = evals_result["valid"][metric_name][-1]
    
    if "train" in evals_result:
        metric_name = list(evals_result["train"].keys())[0]
        result["train_score"] = evals_result["train"][metric_name][-1]
    
    # Log to Comet
    if log_to_comet:
        tracker = get_tracker()
        tracker.log_parameters(default_params)
        if "val_score" in result:
            tracker.log_metric("val_score", result["val_score"])
        if importance_dict:
            tracker.log_feature_importance(importance_dict)
    
    return result


@traced(name="train_xgboost")
def train_xgboost(
    X_train: np.ndarray | pl.DataFrame,
    y_train: np.ndarray | pl.Series,
    X_val: np.ndarray | pl.DataFrame | None = None,
    y_val: np.ndarray | pl.Series | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    log_to_comet: bool = True,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Train XGBoost model with optional early stopping.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        params: XGBoost parameters
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Early stopping patience
        log_to_comet: Whether to log to Comet ML
        feature_names: Optional feature names
    
    Returns:
        Dictionary with model, metrics, and feature importance
    """
    import xgboost as xgb
    
    # Convert Polars to numpy
    if isinstance(X_train, pl.DataFrame):
        feature_names = feature_names or X_train.columns
        X_train = X_train.to_numpy()
    if isinstance(y_train, pl.Series):
        y_train = y_train.to_numpy()
    if isinstance(X_val, pl.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_val, pl.Series):
        y_val = y_val.to_numpy()
    
    # Default parameters
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": 42,
    }
    
    if params:
        default_params.update(params)
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    evals = [(dtrain, "train")]
    
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        evals.append((dval, "valid"))
    
    # Train
    evals_result: dict[str, dict[str, list[float]]] = {}
    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )
    
    # Feature importance
    importance_dict = model.get_score(importance_type="gain")
    
    result = {
        "model": model,
        "model_type": "xgboost",
        "best_iteration": model.best_iteration,
        "training_history": evals_result,
        "feature_importance": importance_dict,
        "params": default_params,
    }
    
    if "valid" in evals_result:
        metric_name = list(evals_result["valid"].keys())[0]
        result["val_score"] = evals_result["valid"][metric_name][-1]
    
    if log_to_comet:
        tracker = get_tracker()
        tracker.log_parameters(default_params)
        if "val_score" in result:
            tracker.log_metric("val_score", result["val_score"])
        if importance_dict:
            tracker.log_feature_importance(importance_dict)
    
    return result


@traced(name="train_catboost")
def train_catboost(
    X_train: np.ndarray | pl.DataFrame,
    y_train: np.ndarray | pl.Series,
    X_val: np.ndarray | pl.DataFrame | None = None,
    y_val: np.ndarray | pl.Series | None = None,
    params: dict[str, Any] | None = None,
    iterations: int = 1000,
    early_stopping_rounds: int = 50,
    cat_features: list[int | str] | None = None,
    log_to_comet: bool = True,
) -> dict[str, Any]:
    """
    Train CatBoost model with native categorical support.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        params: CatBoost parameters
        iterations: Maximum iterations
        early_stopping_rounds: Early stopping patience
        cat_features: Categorical feature indices or names
        log_to_comet: Whether to log to Comet ML
    
    Returns:
        Dictionary with model, metrics, and feature importance
    """
    from catboost import CatBoostRegressor, Pool
    
    # Convert Polars
    feature_names = None
    if isinstance(X_train, pl.DataFrame):
        feature_names = X_train.columns
        X_train = X_train.to_numpy()
    if isinstance(y_train, pl.Series):
        y_train = y_train.to_numpy()
    if isinstance(X_val, pl.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_val, pl.Series):
        y_val = y_val.to_numpy()
    
    # Default parameters
    default_params = {
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    
    if params:
        default_params.update(params)
    
    # Create model
    model = CatBoostRegressor(
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        **default_params,
    )
    
    # Create pools
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    eval_pool = None
    if X_val is not None and y_val is not None:
        eval_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # Train
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    
    # Feature importance
    importance = model.get_feature_importance()
    importance_dict = {}
    if feature_names:
        importance_dict = dict(zip(feature_names, importance.tolist()))
    
    result = {
        "model": model,
        "model_type": "catboost",
        "best_iteration": model.best_iteration_,
        "feature_importance": importance_dict,
        "params": default_params,
    }
    
    if eval_pool:
        result["val_score"] = model.best_score_["validation"]["RMSE"]
    
    if log_to_comet:
        tracker = get_tracker()
        tracker.log_parameters(default_params)
        if "val_score" in result:
            tracker.log_metric("val_score", result["val_score"])
        if importance_dict:
            tracker.log_feature_importance(importance_dict)
    
    return result


@traced(name="cross_validate")
def cross_validate(
    X: np.ndarray | pl.DataFrame,
    y: np.ndarray | pl.Series,
    model_fn: Literal["lightgbm", "xgboost", "catboost"] = "lightgbm",
    params: dict[str, Any] | None = None,
    n_folds: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    log_to_comet: bool = True,
) -> dict[str, Any]:
    """
    Perform k-fold cross-validation.
    
    Args:
        X: Features
        y: Labels
        model_fn: Model type to use
        params: Model parameters
        n_folds: Number of CV folds
        shuffle: Whether to shuffle before splitting
        seed: Random seed
        log_to_comet: Whether to log to Comet ML
    
    Returns:
        Dictionary with OOF predictions, fold scores, and mean/std
    """
    # Convert Polars
    feature_names = None
    if isinstance(X, pl.DataFrame):
        feature_names = X.columns
        X = X.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()
    
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    
    oof_preds = np.zeros(len(X))
    fold_scores: list[float] = []
    models: list[Any] = []
    feature_importances: list[dict[str, float]] = []
    
    # Select training function
    train_fns = {
        "lightgbm": train_lightgbm,
        "xgboost": train_xgboost,
        "catboost": train_catboost,
    }
    train_fn = train_fns[model_fn]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train (don't log individual folds)
        result = train_fn(
            X_train, y_train,
            X_val, y_val,
            params=params,
            log_to_comet=False,
            feature_names=feature_names,
        )
        
        model = result["model"]
        models.append(model)
        
        # Get predictions
        if model_fn == "xgboost":
            import xgboost as xgb
            dval = xgb.DMatrix(X_val, feature_names=feature_names)
            preds = model.predict(dval)
        elif model_fn == "catboost":
            preds = model.predict(X_val)
        else:
            preds = model.predict(X_val)
        
        oof_preds[val_idx] = preds
        
        if "val_score" in result:
            fold_scores.append(result["val_score"])
        
        if result["feature_importance"]:
            feature_importances.append(result["feature_importance"])
    
    # Average feature importance
    avg_importance: dict[str, float] = {}
    if feature_importances:
        all_features = set()
        for fi in feature_importances:
            all_features.update(fi.keys())
        
        for feat in all_features:
            values = [fi.get(feat, 0) for fi in feature_importances]
            avg_importance[feat] = float(np.mean(values))
    
    result = {
        "oof_predictions": oof_preds,
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)) if fold_scores else None,
        "cv_std": float(np.std(fold_scores)) if fold_scores else None,
        "models": models,
        "feature_importance": avg_importance,
        "model_type": model_fn,
    }
    
    # Log to Comet
    if log_to_comet:
        tracker = get_tracker()
        if result["cv_mean"]:
            tracker.log_metric("cv_mean", result["cv_mean"])
        if result["cv_std"]:
            tracker.log_metric("cv_std", result["cv_std"])
        for i, score in enumerate(fold_scores):
            tracker.log_metric(f"fold_{i}_score", score)
        if avg_importance:
            tracker.log_feature_importance(avg_importance)
    
    return result
