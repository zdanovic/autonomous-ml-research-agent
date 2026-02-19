"""Feature engineering tools using Polars."""

from typing import Literal

import numpy as np
import polars as pl

from comet_swarm.integrations.opik import traced


@traced(name="target_encode")
def target_encode(
    df: pl.DataFrame,
    column: str,
    target: str,
    n_folds: int = 5,
    smoothing: float = 10.0,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Apply target encoding with k-fold cross-validation to prevent leakage.
    
    Args:
        df: Input DataFrame
        column: Categorical column to encode
        target: Target column name
        n_folds: Number of CV folds
        smoothing: Smoothing parameter (higher = more regularization)
        seed: Random seed for fold assignment
    
    Returns:
        DataFrame with new encoded column
    """
    new_col_name = f"{column}_target_enc"
    
    # Add fold column
    np.random.seed(seed)
    fold_assignments = np.random.randint(0, n_folds, len(df))
    df = df.with_columns(pl.Series("_fold", fold_assignments))
    
    # Global mean for smoothing
    global_mean = df[target].mean()
    
    # Initialize result column
    encoded_values = np.zeros(len(df))
    
    for fold in range(n_folds):
        # Train on other folds
        train_mask = df["_fold"] != fold
        val_mask = df["_fold"] == fold
        
        train_df = df.filter(train_mask)
        
        # Calculate target statistics per category
        stats = train_df.group_by(column).agg([
            pl.col(target).mean().alias("mean"),
            pl.col(target).count().alias("count"),
        ])
        
        # Apply smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
        stats = stats.with_columns(
            (
                (pl.col("count") * pl.col("mean") + smoothing * global_mean)
                / (pl.col("count") + smoothing)
            ).alias("smoothed_mean")
        )
        
        # Create mapping
        encoding_map = dict(
            zip(stats[column].to_list(), stats["smoothed_mean"].to_list())
        )
        
        # Apply to validation fold
        val_indices = np.where(val_mask.to_numpy())[0]
        val_values = df.filter(val_mask)[column].to_list()
        
        for idx, val in zip(val_indices, val_values):
            encoded_values[idx] = encoding_map.get(val, global_mean)
    
    # Drop fold column and add encoded column
    df = df.drop("_fold")
    df = df.with_columns(pl.Series(new_col_name, encoded_values))
    
    return df


@traced(name="frequency_encode")
def frequency_encode(
    df: pl.DataFrame,
    column: str,
    normalize: bool = True,
) -> pl.DataFrame:
    """
    Apply frequency encoding to categorical column.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        normalize: Whether to normalize counts to [0, 1]
    
    Returns:
        DataFrame with new encoded column
    """
    new_col_name = f"{column}_freq_enc"
    
    # Calculate value counts
    value_counts = df.group_by(column).agg(
        pl.count().alias("_count")
    )
    
    if normalize:
        total = len(df)
        value_counts = value_counts.with_columns(
            (pl.col("_count") / total).alias("_count")
        )
    
    # Join back
    df = df.join(value_counts, on=column, how="left")
    df = df.rename({"_count": new_col_name})
    
    return df


@traced(name="lag_features")
def lag_features(
    df: pl.DataFrame,
    column: str,
    lags: list[int],
    group_by: str | list[str] | None = None,
    sort_by: str | None = None,
) -> pl.DataFrame:
    """
    Create lag features for time-series data.
    
    Args:
        df: Input DataFrame
        column: Column to create lags for
        lags: List of lag values (e.g., [1, 7, 30])
        group_by: Optional column(s) to group by (e.g., user_id)
        sort_by: Column to sort by before computing lags
    
    Returns:
        DataFrame with lag columns
    """
    if sort_by:
        df = df.sort(sort_by)
    
    for lag in lags:
        new_col_name = f"{column}_lag_{lag}"
        
        if group_by:
            df = df.with_columns(
                pl.col(column).shift(lag).over(group_by).alias(new_col_name)
            )
        else:
            df = df.with_columns(
                pl.col(column).shift(lag).alias(new_col_name)
            )
    
    return df


@traced(name="rolling_stats")
def rolling_stats(
    df: pl.DataFrame,
    column: str,
    windows: list[int],
    stats: list[Literal["mean", "std", "min", "max", "sum"]] | None = None,
    group_by: str | list[str] | None = None,
    min_periods: int = 1,
) -> pl.DataFrame:
    """
    Create rolling window statistics.
    
    Args:
        df: Input DataFrame
        column: Column to compute rolling stats for
        windows: List of window sizes
        stats: Statistics to compute (default: mean, std)
        group_by: Optional column(s) to group by
        min_periods: Minimum observations in window
    
    Returns:
        DataFrame with rolling stat columns
    """
    if stats is None:
        stats = ["mean", "std"]
    
    for window in windows:
        for stat in stats:
            new_col_name = f"{column}_roll_{stat}_{window}"
            
            # Build rolling expression
            if stat == "mean":
                expr = pl.col(column).rolling_mean(window, min_periods=min_periods)
            elif stat == "std":
                expr = pl.col(column).rolling_std(window, min_periods=min_periods)
            elif stat == "min":
                expr = pl.col(column).rolling_min(window, min_periods=min_periods)
            elif stat == "max":
                expr = pl.col(column).rolling_max(window, min_periods=min_periods)
            elif stat == "sum":
                expr = pl.col(column).rolling_sum(window, min_periods=min_periods)
            else:
                continue
            
            if group_by:
                df = df.with_columns(expr.over(group_by).alias(new_col_name))
            else:
                df = df.with_columns(expr.alias(new_col_name))
    
    return df


@traced(name="datetime_features")
def datetime_features(
    df: pl.DataFrame,
    column: str,
    features: list[str] | None = None,
) -> pl.DataFrame:
    """
    Extract datetime components as features.
    
    Args:
        df: Input DataFrame
        column: Datetime column
        features: List of components to extract (default: all common)
    
    Returns:
        DataFrame with datetime feature columns
    """
    if features is None:
        features = ["year", "month", "day", "hour", "day_of_week", "day_of_year", "week"]
    
    # Ensure datetime type
    if df[column].dtype != pl.Datetime:
        df = df.with_columns(pl.col(column).str.to_datetime())
    
    dt = pl.col(column)
    
    for feature in features:
        new_col_name = f"{column}_{feature}"
        
        match feature:
            case "year":
                df = df.with_columns(dt.dt.year().alias(new_col_name))
            case "month":
                df = df.with_columns(dt.dt.month().alias(new_col_name))
            case "day":
                df = df.with_columns(dt.dt.day().alias(new_col_name))
            case "hour":
                df = df.with_columns(dt.dt.hour().alias(new_col_name))
            case "minute":
                df = df.with_columns(dt.dt.minute().alias(new_col_name))
            case "day_of_week":
                df = df.with_columns(dt.dt.weekday().alias(new_col_name))
            case "day_of_year":
                df = df.with_columns(dt.dt.ordinal_day().alias(new_col_name))
            case "week":
                df = df.with_columns(dt.dt.week().alias(new_col_name))
            case "is_weekend":
                df = df.with_columns(
                    (dt.dt.weekday() >= 5).cast(pl.Int8).alias(new_col_name)
                )
    
    return df


@traced(name="bin_numeric")
def bin_numeric(
    df: pl.DataFrame,
    column: str,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """
    Bin numeric column into categories.
    
    Args:
        df: Input DataFrame
        column: Numeric column to bin
        n_bins: Number of bins
        strategy: Binning strategy (quantile or uniform)
        labels: Optional labels for bins
    
    Returns:
        DataFrame with binned column
    """
    new_col_name = f"{column}_binned"
    
    col = df[column]
    
    if strategy == "quantile":
        # Calculate quantile boundaries
        quantiles = [i / n_bins for i in range(1, n_bins)]
        boundaries = [float(col.quantile(q)) for q in quantiles]
    else:
        # Uniform boundaries
        min_val = float(col.min())
        max_val = float(col.max())
        step = (max_val - min_val) / n_bins
        boundaries = [min_val + step * i for i in range(1, n_bins)]
    
    # Create bin labels
    if labels is None:
        labels = [f"bin_{i}" for i in range(n_bins)]
    
    # Apply binning using cut
    df = df.with_columns(
        pl.col(column)
        .cut(boundaries, labels=labels)
        .alias(new_col_name)
    )
    
    return df


@traced(name="interaction_features")
def interaction_features(
    df: pl.DataFrame,
    columns: list[str],
    method: Literal["multiply", "divide", "add", "subtract"] = "multiply",
) -> pl.DataFrame:
    """
    Create pairwise interaction features.
    
    Args:
        df: Input DataFrame
        columns: Columns to create interactions for
        method: Interaction method
    
    Returns:
        DataFrame with interaction columns
    """
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            match method:
                case "multiply":
                    new_col_name = f"{col1}_x_{col2}"
                    df = df.with_columns(
                        (pl.col(col1) * pl.col(col2)).alias(new_col_name)
                    )
                case "divide":
                    new_col_name = f"{col1}_div_{col2}"
                    df = df.with_columns(
                        (pl.col(col1) / (pl.col(col2) + 1e-8)).alias(new_col_name)
                    )
                case "add":
                    new_col_name = f"{col1}_plus_{col2}"
                    df = df.with_columns(
                        (pl.col(col1) + pl.col(col2)).alias(new_col_name)
                    )
                case "subtract":
                    new_col_name = f"{col1}_minus_{col2}"
                    df = df.with_columns(
                        (pl.col(col1) - pl.col(col2)).alias(new_col_name)
                    )
    
    return df
