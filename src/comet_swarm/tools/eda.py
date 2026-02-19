"""EDA tools using Polars for fast data analysis."""

from typing import Any

import polars as pl

from comet_swarm.integrations.opik import traced
from comet_swarm.schemas import ColumnInfo, DatasetInfo


@traced(name="profile_dataset")
def profile_dataset(
    df: pl.DataFrame,
    name: str = "dataset",
    target_column: str | None = None,
    sample_size: int = 5,
) -> DatasetInfo:
    """
    Generate comprehensive dataset profile.
    
    Args:
        df: Polars DataFrame to profile
        name: Dataset name
        target_column: Optional target column for analysis
        sample_size: Number of sample values per column
    
    Returns:
        DatasetInfo with schema and statistics
    """
    columns: list[ColumnInfo] = []
    
    for col_name in df.columns:
        col = df[col_name]
        dtype = str(col.dtype)
        null_count = col.null_count()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        unique_count = col.n_unique()
        
        # Sample values
        samples = col.drop_nulls().head(sample_size).to_list()
        
        col_info = ColumnInfo(
            name=col_name,
            dtype=dtype,
            null_count=null_count,
            null_percentage=round(null_pct, 2),
            unique_count=unique_count,
            sample_values=samples,
        )
        
        # Numeric stats
        if col.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            col_info.mean = float(col.mean()) if col.mean() is not None else None
            col_info.std = float(col.std()) if col.std() is not None else None
            col_info.min = float(col.min()) if col.min() is not None else None
            col_info.max = float(col.max()) if col.max() is not None else None
        
        # Categorical stats
        if col.dtype in (pl.Utf8, pl.Categorical) or unique_count < 20:
            value_counts = col.value_counts().sort("count", descending=True).head(10)
            col_info.top_categories = dict(
                zip(
                    value_counts[col_name].to_list(),
                    value_counts["count"].to_list(),
                )
            )
        
        columns.append(col_info)
    
    # Target distribution
    target_dist = None
    if target_column and target_column in df.columns:
        target_col = df[target_column]
        if target_col.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
            target_dist = {
                "mean": float(target_col.mean()) if target_col.mean() else None,
                "std": float(target_col.std()) if target_col.std() else None,
                "min": float(target_col.min()) if target_col.min() else None,
                "max": float(target_col.max()) if target_col.max() else None,
                "median": float(target_col.median()) if target_col.median() else None,
            }
    
    return DatasetInfo(
        name=name,
        num_rows=len(df),
        num_columns=len(df.columns),
        memory_mb=round(df.estimated_size("mb"), 2),
        columns=columns,
        target_column=target_column,
        target_distribution=target_dist,
    )


@traced(name="correlation_analysis")
def correlation_analysis(
    df: pl.DataFrame,
    target_column: str,
    top_n: int = 20,
    method: str = "pearson",
) -> dict[str, float]:
    """
    Calculate correlations between features and target.
    
    Args:
        df: Polars DataFrame
        target_column: Target column name
        top_n: Number of top correlations to return
        method: Correlation method (pearson)
    
    Returns:
        Dictionary of feature -> correlation value, sorted by absolute value
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    correlations: dict[str, float] = {}
    target = df[target_column]
    
    # Only numeric columns
    numeric_cols = [
        col for col in df.columns
        if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        and col != target_column
    ]
    
    for col_name in numeric_cols:
        col = df[col_name]
        
        # Skip columns with all nulls or single value
        if col.null_count() == len(df) or col.n_unique() <= 1:
            continue
        
        # Calculate Pearson correlation
        try:
            corr = target.pearson_corr(col)
            if corr is not None:
                correlations[col_name] = round(float(corr), 4)
        except Exception:
            continue
    
    # Sort by absolute value
    sorted_corrs = dict(
        sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]
    )
    
    return sorted_corrs


@traced(name="distribution_analysis")
def distribution_analysis(
    df: pl.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Analyze distributions of numeric columns.
    
    Args:
        df: Polars DataFrame
        columns: Specific columns to analyze (default: all numeric)
    
    Returns:
        Dictionary of column -> distribution stats
    """
    if columns is None:
        columns = [
            col for col in df.columns
            if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        ]
    
    results: dict[str, dict[str, Any]] = {}
    
    for col_name in columns:
        if col_name not in df.columns:
            continue
        
        col = df[col_name]
        
        # Skip non-numeric
        if col.dtype not in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            continue
        
        # Calculate statistics
        stats: dict[str, Any] = {
            "mean": float(col.mean()) if col.mean() else None,
            "median": float(col.median()) if col.median() else None,
            "std": float(col.std()) if col.std() else None,
            "min": float(col.min()) if col.min() else None,
            "max": float(col.max()) if col.max() else None,
        }
        
        # Skewness detection (simple heuristic)
        if stats["mean"] and stats["median"]:
            skew_ratio = stats["mean"] / stats["median"] if stats["median"] != 0 else 1
            if skew_ratio > 1.5:
                stats["skew"] = "right"
                stats["suggestion"] = "Consider log1p transform"
            elif skew_ratio < 0.67:
                stats["skew"] = "left"
                stats["suggestion"] = "Consider square transform"
            else:
                stats["skew"] = "symmetric"
        
        # Outlier detection (IQR method)
        try:
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_count = (
                    (col < lower_bound) | (col > upper_bound)
                ).sum()
                
                stats["outlier_count"] = int(outlier_count)
                stats["outlier_percentage"] = round(
                    (outlier_count / len(df)) * 100, 2
                )
        except Exception:
            pass
        
        results[col_name] = stats
    
    return results


@traced(name="temporal_patterns")
def temporal_patterns(
    df: pl.DataFrame,
    datetime_column: str,
    target_column: str,
    aggregations: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Analyze temporal patterns in time-series data.
    
    Args:
        df: Polars DataFrame with datetime column
        datetime_column: Name of datetime column
        target_column: Target variable to analyze
        aggregations: Time aggregation levels (default: hour, dow, month)
    
    Returns:
        Dictionary of temporal aggregation -> stats
    """
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found")
    
    if aggregations is None:
        aggregations = ["hour", "day_of_week", "month"]
    
    results: dict[str, dict[str, float]] = {}
    
    # Ensure datetime type
    dt_col = df[datetime_column]
    if dt_col.dtype != pl.Datetime:
        df = df.with_columns(pl.col(datetime_column).str.to_datetime())
    
    for agg in aggregations:
        try:
            if agg == "hour":
                grouped = df.group_by(
                    pl.col(datetime_column).dt.hour().alias("period")
                )
            elif agg == "day_of_week":
                grouped = df.group_by(
                    pl.col(datetime_column).dt.weekday().alias("period")
                )
            elif agg == "month":
                grouped = df.group_by(
                    pl.col(datetime_column).dt.month().alias("period")
                )
            else:
                continue
            
            agg_df = grouped.agg(
                pl.col(target_column).mean().alias("mean")
            ).sort("period")
            
            results[agg] = dict(
                zip(
                    agg_df["period"].to_list(),
                    agg_df["mean"].to_list(),
                )
            )
        except Exception:
            continue
    
    return results
