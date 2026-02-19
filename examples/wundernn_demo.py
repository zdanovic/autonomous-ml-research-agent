"""Demo script for Wundernn competition using Comet-Swarm."""

import asyncio
from pathlib import Path

import polars as pl

from comet_swarm.agents import EvaluatorAgent, StrategistAgent
from comet_swarm.integrations import ExperimentTracker, get_tracer
from comet_swarm.platforms import WundernnAdapter
from comet_swarm.tools import (
    correlation_analysis,
    cross_validate,
    lag_features,
    profile_dataset,
    rolling_stats,
)


async def main():
    """Run Comet-Swarm on Wundernn Predictorium competition."""
    
    print("=" * 60)
    print("Comet-Swarm: Autonomous ML Competition Agent")
    print("Competition: Wundernn Predictorium (LOB Prediction)")
    print("=" * 60)
    
    # Initialize tracing
    tracer = get_tracer()
    tracer.start_trace("wundernn_demo")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        project_name="comet-swarm",
        competition_name="wundernn_predictorium",
    )
    
    # Initialize platform adapter
    adapter = WundernnAdapter()
    context = adapter.get_competition_context()
    
    print(f"\n📊 Competition: {context.name}")
    print(f"📈 Metric: {context.metric.value} ({context.metric_direction})")
    print(f"📤 Submission: {context.submission_format}")
    
    # Download data (if not exists)
    data_dir = Path("./data/wundernn")
    if not data_dir.exists():
        print("\n⬇️ Downloading competition data...")
        # In real usage: await adapter.download_data(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample data (or create synthetic for demo)
    print("\n📂 Loading dataset...")
    
    # Create synthetic data for demo
    import numpy as np
    np.random.seed(42)
    
    n_samples = 10000
    n_sequences = 100
    
    df = pl.DataFrame({
        "seq_ix": np.repeat(range(n_sequences), n_samples // n_sequences),
        "step_in_seq": np.tile(range(n_samples // n_sequences), n_sequences),
        "need_prediction": np.random.choice([True, False], n_samples, p=[0.8, 0.2]),
        **{f"p{i}": np.random.randn(n_samples) for i in range(12)},
        "t0": np.random.randn(n_samples),
        "t1": np.random.randn(n_samples),
    })
    
    # Save synthetic data
    train_path = data_dir / "train.parquet"
    df.write_parquet(train_path)
    print(f"✅ Created synthetic dataset: {len(df)} rows")
    
    # Phase 1: EDA
    print("\n" + "=" * 60)
    print("Phase 1: Exploratory Data Analysis")
    print("=" * 60)
    
    dataset_info = profile_dataset(
        df, 
        name="wundernn_train",
        target_column="t0",
    )
    
    print(f"  Rows: {dataset_info.num_rows:,}")
    print(f"  Columns: {dataset_info.num_columns}")
    print(f"  Memory: {dataset_info.memory_mb:.2f} MB")
    
    # Target correlations
    correlations = correlation_analysis(df, target_column="t0", top_n=5)
    print("\n  Top correlations with t0:")
    for feat, corr in correlations.items():
        print(f"    {feat}: {corr:.4f}")
    
    # Phase 2: Strategy
    print("\n" + "=" * 60)
    print("Phase 2: Strategy Generation")
    print("=" * 60)
    
    strategist = StrategistAgent()
    
    dataset_summary = f"LOB data with {len(df)} samples, 12 price features (p0-p11), dual targets (t0, t1)"
    eda_summary = "Top correlated features: " + ", ".join(
        [f"{k}={v:.3f}" for k, v in list(correlations.items())[:3]]
    )
    
    strategy = await strategist.generate_strategy(
        competition_context=context.model_dump(),
        dataset_info=dataset_summary,
        eda_insights=eda_summary,
    )
    
    print(f"\n  Priority: {strategy.priority}")
    print("\n  Hypotheses:")
    for i, h in enumerate(strategy.hypotheses[:3], 1):
        print(f"    {i}. {h[:80]}...")
    
    print("\n  Model Recommendations:")
    for rec in strategy.model_recommendations[:2]:
        print(f"    • {rec[:60]}...")
    
    # Phase 3: Feature Engineering
    print("\n" + "=" * 60)
    print("Phase 3: Feature Engineering")
    print("=" * 60)
    
    # Apply lag features
    df_features = lag_features(
        df,
        column="p0",
        lags=[1, 5, 10],
        group_by="seq_ix",
        sort_by="step_in_seq",
    )
    
    # Apply rolling stats
    df_features = rolling_stats(
        df_features,
        column="p0",
        windows=[5, 10],
        stats=["mean", "std"],
        group_by="seq_ix",
    )
    
    new_features = [c for c in df_features.columns if c not in df.columns]
    print(f"  Created {len(new_features)} new features:")
    for feat in new_features:
        print(f"    • {feat}")
    
    # Phase 4: Model Training
    print("\n" + "=" * 60)
    print("Phase 4: Model Training (LightGBM)")
    print("=" * 60)
    
    # Prepare data
    feature_cols = [f"p{i}" for i in range(12)] + new_features
    target_col = "t0"
    
    # Filter valid rows
    df_train = df_features.filter(
        pl.col("need_prediction") & 
        pl.all_horizontal(pl.col(feature_cols).is_not_null())
    )
    
    X = df_train.select(feature_cols).to_numpy()
    y = df_train.select(target_col).to_numpy().flatten()
    
    print(f"  Training samples: {len(X):,}")
    print(f"  Features: {len(feature_cols)}")
    
    # Start experiment
    tracker.start_experiment(
        name="lgbm_baseline_v1",
        tags=["baseline", "lgbm", "wundernn"],
    )
    
    # Cross-validate
    cv_result = cross_validate(
        X, y,
        model_fn="lightgbm",
        params={"num_leaves": 31, "learning_rate": 0.05},
        n_folds=5,
        log_to_comet=True,
    )
    
    print("\n  CV Results:")
    print(f"    Mean RMSE: {cv_result['cv_mean']:.4f}")
    print(f"    Std: {cv_result['cv_std']:.4f}")
    print(f"    Fold scores: {[f'{s:.4f}' for s in cv_result['fold_scores']]}")
    
    # Log to Comet
    tracker.log_metrics({
        "cv_mean": cv_result["cv_mean"],
        "cv_std": cv_result["cv_std"],
    })
    
    if cv_result["feature_importance"]:
        tracker.log_feature_importance(cv_result["feature_importance"])
    
    # End experiment
    experiment_result = tracker.end_experiment()
    print(f"\n  Experiment logged: {experiment_result.experiment_url if experiment_result else 'N/A'}")
    
    # Phase 5: Evaluation
    print("\n" + "=" * 60)
    print("Phase 5: Evaluation Analysis")
    print("=" * 60)
    
    evaluator = EvaluatorAgent()
    analysis = await evaluator.analyze_results(
        experiment_results=[{
            "name": "lgbm_baseline",
            "model_type": "lightgbm",
            "cv_mean": cv_result["cv_mean"],
            "cv_std": cv_result["cv_std"],
        }],
        metric="rmse",
        metric_direction="minimize",
    )
    
    print(f"  Summary: {analysis.get('performance_summary', 'N/A')}")
    print(f"  Overfitting: {analysis.get('overfitting_status', 'N/A')}")
    print(f"  Should Submit: {analysis.get('should_submit', False)}")
    
    # Budget Summary
    print("\n" + "=" * 60)
    print("Budget Summary")
    print("=" * 60)
    
    budget = tracer.get_budget_status()
    print(f"  Total Spent: ${budget.total_spent_usd:.4f}")
    print(f"  Budget Remaining: ${budget.remaining_usd:.2f}")
    print(f"  Percentage Used: {budget.percentage_used:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    
    return cv_result


if __name__ == "__main__":
    asyncio.run(main())
