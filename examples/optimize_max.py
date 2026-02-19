#!/usr/bin/env python
"""
Autonomous Optimization Runner - Iterates until maximum performance.

This script runs multiple optimization strategies and keeps improving
until no more gains are possible.
"""

# Add src to path
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



@dataclass
class OptimizationResult:
    strategy: str
    cv_score: float
    cv_std: float
    features: int
    params: dict
    improvement: float = 0.0


class AutonomousOptimizer:
    """Autonomous optimization loop that keeps improving until convergence."""
    
    def __init__(self, train_path: str, target_col: str, budget_usd: float = 1.0):
        self.train_path = Path(train_path)
        self.target_col = target_col
        self.budget_usd = budget_usd
        self.results: list[OptimizationResult] = []
        self.best_score = float('inf')  # RMSE (lower is better)
        self.no_improvement_count = 0
        self.max_no_improvement = 3  # Stop after 3 iterations without improvement
        
    def load_data(self) -> pl.DataFrame:
        """Load training data."""
        if self.train_path.suffix == '.parquet':
            return pl.read_parquet(self.train_path)
        return pl.read_csv(self.train_path)
    
    def encode_categoricals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Encode categorical columns."""
        for col in df.columns:
            if df[col].dtype == pl.Utf8 and col != 'data_id':
                # Label encode
                unique_vals = df[col].unique().to_list()
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df = df.with_columns(
                    pl.col(col).replace(mapping).alias(f"{col}_encoded")
                )
        return df
    
    def add_features_v1(self, df: pl.DataFrame) -> pl.DataFrame:
        """Basic feature engineering."""
        # Numeric interactions
        numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric() and c != self.target_col]
        
        if 'year' in df.columns and 'deflated_gdp_usd' in df.columns:
            df = df.with_columns([
                (pl.col('deflated_gdp_usd') / 1e9).alias('gdp_billions'),
            ])
        
        return df
    
    def add_features_v2(self, df: pl.DataFrame) -> pl.DataFrame:
        """Advanced feature engineering with country aggregates."""
        df = self.add_features_v1(df)
        
        # Country-level aggregates
        if 'country' in df.columns and self.target_col in df.columns:
            country_stats = df.group_by('country').agg([
                pl.col(self.target_col).mean().alias('country_mean_cost'),
                pl.col(self.target_col).std().alias('country_std_cost'),
                pl.col(self.target_col).count().alias('country_count'),
            ])
            df = df.join(country_stats, on='country', how='left')
        
        return df
    
    def add_features_v3(self, df: pl.DataFrame) -> pl.DataFrame:
        """Maximum feature engineering."""
        df = self.add_features_v2(df)
        
        # Economic indicators
        if 'deflated_gdp_usd' in df.columns and 'us_cpi' in df.columns:
            df = df.with_columns([
                (pl.col('deflated_gdp_usd') / pl.col('us_cpi')).alias('gdp_cpi_ratio'),
            ])
        
        # Risk indicators
        risk_cols = ['seismic_hazard_zone', 'flood_risk_class', 'tropical_cyclone_wind_risk']
        for col in risk_cols:
            if col in df.columns:
                risk_map = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
                if df[col].dtype == pl.Utf8:
                    df = df.with_columns(
                        pl.col(col).replace(risk_map).fill_null(3).alias(f'{col}_numeric')
                    )
        
        # Distance features
        if 'straight_distance_to_capital_km' in df.columns:
            df = df.with_columns([
                pl.col('straight_distance_to_capital_km').log1p().alias('log_distance'),
            ])
        
        return df
    
    def prepare_data(self, df: pl.DataFrame, feature_version: int = 1) -> tuple:
        """Prepare X, y for training."""
        # Apply feature engineering
        if feature_version == 1:
            df = self.add_features_v1(df)
        elif feature_version == 2:
            df = self.add_features_v2(df)
        else:
            df = self.add_features_v3(df)
        
        # Encode categoricals
        df = self.encode_categoricals(df)
        
        # Get all numeric columns except target
        feature_cols = [c for c in df.columns 
                       if df[c].dtype.is_numeric() 
                       and c != self.target_col 
                       and c != 'data_id']
        
        X = df.select(feature_cols).fill_null(0).to_numpy()
        y = df.select(self.target_col).to_numpy().flatten()
        
        return X, y, feature_cols
    
    def run_experiment(self, strategy_name: str, feature_version: int, params: dict) -> OptimizationResult:
        """Run a single experiment."""
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy_name}")
        print(f"  Features: v{feature_version}, Params: {params.get('num_leaves', 31)} leaves, {params.get('learning_rate', 0.05)} LR")
        print('='*60)
        
        df = self.load_data()
        X, y, feature_cols = self.prepare_data(df, feature_version)
        
        print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")
        
        model = LGBMRegressor(**params, verbose=-1, n_jobs=-1)
        
        # 5-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        
        cv_mean = scores.mean()
        cv_std = scores.std()
        
        improvement = self.best_score - cv_mean if self.best_score != float('inf') else 0
        
        result = OptimizationResult(
            strategy=strategy_name,
            cv_score=cv_mean,
            cv_std=cv_std,
            features=len(feature_cols),
            params=params,
            improvement=improvement
        )
        
        # Check if this is the best
        if cv_mean < self.best_score:
            self.best_score = cv_mean
            self.no_improvement_count = 0
            print(f"  ✅ NEW BEST: {cv_mean:.4f} ± {cv_std:.4f} (improvement: {improvement:.4f})")
        else:
            self.no_improvement_count += 1
            print(f"  ❌ No improvement: {cv_mean:.4f} ± {cv_std:.4f}")
        
        self.results.append(result)
        return result
    
    def run(self) -> list[OptimizationResult]:
        """Run the full optimization loop."""
        print("\n" + "="*60)
        print("  🚀 AUTONOMOUS OPTIMIZATION - MAXIMUM PERFORMANCE MODE")
        print("="*60)
        print(f"  Target: {self.target_col}")
        print(f"  Budget: ${self.budget_usd}")
        print(f"  Convergence: Stop after {self.max_no_improvement} iterations without improvement")
        
        start_time = time.time()
        
        # Strategy 1: Baseline
        self.run_experiment(
            "Baseline",
            feature_version=1,
            params={'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 500}
        )
        
        # Strategy 2: More features
        self.run_experiment(
            "Extended Features",
            feature_version=2,
            params={'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 500}
        )
        
        # Strategy 3: Maximum features
        self.run_experiment(
            "Maximum Features",
            feature_version=3,
            params={'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 500}
        )
        
        # Strategy 4: Deeper trees
        if self.no_improvement_count < self.max_no_improvement:
            self.run_experiment(
                "Deeper Trees",
                feature_version=3,
                params={'num_leaves': 63, 'learning_rate': 0.03, 'n_estimators': 1000}
            )
        
        # Strategy 5: Regularized
        if self.no_improvement_count < self.max_no_improvement:
            self.run_experiment(
                "Regularized",
                feature_version=3,
                params={'num_leaves': 31, 'learning_rate': 0.02, 'n_estimators': 1500, 
                       'reg_alpha': 0.1, 'reg_lambda': 0.1}
            )
        
        # Strategy 6: Small learning rate
        if self.no_improvement_count < self.max_no_improvement:
            self.run_experiment(
                "Fine-tuned LR",
                feature_version=3,
                params={'num_leaves': 50, 'learning_rate': 0.01, 'n_estimators': 2000,
                       'min_child_samples': 10}
            )
        
        # Strategy 7: Larger ensemble
        if self.no_improvement_count < self.max_no_improvement:
            self.run_experiment(
                "Large Ensemble",
                feature_version=3,
                params={'num_leaves': 40, 'learning_rate': 0.015, 'n_estimators': 3000,
                       'bagging_fraction': 0.9, 'bagging_freq': 1, 'feature_fraction': 0.9}
            )
        
        elapsed = time.time() - start_time
        
        # Summary
        print("\n" + "="*60)
        print("  📊 OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"\n  Total iterations: {len(self.results)}")
        print(f"  Time: {elapsed:.1f} seconds")
        
        print("\n  Results by strategy:")
        print("  " + "-"*56)
        for r in sorted(self.results, key=lambda x: x.cv_score):
            status = "🏆" if r.cv_score == self.best_score else "  "
            print(f"  {status} {r.strategy:20} | RMSE: {r.cv_score:8.4f} ± {r.cv_std:.4f} | Features: {r.features}")
        
        best = min(self.results, key=lambda x: x.cv_score)
        print(f"\n  🏆 BEST: {best.strategy}")
        print(f"     RMSE: {best.cv_score:.4f} ± {best.cv_std:.4f}")
        print(f"     Features: {best.features}")
        print(f"     Params: {best.params}")
        
        return self.results


def main():
    optimizer = AutonomousOptimizer(
        train_path="/Users/a1234/Documents/vscode_projects/kaggle/solafune.construction-cost-prediction/data/train.csv",
        target_col="construction_cost_per_m2_usd",
        budget_usd=1.0
    )
    
    results = optimizer.run()
    
    # Generate submission with best model
    best = min(results, key=lambda x: x.cv_score)
    
    print("\n" + "="*60)
    print("  📝 GENERATING SUBMISSION WITH BEST MODEL")
    print("="*60)
    
    # Load data
    train = pl.read_csv("/Users/a1234/Documents/vscode_projects/kaggle/solafune.construction-cost-prediction/data/train.csv")
    test = pl.read_csv("/Users/a1234/Documents/vscode_projects/kaggle/solafune.construction-cost-prediction/data/test.csv")
    
    # Apply best feature engineering
    train = optimizer.add_features_v3(train)
    test = optimizer.add_features_v3(test)
    
    train = optimizer.encode_categoricals(train)
    test = optimizer.encode_categoricals(test)
    
    # Get features
    feature_cols = [c for c in train.columns 
                   if train[c].dtype.is_numeric() 
                   and c != optimizer.target_col 
                   and c != 'data_id'
                   and c in test.columns]
    
    X_train = train.select(feature_cols).fill_null(0).to_numpy()
    y_train = train.select(optimizer.target_col).to_numpy().flatten()
    X_test = test.select(feature_cols).fill_null(0).to_numpy()
    
    # Train with best params
    model = LGBMRegressor(**best.params, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Save
    submission = pl.DataFrame({
        'data_id': test['data_id'],
        'construction_cost_per_m2_usd': predictions
    })
    
    submission_path = Path("/Users/a1234/Documents/vscode_projects/kaggle/encodeclub.commit-to-change/submission_optimized.csv")
    submission.write_csv(submission_path)
    
    print(f"\n  ✅ Saved: {submission_path}")
    print(f"  Predictions: [{predictions.min():.2f}, {predictions.max():.2f}]")


if __name__ == "__main__":
    main()
