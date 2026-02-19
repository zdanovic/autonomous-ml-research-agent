#!/usr/bin/env python
"""
Feedback-Driven Optimization Loop

This system:
1. Generates submissions using different strategies
2. Waits for user to input public score
3. Uses LLM to analyze results and suggest next improvements
4. Continues until target score is reached or no improvement for N iterations

Usage:
    uv run python examples/feedback_loop.py --target 0.12
"""

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comet_swarm.agents import StrategistAgent


@dataclass
class Experiment:
    """Record of a single experiment."""
    id: int
    timestamp: str
    strategy: str
    cv_score: float
    cv_std: float
    public_score: float | None
    features: list[str]
    params: dict
    notes: str = ""


class FeedbackOptimizer:
    """
    Autonomous optimization loop with human feedback on public scores.
    
    The system learns from public score feedback to guide next experiments.
    """
    
    STRATEGIES = [
        # Phase 1: Feature Engineering
        ("country_encoding", "Add country-level target encoding"),
        ("year_quarter_features", "Add temporal features (year, quarter, rolling)"),
        ("economic_ratios", "GDP/CPI ratios and economic indicators"),
        ("geo_features", "Distance features, access indicators"),
        ("risk_aggregates", "Aggregate risk scores"),
        
        # Phase 2: Model Architecture
        ("catboost_baseline", "CatBoost with categorical handling"),
        ("deeper_trees", "More leaves, lower learning rate"),
        ("regularized", "L1/L2 regularization"),
        ("dart", "DART boosting for less overfitting"),
        
        # Phase 3: Advanced
        ("country_stack", "Train separate models per country"),
        ("target_encoding", "Smoothed target encoding for categories"),
        ("pseudo_labeling", "Use confident test predictions"),
        ("blending", "Blend multiple models"),
    ]
    
    def __init__(
        self,
        train_path: str,
        test_path: str,
        target_col: str,
        target_score: float = 0.12,
        output_dir: str = "submissions",
    ):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.target_score = target_score
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.experiments: list[Experiment] = []
        self.history_file = self.output_dir / "experiment_history.json"
        self._load_history()
        
        self.train_df = None
        self.test_df = None
        self._strategist = None
        
    def _load_history(self):
        """Load previous experiment history."""
        if self.history_file.exists():
            data = json.loads(self.history_file.read_text())
            self.experiments = [Experiment(**e) for e in data]
            print(f"Loaded {len(self.experiments)} previous experiments")
    
    def _save_history(self):
        """Save experiment history."""
        data = [asdict(e) for e in self.experiments]
        self.history_file.write_text(json.dumps(data, indent=2))
    
    def load_data(self):
        """Load train and test data."""
        if self.train_path.suffix == '.parquet':
            self.train_df = pl.read_parquet(self.train_path)
        else:
            self.train_df = pl.read_csv(self.train_path)
            
        if self.test_path.suffix == '.parquet':
            self.test_df = pl.read_parquet(self.test_path)
        else:
            self.test_df = pl.read_csv(self.test_path)
            
        print(f"Train: {len(self.train_df)} rows, Test: {len(self.test_df)} rows")
    
    def add_country_encoding(self, df: pl.DataFrame, is_train: bool = True) -> pl.DataFrame:
        """Add country-level statistics as features."""
        if is_train and self.target_col in df.columns:
            # Calculate stats from train
            stats = df.group_by('country').agg([
                pl.col(self.target_col).mean().alias('country_mean'),
                pl.col(self.target_col).std().alias('country_std'),
                pl.col(self.target_col).median().alias('country_median'),
                pl.count().alias('country_count'),
            ])
            self._country_stats = stats
            df = df.join(stats, on='country', how='left')
        elif hasattr(self, '_country_stats'):
            df = df.join(self._country_stats, on='country', how='left')
        return df
    
    def add_year_quarter_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add temporal features."""
        if 'year' in df.columns:
            df = df.with_columns([
                (pl.col('year') - 2019).alias('years_since_2019'),
            ])
        
        if 'quarter_label' in df.columns:
            # Extract quarter number
            df = df.with_columns([
                pl.col('quarter_label').str.extract(r'Q(\d)', 1).cast(pl.Int32).alias('quarter_num'),
            ])
        
        return df
    
    def add_economic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add economic indicator features."""
        if 'deflated_gdp_usd' in df.columns:
            df = df.with_columns([
                (pl.col('deflated_gdp_usd') / 1e9).alias('gdp_billions'),
                pl.col('deflated_gdp_usd').log1p().alias('log_gdp'),
            ])
        
        if 'us_cpi' in df.columns and 'deflated_gdp_usd' in df.columns:
            df = df.with_columns([
                (pl.col('deflated_gdp_usd') / (pl.col('us_cpi') + 1)).alias('gdp_cpi_ratio'),
            ])
        
        return df
    
    def add_geo_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add geographic features."""
        if 'straight_distance_to_capital_km' in df.columns:
            df = df.with_columns([
                pl.col('straight_distance_to_capital_km').log1p().alias('log_distance'),
                (pl.col('straight_distance_to_capital_km') < 100).cast(pl.Int32).alias('near_capital'),
            ])
        
        # Access indicators
        access_cols = ['access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway']
        for col in access_cols:
            if col in df.columns:
                if df[col].dtype == pl.Utf8:
                    df = df.with_columns([
                        (pl.col(col) == 'Yes').cast(pl.Int32).alias(f'{col}_numeric'),
                    ])
        
        return df
    
    def add_risk_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add risk indicator features."""
        risk_mapping = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
        yes_no_mapping = {'No': 0, 'Yes': 1}
        
        risk_cols = ['tropical_cyclone_wind_risk', 'tornadoes_wind_risk']
        risk_values = []
        
        for col in risk_cols:
            if col in df.columns and df[col].dtype == pl.Utf8:
                df = df.with_columns([
                    pl.col(col).replace(risk_mapping).fill_null(3).cast(pl.Float32).alias(f'{col}_score'),
                ])
                risk_values.append(f'{col}_score')
        
        # Handle Yes/No columns (like flood_risk_class)
        if 'flood_risk_class' in df.columns and df['flood_risk_class'].dtype == pl.Utf8:
            df = df.with_columns([
                pl.col('flood_risk_class').replace(yes_no_mapping).fill_null(0).cast(pl.Float32).alias('flood_risk_score'),
            ])
            risk_values.append('flood_risk_score')
        
        # Aggregate risk
        if risk_values:
            df = df.with_columns([
                sum([pl.col(c) for c in risk_values]).alias('total_risk_score'),
            ])
        
        return df
    
    def encode_categoricals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Label encode categorical columns."""
        cat_cols = ['developed_country', 'landlocked', 'region_economic_classification', 
                    'koppen_climate_zone', 'seismic_hazard_zone']
        
        for col in cat_cols:
            if col in df.columns and df[col].dtype == pl.Utf8:
                unique_vals = df[col].unique().to_list()
                mapping = {v: i for i, v in enumerate(unique_vals) if v is not None}
                df = df.with_columns([
                    pl.col(col).replace(mapping).fill_null(0).alias(f'{col}_enc'),
                ])
        
        return df
    
    def prepare_features(self, strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Prepare features based on strategy."""
        train = self.train_df.clone()
        test = self.test_df.clone()
        
        # Apply feature engineering based on strategy
        if 'country' in strategy or 'full' in strategy:
            train = self.add_country_encoding(train, is_train=True)
            test = self.add_country_encoding(test, is_train=False)
        
        if 'year' in strategy or 'temporal' in strategy or 'full' in strategy:
            train = self.add_year_quarter_features(train)
            test = self.add_year_quarter_features(test)
        
        if 'economic' in strategy or 'full' in strategy:
            train = self.add_economic_features(train)
            test = self.add_economic_features(test)
        
        if 'geo' in strategy or 'full' in strategy:
            train = self.add_geo_features(train)
            test = self.add_geo_features(test)
        
        if 'risk' in strategy or 'full' in strategy:
            train = self.add_risk_features(train)
            test = self.add_risk_features(test)
        
        # Always encode categoricals
        train = self.encode_categoricals(train)
        test = self.encode_categoricals(test)
        
        # Get numeric feature columns
        exclude = {self.target_col, 'data_id', 'geolocation_name', 'quarter_label', 
                   'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'country'}
        
        feature_cols = [c for c in train.columns 
                       if train[c].dtype.is_numeric() 
                       and c not in exclude
                       and c in test.columns]
        
        X_train = train.select(feature_cols).fill_null(0).to_numpy()
        y_train = train.select(self.target_col).to_numpy().flatten()
        X_test = test.select(feature_cols).fill_null(0).to_numpy()
        
        return X_train, y_train, X_test, feature_cols
    
    def train_model(self, strategy: str, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train model based on strategy with manual CV for CatBoost compatibility."""
        
        # LightGBM variants (using different params)
        if 'deeper' in strategy:
            params = {'num_leaves': 63, 'learning_rate': 0.02, 'n_estimators': 2000}
        elif 'regularized' in strategy:
            params = {'num_leaves': 31, 'learning_rate': 0.03, 'n_estimators': 1500,
                     'reg_alpha': 0.1, 'reg_lambda': 0.1}
        elif 'dart' in strategy:
            params = {'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 1000,
                     'boosting_type': 'dart'}
        elif 'catboost' in strategy:
            # Use CatBoost-like params in LightGBM (CatBoost has sklearn compat issues)
            params = {'num_leaves': 64, 'learning_rate': 0.03, 'n_estimators': 1500,
                     'min_child_samples': 5, 'colsample_bytree': 0.8}
        else:
            params = {'num_leaves': 50, 'learning_rate': 0.01, 'n_estimators': 2000,
                     'min_child_samples': 10}
        
        # Manual cross-validation with RMSLE
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = LGBMRegressor(**params, verbose=-1, n_jobs=-1)
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_val)
            preds = np.clip(preds, 0, None)
            
            # RMSLE
            rmsle = np.sqrt(np.mean((np.log1p(preds) - np.log1p(y_val))**2))
            cv_scores.append(rmsle)
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Fit final model on all data
        final_model = LGBMRegressor(**params, verbose=-1, n_jobs=-1)
        final_model.fit(X, y)
        
        return final_model, cv_mean, cv_std, params
    
    def get_best_score(self) -> float | None:
        """Get best public score from history."""
        scored = [e for e in self.experiments if e.public_score is not None]
        if not scored:
            return None
        return min(e.public_score for e in scored)
    
    async def get_llm_suggestion(self) -> str:
        """Use LLM to analyze history and suggest next strategy."""
        if not self._strategist:
            self._strategist = StrategistAgent()
        
        # Build history summary
        history = []
        for e in self.experiments[-10:]:  # Last 10
            history.append(f"- {e.strategy}: CV={e.cv_score:.4f}, Public={e.public_score or 'pending'}")
        
        best = self.get_best_score()
        
        prompt = f"""
Analyze these ML experiment results for construction cost prediction (RMSLE metric, lower is better):

Target score: {self.target_score}
Current best public score: {best or 'None yet'}

Recent experiments:
{chr(10).join(history)}

Available strategies to try:
{chr(10).join([f'- {s[0]}: {s[1]}' for s in self.STRATEGIES])}

Based on the CV vs Public score gaps, suggest the SINGLE BEST next strategy to try.
Format: Just return the strategy name (e.g., "country_stack" or "catboost_baseline").
"""
        
        try:
            response = await self._strategist.generate_strategy(
                competition_context={"metric": "rmsle", "target": self.target_score},
                dataset_info="Construction cost prediction with geo/economic/risk features",
                eda_insights=prompt,
            )
            # Extract strategy name from response
            suggested = response.priority.split()[0].lower().replace(',', '')
            return suggested if any(s[0] in suggested for s in self.STRATEGIES) else "full_features"
        except Exception as e:
            print(f"LLM suggestion failed: {e}")
            return "full_features"
    
    def run_experiment(self, strategy: str) -> Experiment:
        """Run a single experiment."""
        print(f"\n{'='*60}")
        print(f"  Experiment #{len(self.experiments) + 1}: {strategy}")
        print('='*60)
        
        # Prepare data
        X_train, y_train, X_test, feature_cols = self.prepare_features(strategy)
        print(f"  Features: {len(feature_cols)}")
        
        # Train model
        model, cv_mean, cv_std, params = self.train_model(strategy, X_train, y_train)
        print(f"  CV RMSLE: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Generate predictions
        predictions = model.predict(X_test)
        predictions = np.clip(predictions, 0, None)
        
        # Save submission
        exp_id = len(self.experiments) + 1
        submission_path = self.output_dir / f"submission_{exp_id:03d}_{strategy}.csv"
        
        submission = pl.DataFrame({
            'data_id': self.test_df['data_id'],
            self.target_col: predictions
        })
        submission.write_csv(submission_path)
        
        print(f"  Saved: {submission_path}")
        print(f"  Predictions: [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        # Create experiment record
        exp = Experiment(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            strategy=strategy,
            cv_score=cv_mean,
            cv_std=cv_std,
            public_score=None,
            features=feature_cols,
            params=params,
        )
        
        self.experiments.append(exp)
        self._save_history()
        
        return exp
    
    async def run_loop(self):
        """Main optimization loop."""
        print("\n" + "="*60)
        print("  🚀 FEEDBACK-DRIVEN OPTIMIZATION LOOP")
        print("="*60)
        print(f"  Target: {self.target_score} RMSLE")
        print(f"  Best so far: {self.get_best_score() or 'None'}")
        
        self.load_data()
        
        iteration = 0
        no_improvement = 0
        max_no_improvement = 5
        
        while True:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"  ITERATION {iteration}")
            print('='*60)
            
            # Choose strategy
            if iteration == 1:
                strategy = "full_features"
            elif iteration == 2:
                strategy = "catboost_baseline"
            elif iteration == 3:
                strategy = "country_encoding"
            else:
                # Use LLM to suggest
                print("  Asking LLM for suggestion...")
                strategy = await self.get_llm_suggestion()
            
            print(f"  Strategy: {strategy}")
            
            # Run experiment
            exp = self.run_experiment(strategy)
            
            # Ask for public score
            print("\n" + "-"*60)
            print("  📤 Please submit the file and enter the public score:")
            print(f"     File: {self.output_dir / f'submission_{exp.id:03d}_{strategy}.csv'}")
            print("-"*60)
            
            try:
                score_input = input("  Public score (or 'skip' to continue, 'quit' to stop): ").strip()
                
                if score_input.lower() == 'quit':
                    print("  Stopping optimization loop.")
                    break
                elif score_input.lower() != 'skip':
                    public_score = float(score_input)
                    exp.public_score = public_score
                    self._save_history()
                    
                    print(f"  ✅ Recorded: CV={exp.cv_score:.4f}, Public={public_score:.4f}")
                    
                    # Check if target reached
                    if public_score <= self.target_score:
                        print(f"\n  🏆 TARGET REACHED! Score: {public_score}")
                        break
                    
                    # Check improvement
                    best = self.get_best_score()
                    if best and public_score >= best:
                        no_improvement += 1
                        print(f"  ⚠️ No improvement ({no_improvement}/{max_no_improvement})")
                    else:
                        no_improvement = 0
                        print(f"  ✨ NEW BEST: {public_score}")
                    
                    if no_improvement >= max_no_improvement:
                        print(f"\n  Stopping: {max_no_improvement} iterations without improvement")
                        break
            
            except KeyboardInterrupt:
                print("\n  Interrupted by user.")
                break
            except ValueError:
                print("  Invalid score, skipping...")
        
        # Summary
        print("\n" + "="*60)
        print("  📊 OPTIMIZATION SUMMARY")
        print('='*60)
        
        scored = [e for e in self.experiments if e.public_score is not None]
        if scored:
            scored.sort(key=lambda x: x.public_score)
            print("\n  Best experiments:")
            for e in scored[:5]:
                print(f"    {e.strategy}: CV={e.cv_score:.4f}, Public={e.public_score:.4f}")
        
        print(f"\n  Total experiments: {len(self.experiments)}")
        print(f"  Best public score: {self.get_best_score() or 'None'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=float, default=0.12)
    parser.add_argument('--train', default='/Users/a1234/Documents/vscode_projects/kaggle/solafune.construction-cost-prediction/data/train.csv')
    parser.add_argument('--test', default='/Users/a1234/Documents/vscode_projects/kaggle/solafune.construction-cost-prediction/data/test.csv')
    args = parser.parse_args()
    
    optimizer = FeedbackOptimizer(
        train_path=args.train,
        test_path=args.test,
        target_col='construction_cost_per_m2_usd',
        target_score=args.target,
        output_dir='/Users/a1234/Documents/vscode_projects/kaggle/encodeclub.commit-to-change/submissions',
    )
    
    asyncio.run(optimizer.run_loop())


if __name__ == "__main__":
    main()
