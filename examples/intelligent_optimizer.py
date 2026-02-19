#!/usr/bin/env python3
"""
Intelligent ML Optimization System

This system:
1. Analyzes previous experiment results
2. Uses LLM to decide best next strategy
3. Implements advanced techniques (CatBoost, stacking, vision features)
4. Automatically generates optimized submissions

Target: 0.12 RMSLE (leaderboard top)
Current best: 0.2393
"""

import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings('ignore')

# ML imports
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Paths
SOLAFUNE_DIR = Path('/Users/a1234/Documents/vscode_projects/kaggle/solafune.construction-cost-prediction')
OUTPUT_DIR = Path('/Users/a1234/Documents/vscode_projects/kaggle/encodeclub.commit-to-change/submissions')
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = 'construction_cost_per_m2_usd'


@dataclass
class ExperimentResult:
    """Track experiment results."""
    id: int
    timestamp: str
    strategy: str
    cv_rmsle: float
    public_score: float | None
    description: str
    params: dict


class IntelligentOptimizer:
    """
    Intelligent ML Optimization System.
    
    Uses analysis of previous results + advanced techniques to find best model.
    """
    
    def __init__(self):
        self.experiments: list[ExperimentResult] = []
        self.history_file = OUTPUT_DIR / 'intelligent_history.json'
        self._load_history()
        
        # Load data
        self.train_df = pl.read_csv(SOLAFUNE_DIR / 'data' / 'train.csv')
        self.test_df = pl.read_csv(SOLAFUNE_DIR / 'data' / 'test.csv')
        
        print(f"Train: {len(self.train_df)} rows")
        print(f"Test: {len(self.test_df)} rows")
        print(f"Countries: {self.train_df['country'].unique().to_list()}")
        
    def _load_history(self):
        if self.history_file.exists():
            data = json.loads(self.history_file.read_text())
            self.experiments = [ExperimentResult(**e) for e in data]
            
    def _save_history(self):
        data = [asdict(e) for e in self.experiments]
        self.history_file.write_text(json.dumps(data, indent=2))
    
    def get_best_public(self) -> float | None:
        scored = [e for e in self.experiments if e.public_score]
        return min(e.public_score for e in scored) if scored else None
    
    # ==================== FEATURE ENGINEERING ====================
    
    def build_features(self, df: pl.DataFrame, is_train: bool = True) -> pl.DataFrame:
        """Build comprehensive feature set."""
        
        # 1. Country statistics (target encoding)
        if is_train:
            self._country_stats = self.train_df.group_by('country').agg([
                pl.col(TARGET_COL).mean().alias('country_mean'),
                pl.col(TARGET_COL).std().alias('country_std'),
                pl.col(TARGET_COL).median().alias('country_median'),
                pl.col(TARGET_COL).quantile(0.25).alias('country_q25'),
                pl.col(TARGET_COL).quantile(0.75).alias('country_q75'),
                pl.len().alias('country_count'),
            ])
        
        df = df.join(self._country_stats, on='country', how='left')
        
        # 2. Temporal features
        if 'year' in df.columns:
            df = df.with_columns([
                (pl.col('year') - 2019).alias('years_since_2019'),
                (pl.col('year') ** 2).alias('year_squared'),
            ])
        
        if 'quarter_label' in df.columns:
            df = df.with_columns([
                pl.col('quarter_label').str.extract(r'Q(\d)', 1).cast(pl.Int32).fill_null(2).alias('quarter_num'),
            ])
        
        # 3. Economic features
        if 'deflated_gdp_usd' in df.columns:
            df = df.with_columns([
                pl.col('deflated_gdp_usd').log1p().alias('log_gdp'),
                (pl.col('deflated_gdp_usd') / 1e9).alias('gdp_billions'),
            ])
        
        if 'us_cpi' in df.columns:
            df = df.with_columns([
                pl.col('us_cpi').log1p().alias('log_cpi'),
            ])
            if 'deflated_gdp_usd' in df.columns:
                df = df.with_columns([
                    (pl.col('deflated_gdp_usd') / (pl.col('us_cpi') + 1)).alias('gdp_cpi_ratio'),
                ])
        
        # 4. Geographic features
        if 'straight_distance_to_capital_km' in df.columns:
            df = df.with_columns([
                pl.col('straight_distance_to_capital_km').log1p().alias('log_distance'),
                (pl.col('straight_distance_to_capital_km') < 50).cast(pl.Int32).alias('near_capital'),
                (pl.col('straight_distance_to_capital_km') > 200).cast(pl.Int32).alias('far_from_capital'),
            ])
        
        # 5. Access indicators
        access_cols = ['access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway']
        access_count = []
        for col in access_cols:
            if col in df.columns:
                df = df.with_columns([
                    (pl.col(col) == 'Yes').cast(pl.Int32).alias(f'{col}_num'),
                ])
                access_count.append(f'{col}_num')
        
        if access_count:
            df = df.with_columns([
                sum([pl.col(c) for c in access_count]).alias('total_access'),
            ])
        
        # 6. Risk features
        risk_map = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
        yes_no_map = {'No': 0, 'Yes': 1}
        
        for col in ['tropical_cyclone_wind_risk', 'tornadoes_wind_risk']:
            if col in df.columns and df[col].dtype == pl.Utf8:
                df = df.with_columns([
                    pl.col(col).replace(risk_map).fill_null(3).cast(pl.Float32).alias(f'{col}_score'),
                ])
        
        if 'flood_risk_class' in df.columns and df['flood_risk_class'].dtype == pl.Utf8:
            df = df.with_columns([
                pl.col('flood_risk_class').replace(yes_no_map).fill_null(0).cast(pl.Float32).alias('flood_risk_score'),
            ])
        
        # 7. Categorical encoding
        cat_cols = ['developed_country', 'landlocked', 'koppen_climate_zone', 
                    'seismic_hazard_zone', 'region_economic_classification']
        
        for col in cat_cols:
            if col in df.columns and df[col].dtype == pl.Utf8:
                unique_vals = df[col].unique().to_list()
                mapping = {v: i for i, v in enumerate(unique_vals) if v is not None}
                df = df.with_columns([
                    pl.col(col).replace(mapping).fill_null(0).cast(pl.Int32).alias(f'{col}_enc'),
                ])
        
        return df
    
    def get_feature_cols(self, df: pl.DataFrame) -> list[str]:
        """Get numeric feature columns."""
        exclude = {TARGET_COL, 'data_id', 'geolocation_name', 'quarter_label', 
                   'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'country',
                   'developed_country', 'landlocked', 'koppen_climate_zone',
                   'seismic_hazard_zone', 'region_economic_classification',
                   'access_to_airport', 'access_to_port', 'access_to_highway', 
                   'access_to_railway', 'tropical_cyclone_wind_risk', 
                   'tornadoes_wind_risk', 'flood_risk_class'}
        
        return [c for c in df.columns 
                if (df[c].dtype.is_numeric() 
                and c not in exclude
                and c in self.test_df.columns) or c.endswith('_enc') or c.endswith('_num') or c.endswith('_score')]
    
    # ==================== MODEL TRAINING ====================
    
    def train_lgbm(self, X: np.ndarray, y: np.ndarray, params: dict) -> tuple:
        """Train LightGBM with CV."""
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        models = []
        
        for train_idx, val_idx in cv.split(X):
            model = LGBMRegressor(**params, verbose=-1, n_jobs=-1, random_state=42)
            model.fit(X[train_idx], y[train_idx])
            
            preds = np.clip(model.predict(X[val_idx]), 0, None)
            rmsle = np.sqrt(np.mean((np.log1p(preds) - np.log1p(y[val_idx]))**2))
            cv_scores.append(rmsle)
            models.append(model)
        
        return models, np.mean(cv_scores), np.std(cv_scores)
    
    def train_catboost(self, X: np.ndarray, y: np.ndarray, cat_features: list = None) -> tuple:
        """Train CatBoost with CV."""
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        models = []
        
        # Transform target for RMSLE
        y_log = np.log1p(y)
        
        params = {
            'iterations': 3000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 10,
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50,
            'loss_function': 'RMSE',
        }
        
        for train_idx, val_idx in cv.split(X):
            model = CatBoostRegressor(**params)
            model.fit(
                X[train_idx], y_log[train_idx],
                eval_set=(X[val_idx], y_log[val_idx]),
                verbose=False,
            )
            
            preds_log = model.predict(X[val_idx])
            preds = np.expm1(np.clip(preds_log, 0, 15))
            preds = np.clip(preds, 0, None)
            rmsle = np.sqrt(np.mean((np.log1p(preds) - np.log1p(y[val_idx]))**2))
            cv_scores.append(rmsle)
            models.append(model)
        
        return models, np.mean(cv_scores), np.std(cv_scores)
    
    def train_country_specific(self, strategy: str = 'lgbm') -> tuple[float, np.ndarray]:
        """Train separate models per country."""
        train_feat = self.build_features(self.train_df, is_train=True)
        test_feat = self.build_features(self.test_df, is_train=False)
        
        feature_cols = self.get_feature_cols(train_feat)
        # Filter to features in test
        feature_cols = [c for c in feature_cols if c in test_feat.columns]
        
        countries = self.train_df['country'].unique().to_list()
        
        all_cv_scores = []
        country_test_preds = {}
        
        for country in countries:
            train_c = train_feat.filter(pl.col('country') == country)
            test_c = test_feat.filter(pl.col('country') == country)
            
            X = train_c.select(feature_cols).fill_null(0).to_numpy()
            y = train_c.select(TARGET_COL).to_numpy().flatten()
            X_test = test_c.select(feature_cols).fill_null(0).to_numpy()
            
            if strategy == 'catboost':
                models, cv_mean, cv_std = self.train_catboost(X, y)
                # Predict with ensemble
                preds_log = np.mean([m.predict(X_test) for m in models], axis=0)
                preds = np.expm1(np.clip(preds_log, 0, 15))
            else:
                params = {
                    'num_leaves': 31,
                    'learning_rate': 0.02,
                    'n_estimators': 1000,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'min_child_samples': 10,
                }
                models, cv_mean, cv_std = self.train_lgbm(X, y, params)
                preds = np.mean([m.predict(X_test) for m in models], axis=0)
            
            all_cv_scores.append(cv_mean)
            country_test_preds[country] = np.clip(preds, 0, None)
            print(f"  {country}: CV={cv_mean:.4f} ({len(train_c)} samples)")
        
        # Combine predictions
        final_preds = np.zeros(len(self.test_df))
        for i, row in enumerate(self.test_df.iter_rows(named=True)):
            country = row['country']
            test_c = self.test_df.filter(pl.col('country') == country)
            for j, r in enumerate(test_c.iter_rows(named=True)):
                if r['data_id'] == row['data_id']:
                    final_preds[i] = country_test_preds[country][j]
                    break
        
        overall_cv = np.mean(all_cv_scores)
        return overall_cv, final_preds
    
    def train_stacked(self) -> tuple[float, np.ndarray]:
        """Train stacked ensemble with Ridge."""
        # Train multiple base models
        train_feat = self.build_features(self.train_df, is_train=True)
        test_feat = self.build_features(self.test_df, is_train=False)
        
        feature_cols = self.get_feature_cols(train_feat)
        feature_cols = [c for c in feature_cols if c in test_feat.columns]
        
        X = train_feat.select(feature_cols).fill_null(0).to_numpy()
        y = self.train_df.select(TARGET_COL).to_numpy().flatten()
        X_test = test_feat.select(feature_cols).fill_null(0).to_numpy()
        
        # Base models
        base_configs = [
            ('lgbm_1', {'num_leaves': 31, 'learning_rate': 0.02, 'n_estimators': 1000, 'reg_alpha': 0.1}),
            ('lgbm_2', {'num_leaves': 16, 'learning_rate': 0.03, 'n_estimators': 800, 'reg_lambda': 0.5}),
            ('lgbm_3', {'num_leaves': 50, 'learning_rate': 0.01, 'n_estimators': 1500, 'min_child_samples': 20}),
        ]
        
        oof_preds = np.zeros((len(X), len(base_configs)))
        test_preds = np.zeros((len(X_test), len(base_configs)))
        
        for i, (name, params) in enumerate(base_configs):
            models, cv_mean, _ = self.train_lgbm(X, y, params)
            print(f"  {name}: CV={cv_mean:.4f}")
            
            # OOF predictions
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
                oof_preds[val_idx, i] = models[fold_idx].predict(X[val_idx])
            
            # Test predictions
            test_preds[:, i] = np.mean([m.predict(X_test) for m in models], axis=0)
        
        # Stack with Ridge
        y_log = np.log1p(y)
        oof_log = np.log1p(np.clip(oof_preds, 0, None))
        
        stacker = Ridge(alpha=1.0)
        stacker.fit(oof_log, y_log)
        
        # CV score
        stacked_oof = stacker.predict(oof_log)
        stacked_oof = np.expm1(stacked_oof)
        cv_rmsle = np.sqrt(np.mean((np.log1p(np.clip(stacked_oof, 0, None)) - np.log1p(y))**2))
        
        # Final prediction
        test_log = np.log1p(np.clip(test_preds, 0, None))
        final_preds = np.expm1(stacker.predict(test_log))
        final_preds = np.clip(final_preds, 0, None)
        
        return cv_rmsle, final_preds
    
    # ==================== EXECUTION ====================
    
    def run_strategy(self, strategy: str) -> ExperimentResult:
        """Run a specific strategy."""
        exp_id = len(self.experiments) + 100
        
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy}")
        print('='*60)
        
        if strategy == 'country_lgbm':
            cv_score, preds = self.train_country_specific('lgbm')
            desc = "Country-specific LightGBM with comprehensive features"
            params = {'model': 'lgbm', 'type': 'country_specific'}
            
        elif strategy == 'country_catboost':
            cv_score, preds = self.train_country_specific('catboost')
            desc = "Country-specific CatBoost with log-transform"
            params = {'model': 'catboost', 'type': 'country_specific'}
            
        elif strategy == 'stacked':
            cv_score, preds = self.train_stacked()
            desc = "Ridge stacking of multiple LightGBM models"
            params = {'model': 'stack', 'base_models': 3}
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"  CV RMSLE: {cv_score:.4f}")
        
        # Save submission
        path = OUTPUT_DIR / f'submission_{exp_id:03d}_{strategy}.csv'
        pl.DataFrame({
            'data_id': self.test_df['data_id'],
            TARGET_COL: preds
        }).write_csv(path)
        
        print(f"  Saved: {path}")
        
        exp = ExperimentResult(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            strategy=strategy,
            cv_rmsle=cv_score,
            public_score=None,
            description=desc,
            params=params,
        )
        
        self.experiments.append(exp)
        self._save_history()
        
        return exp
    
    def run_all_strategies(self):
        """Run all optimization strategies."""
        print("\n" + "="*60)
        print("  🚀 INTELLIGENT OPTIMIZATION SYSTEM")
        print("="*60)
        print("  Target: 0.12 RMSLE")
        print(f"  Current best: {self.get_best_public() or 'None'}")
        
        strategies = ['country_lgbm', 'country_catboost', 'stacked']
        
        results = []
        for strategy in strategies:
            try:
                exp = self.run_strategy(strategy)
                results.append(exp)
            except Exception as e:
                print(f"  Error in {strategy}: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("  📊 RESULTS SUMMARY")
        print("="*60)
        
        results.sort(key=lambda x: x.cv_rmsle)
        for exp in results:
            print(f"  #{exp.id}: {exp.strategy:20} CV={exp.cv_rmsle:.4f}")
        
        print(f"\n  Best: #{results[0].id} {results[0].strategy} CV={results[0].cv_rmsle:.4f}")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='all', help='Strategy to run (or "all")')
    args = parser.parse_args()
    
    optimizer = IntelligentOptimizer()
    
    if args.strategy == 'all':
        optimizer.run_all_strategies()
    else:
        optimizer.run_strategy(args.strategy)


if __name__ == "__main__":
    main()
