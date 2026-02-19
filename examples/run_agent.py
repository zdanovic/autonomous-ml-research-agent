#!/usr/bin/env python
"""
Hackathon Optimizer Agent CLI - Self-improving ML system.

Usage:
    # Run autonomous optimization (5 iterations)
    uv run python examples/run_agent.py --iterations 5

    # Run with specific target score
    uv run python examples/run_agent.py --target 0.18 --iterations 10

    # Run single strategy
    uv run python examples/run_agent.py --strategy country_specific_models

    # List available strategies
    uv run python examples/run_agent.py --list-strategies
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hackathon_agent import HackathonOptimizerAgent


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data."""
    data_dir = ROOT / "data"
    
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"Loaded train: {len(train)} rows")
    print(f"Loaded test: {len(test)} rows")
    
    return train, test


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic features for modeling."""
    df = df.copy()
    
    # Numeric features already available
    numeric_cols = [
        "year", "deflated_gdp_usd", "us_cpi", 
        "straight_distance_to_capital_km",
    ]
    
    # Binary access features
    binary_cols = [
        "access_to_airport", "access_to_port", 
        "access_to_highway", "access_to_railway",
        "developed_country", "landlocked",
    ]
    
    for col in binary_cols:
        if col in df.columns:
            df[f"{col}_enc"] = (df[col] == "Yes").astype(int)
    
    # Country encoding
    if "country" in df.columns:
        df["is_japan"] = (df["country"] == "Japan").astype(int)
        df["is_philippines"] = (df["country"] == "Philippines").astype(int)
    
    # Flood/risk encoding
    if "flood_risk_class" in df.columns:
        df["flood_risk_enc"] = (df["flood_risk_class"] == "Yes").astype(int)
    
    # Seismic hazard encoding
    if "seismic_hazard_zone" in df.columns:
        hazard_map = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
        df["seismic_hazard_enc"] = df["seismic_hazard_zone"].map(hazard_map).fillna(1)
    
    # Economic classification encoding
    if "region_economic_classification" in df.columns:
        econ_map = {
            "Low income": 0, "Lower-middle income": 1, 
            "Upper-middle income": 2, "High income": 3
        }
        df["economic_class_enc"] = df["region_economic_classification"].map(econ_map).fillna(1)
    
    # Cyclone/tornado risk encoding
    risk_map = {"Very Low": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
    if "tropical_cyclone_wind_risk" in df.columns:
        df["cyclone_risk_enc"] = df["tropical_cyclone_wind_risk"].map(risk_map).fillna(1)
    if "tornadoes_wind_risk" in df.columns:
        df["tornado_risk_enc"] = df["tornadoes_wind_risk"].map(risk_map).fillna(0)
    
    # Climate zone encoding (simplified)
    if "koppen_climate_zone" in df.columns:
        climate_map = {"Af": 0, "Am": 1, "Aw": 2, "Cfa": 3, "Cfb": 4, "Cwa": 5, "Dfa": 6}
        df["climate_enc"] = df["koppen_climate_zone"].map(climate_map).fillna(0)
    
    # Year features
    if "year" in df.columns:
        df["year_normalized"] = (df["year"] - 2019) / 5
    
    # GDP log
    if "deflated_gdp_usd" in df.columns:
        df["log_gdp"] = np.log1p(df["deflated_gdp_usd"])
    
    # Distance features
    if "straight_distance_to_capital_km" in df.columns:
        df["log_distance_capital"] = np.log1p(df["straight_distance_to_capital_km"])
    
    # Fill NaN for all numeric columns
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Hackathon Optimizer Agent - Self-improving ML system"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Maximum optimization iterations (default: 5)",
    )
    
    parser.add_argument(
        "--target", "-t",
        type=float,
        default=0.18,
        help="Target RMSLE score (default: 0.18)",
    )
    
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default=None,
        help="Run single strategy by name",
    )
    
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all available strategies",
    )
    
    parser.add_argument(
        "--experiments-path",
        type=str,
        default=None,
        help="Path to experiments.csv for knowledge base",
    )
    
    args = parser.parse_args()
    
    # List strategies mode
    if args.list_strategies:
        from src.hackathon_agent.strategist import STRATEGIES
        print("\n📋 Available Strategies:\n")
        for name, s in STRATEGIES.items():
            print(f"  {name}")
            print(f"    {s.description}")
            print(f"    Priority: {s.priority}, Expected gain: {s.expected_gain:.4f}")
            print()
        return 0
    
    # Load data
    train, test = load_data()
    
    # Add features
    train = add_basic_features(train)
    test = add_basic_features(test)
    
    # Initialize agent
    agent = HackathonOptimizerAgent(
        train_df=train,
        test_df=test,
        target_score=args.target,
        submissions_dir=str(ROOT / "submissions"),
        experiments_path=args.experiments_path,
    )
    
    # Single strategy mode
    if args.strategy:
        result = agent.run_single_strategy(args.strategy)
        if result:
            print(f"\n✅ Result: CV={result.cv_score:.5f}")
            print(f"   Submission: {result.submission_path}")
        return 0
    
    # Full autonomous mode
    result = agent.run(
        max_iterations=args.iterations,
        strategies_per_iteration=2,
        verbose=True,
    )
    
    # Save history
    agent.save_history(str(ROOT / "agent_history.json"))
    
    if result:
        print(f"\n🏆 Best Result: {result.strategy_name}")
        print(f"   CV Score: {result.cv_score:.5f}")
        print(f"   Submission: {result.submission_path}")
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
