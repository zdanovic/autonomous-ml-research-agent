#!/usr/bin/env python3
"""
Intelligent Hackathon Agent CLI.

Run with: uv run python examples/run_intelligent_agent.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hackathon_agent.intelligent_agent import IntelligentAgent


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load competition data."""
    data_path = ROOT / "data"
    
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    
    print(f"Loaded train: {len(train)} rows")
    print(f"Loaded test: {len(test)} rows")
    
    return train, test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features for modeling."""
    # Year features
    if "year" in df.columns:
        df["year_normalized"] = (df["year"] - 2000) / 25
    
    # Economic features
    if "deflated_gdp_usd" in df.columns:
        df["log_gdp"] = np.log1p(df["deflated_gdp_usd"])
    
    # Distance features
    if "straight_distance_to_capital_km" in df.columns:
        df["log_distance_capital"] = np.log1p(df["straight_distance_to_capital_km"])
    
    # Country encoding
    if "country" in df.columns:
        df["is_japan"] = (df["country"] == "Japan").astype(int)
        df["is_philippines"] = (df["country"] == "Philippines").astype(int)
    
    # Risk encodings
    if "flood_risk_class" in df.columns:
        df["flood_risk_enc"] = (df["flood_risk_class"] == "Yes").astype(int)
    
    if "seismic_hazard_zone" in df.columns:
        hazard_map = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
        df["seismic_hazard_enc"] = df["seismic_hazard_zone"].map(hazard_map).fillna(1)
    
    if "economic_class" in df.columns:
        econ_map = {"Low Income": 0, "Lower-Middle Income": 1, "Upper-Middle Income": 2, "High Income": 3}
        df["economic_class_enc"] = df["economic_class"].map(econ_map).fillna(1)
    
    if "cyclone_wind_risk_class" in df.columns:
        risk_map = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
        df["cyclone_risk_enc"] = df["cyclone_wind_risk_class"].map(risk_map).fillna(1)
    
    if "tornado_risk_class" in df.columns:
        risk_map = {"Low": 0, "Moderate": 1, "High": 2}
        df["tornado_risk_enc"] = df["tornado_risk_class"].map(risk_map).fillna(0)
    
    if "koppen_geiger_zone" in df.columns:
        climate_map = {"Af": 0, "Am": 1, "Aw": 2, "Cfa": 3, "Cwa": 4, "Dfa": 5, "Dfb": 6}
        df["climate_enc"] = df["koppen_geiger_zone"].map(climate_map).fillna(0)
    
    # Access features
    if "access_to_airport" in df.columns:
        df["access_to_airport_enc"] = (df["access_to_airport"] == "Yes").astype(int)
    
    if "access_to_seaport" in df.columns:
        df["access_to_seaport_enc"] = (df["access_to_seaport"] == "Yes").astype(int)
        
    # Spatial Features: JIS Code Extraction (Japan)
    # Extracts "29000" from "29000 Nara"
    def extract_jis(val):
        if pd.isna(val): return -1
        parts = str(val).split()
        if len(parts) > 0 and parts[0].isdigit() and len(parts[0]) == 5:
            return int(parts[0])
        return -1
        
    if "geolocation_name" in df.columns:
        df["jis_code"] = df["geolocation_name"].apply(extract_jis)
        # Create interaction: is_japan * jis_code (others are -1 or 0)
        df["jis_code_clean"] = df["jis_code"].replace(-1, 0)
    
    # Fill NaN for numeric columns
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Hackathon Agent - Scientific ML Competition Solver"
    )
    parser.add_argument(
        "--experiments", 
        type=int, 
        default=5,
        help="Maximum number of experiments to run (default: 5)"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.20,
        help="Target CV score to achieve (default: 0.20)"
    )
    
    args = parser.parse_args()
    
    # Load data
    train, test = load_data()
    
    # Add features
    train = add_features(train)
    test = add_features(test)
    
    # Run intelligent agent
    agent = IntelligentAgent(
        train_df=train,
        test_df=test,
        target_col="construction_cost_per_m2_usd",
        submissions_dir=str(ROOT / "submissions"),
    )
    
    result = agent.run(
        max_experiments=args.experiments,
        target_score=args.target,
        verbose=True,
    )
    
    # Save report
    agent.save_report(str(ROOT / "intelligent_agent_report.json"))
    
    print(f"\n🏆 Best Result: CV={result['best_cv']:.5f}")
    if result["best_submission"]:
        print(f"   Submission: {result['best_submission']}")
    
    return 0 if result["best_cv"] <= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
