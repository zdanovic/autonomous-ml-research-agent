"""
Run Autonomous ML Research Agent.

This script demonstrates the new multi-agent architecture (Orchestrator -> Designer -> MLEngineer).
"""
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hackathon_agent.competition_config import CompetitionConfig
from hackathon_agent.core.orchestrator import Orchestrator


def load_data(competition: str):
    """Load data for a given competition."""
    data_dir = Path("data") / competition
    if not data_dir.exists():
        # Fallback to root data dir if competition subfolder missing
        data_dir = Path("data")
    
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test

def main():
    print("🚀 Starting Autonomous ML Research Agent...")
    
    # Load data (Assuming solafune structure for now)
    try:
        train_df, test_df = load_data("solafune")
    except FileNotFoundError:
        print("Error: content data not found in data/")
        return

    # Auto-detect config (Legacy helper, still useful)
    config = CompetitionConfig.auto_detect(train_df, test_df)
    print(f"Target: {config.target_col}")

    # Initialize Context
    context = {
        "train_df": train_df,
        "test_df": test_df,
        "target_col": config.target_col,
        "max_iterations": 3,
        "hypotheses": [],      # Shared memory for hypotheses
        "planned_experiments": [], # Shared memory for experiments
        "history": []          # specific run history
    }

    # Initialize Orchestrator
    orchestrator = Orchestrator()
    
    # Run
    result = orchestrator.run(context)
    
    print("\n✅ Research Complete!")
    print(f"Best CV Score: {result['best_score']:.4f}")

if __name__ == "__main__":
    main()
