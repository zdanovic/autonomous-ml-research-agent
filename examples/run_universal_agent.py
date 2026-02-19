#!/usr/bin/env python
"""
Universal Hackathon Agent CLI - Run on any competition automatically.

Usage:
    uv run python examples/run_universal_agent.py --competition solafune --experiments 10
    uv run python examples/run_universal_agent.py --competition wundernn --experiments 5
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


from hackathon_agent.universal_agent import UniversalAgent

from hackathon_agent.competition_config import CompetitionConfig, load_competition_data


def main():
    parser = argparse.ArgumentParser(
        description="Universal Hackathon Agent - Adapts to any competition"
    )
    parser.add_argument(
        "--competition",
        type=str,
        required=True,
        help="Competition name (e.g., 'solafune', 'wundernn')",
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=10,
        help="Maximum number of experiments to run (default: 10)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Target CV score to achieve (optional)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: 'data')",
    )
    
    args = parser.parse_args()
    
    print(f"\n🚀 Starting Universal Agent for: {args.competition}")
    print("=" * 60)
    
    # Load data
    train, test = load_competition_data(args.competition, args.data_dir)
    
    # Auto-detect configuration
    config = CompetitionConfig.auto_detect(train, test, name=args.competition)
    
    # Initialize and run agent
    agent = UniversalAgent(
        train_df=train,
        test_df=test,
        config=config,
        submissions_dir="submissions",
    )
    
    result = agent.run(
        max_experiments=args.experiments,
        target_score=args.target,
        verbose=True,
    )
    
    # Save report
    report_path = f"{args.competition}_agent_report.json"
    agent.save_report(report_path)
    
    print(f"\n🏆 Best Result: CV={result['best_cv']:.5f}")
    if result["best_submission"]:
        print(f"   Submission: {result['best_submission']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
