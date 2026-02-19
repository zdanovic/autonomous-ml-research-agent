
import sys
from pathlib import Path

# Add project root
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hackathon_agent.data_scientist import DataScientistAgent

from examples.run_intelligent_agent import add_features, load_data


def main():
    print("Loading data...")
    train, test = load_data()
    train = add_features(train)
    test = add_features(test)
    
    agent = DataScientistAgent()
    
    # Run Adversarial Validation
    print("\nRunning Adversarial Validation...")
    result = agent.verify_validation_strategy(train, test)
    
    print("\n" + "="*40)
    print(f"AUC Score: {result['auc']:.4f}")
    print(f"Status: {result['status']}")
    print(result['message'])
    print("="*40)

if __name__ == "__main__":
    main()
